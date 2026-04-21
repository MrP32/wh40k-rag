import os
import json
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from anthropic import Anthropic
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

load_dotenv()

# Config — loaded from .env (see .env.example for template)
CHROMA_PATH     = os.getenv("CHROMA_PATH",     r"C:\Projects\wh40k-app\chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "warhammer40k")
OLLAMA_URL      = os.getenv("OLLAMA_URL",      "http://127.0.0.1:11434/api/embeddings")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "nomic-embed-text")
N_RESULTS       = 40
N_FINAL         = 8

embedding_fn = OllamaEmbeddingFunction(
    url=OLLAMA_URL,
    model_name=OLLAMA_MODEL,
)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection    = chroma_client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL      = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024

app = FastAPI()

class ChatRequest(BaseModel):
    messages: list[dict]

FILTER_PROMPT = """You are a metadata extractor for a Warhammer 40,000 rules database.
Given a user query, extract structured search filters as JSON.

Available doc_type values (pick the ONE that best matches):
  combat_patrol            — a specific named combat patrol box
  combat_patrol_rules      — the universal combat patrol rulebook
  faction_pack             — a faction's rules pack (detachments, stratagems, datasheets)
  core_rules               — the main core rules
  core_rules_updates       — rolling patch document to core rules
  core_rules_quickstart    — beginner's short rules
  points_costs             — the Munitorum Field Manual (points for all factions)
  balance_dataslate        — the rolling balance patch
  crusade_rules            — narrative/campaign rules
  boarding_actions_rules   — boarding-actions game mode
  tournament_rules         — Chapter Approved / Pariah Nexus tournament packs
  imperial_armour          — Imperial Armour / Forge World rules
  army_roster              — army roster templates
  other                    — anything else

Available subject values (always lowercase, codex-faction level — NEVER a
sub-chapter like 'ultramarines' or a patrol box name):
  space marines, grey knights, blood angels, dark angels, black templars,
  space wolves, deathwatch, adepta sororitas, adeptus custodes,
  adeptus mechanicus, astra militarum, imperial knights, imperial agents,
  chaos space marines, death guard, thousand sons, world eaters,
  emperor's children, chaos knights, chaos daemons,
  aeldari, drukhari, genestealer cults, leagues of votann, necrons, orks,
  t'au empire, tyranids,
  core rules, munitorum field manual, balance dataslate, combat patrol rules,
  crusade rules, boarding actions

Output fields (all OPTIONAL — omit any you cannot confidently determine):
  doc_type           — from the list above
  subject            — from the list above; always the codex faction or the
                       rulebook name, NEVER a patrol box, stratagem, or unit name
  patrol_name        — ONLY for a specific named combat patrol box, e.g.
                       "aurellios banishers". Use together with subject.
  munitorum_faction  — ONLY for points-cost queries; the faction whose points
                       the user wants, e.g. "grey knights" for "Nemesis
                       Dreadknight points"

Return ONLY valid JSON. Return {} if no field can be confidently determined.

Examples:
  "Grey Knights stratagems"
    -> {"subject": "grey knights", "doc_type": "faction_pack"}
  "Librarius Conclave rules"
    -> {"subject": "space marines", "doc_type": "faction_pack"}
  "How does overwatch work?"
    -> {"doc_type": "core_rules"}
  "Points cost for a Leman Russ"
    -> {"doc_type": "points_costs", "munitorum_faction": "astra militarum"}
  "How many points is a Nemesis Dreadknight?"
    -> {"doc_type": "points_costs", "munitorum_faction": "grey knights"}
  "Aurellios Banishers combat patrol"
    -> {"subject": "grey knights", "patrol_name": "aurellios banishers", "doc_type": "combat_patrol"}
  "Teleport Assault rule"
    -> {"subject": "grey knights"}
  "What is a stratagem?"
    -> {}
"""

def extract_filters(query: str) -> dict:
    """
    Use Claude to extract metadata filters from a natural-language query.
    Returns a ChromaDB `where` dict or {} on any failure / no extractable
    fields. Never raises.
    """
    try:
        resp = anthropic_client.messages.create(
            model=MODEL,
            max_tokens=200,   # bumped from 100 to fit larger JSON responses
            messages=[{"role": "user", "content": f"{FILTER_PROMPT}\n\nQuery: {query}"}]
        )
        data = json.loads(resp.content[0].text.strip())
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    # Build one $eq clause per extracted field. Field names here must match
    # the metadata keys written by ingest.py's flatten_chunk_metadata().
    filters = []
    for field in ("subject", "doc_type", "patrol_name", "munitorum_faction"):
        val = data.get(field)
        if isinstance(val, str) and val.strip():
            filters.append({field: {"$eq": val.strip().lower()}})

    if not filters:
        return {}
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}

def _subject_from_filter(where: dict) -> str | None:
    """Pull the subject value out of a filter dict, whether flat or nested in $and."""
    if not isinstance(where, dict):
        return None
    if "subject" in where:
        val = where["subject"]
        if isinstance(val, str):
            return val
        if isinstance(val, dict) and "$eq" in val:
            return val["$eq"]
    for clause in where.get("$and", []) or []:
        s = _subject_from_filter(clause)
        if s:
            return s
    return None


def _chroma_query(query: str, where):
    """Run one ChromaDB query. Returns (chunks, metas) — possibly empty."""
    try:
        if where:
            r = collection.query(query_texts=[query], n_results=N_RESULTS, where=where)
        else:
            r = collection.query(query_texts=[query], n_results=N_RESULTS)
    except Exception:
        return [], []
    return (r.get("documents", [[]])[0] or [],
            r.get("metadatas", [[]])[0] or [])


def search_context(query: str) -> str:
    """
    Retrieve context for the query using a three-tier fallback:
      1. exact filter        — best precision
      2. subject-only filter — if tier 1 returns nothing
      3. unfiltered          — if tiers 1-2 return nothing
    """
    where = extract_filters(query)

    # Tier 1: exact
    chunks, metas = _chroma_query(query, where) if where else _chroma_query(query, None)

    # Tier 2: subject-only
    if not chunks and where:
        subject = _subject_from_filter(where)
        if subject:
            chunks, metas = _chroma_query(query, {"subject": {"$eq": subject}})

    # Tier 3: unfiltered
    if not chunks:
        chunks, metas = _chroma_query(query, None)

    # Drop tiny noise chunks; cap to N_FINAL most relevant
    filtered = [(c, m) for c, m in zip(chunks, metas) if len(c.split()) >= 5][:N_FINAL]
    return "\n\n".join(
        f"[{(m.get('source') or 'unknown')}]\n{c}" for c, m in filtered
    )

@app.post("/chat")
async def chat(request: ChatRequest):
    user_message = request.messages[-1]["content"]
    context      = search_context(user_message)
    system_prompt = f"""You are a Warhammer 40,000 rules expert assistant.
Answer using only the context provided below. If the answer is not in the context, say so clearly.
Always cite the source PDF name when referencing rules or stats.

CONTEXT:
{context}"""

    def stream():
        with anthropic_client.messages.stream(
            model=MODEL, max_tokens=MAX_TOKENS,
            system=system_prompt, messages=request.messages,
        ) as s:
            for text in s.text_stream:
                yield f"data: {json.dumps({'text': text})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")

@app.get("/db-info")
async def db_info():
    total    = collection.count()
    all_meta = collection.get(include=["metadatas"])
    sources  = sorted({m.get("source", "unknown") for m in all_meta["metadatas"]})
    return {"total_chunks": total, "sources": sources}

app.mount("/", StaticFiles(directory="static", html=True), name="static")

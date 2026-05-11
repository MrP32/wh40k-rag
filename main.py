import os
import json
import signal
import threading
import time
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from anthropic import Anthropic
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

load_dotenv()

# ─── Config — loaded from .env (see .env.example for template) ───────────────
CHROMA_PATH = os.getenv("CHROMA_PATH", r"C:\Projects\wh40k-app\chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "warhammer40k")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/embeddings")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text")
PDF_FOLDER = os.getenv("PDF_FOLDER", r"C:\Personal Projects\warhammer_40k_pdfs")
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6")

N_RESULTS = 40
N_FINAL = 8

embedding_fn = OllamaEmbeddingFunction(
    url=OLLAMA_URL,
    model_name=OLLAMA_MODEL,
)

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)

anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = ANTHROPIC_MODEL
MAX_TOKENS = 1024

app = FastAPI()


class ChatRequest(BaseModel):
    messages: list[dict]


FILTER_PROMPT = """You are a metadata extractor for a Warhammer 40,000 rules database.
Extract structured search filters from the user's query and return them as JSON.

Your job is to be HELPFUL, not cautious. If the query clearly refers to a
specific faction, rulebook, or document type, extract that information even
if the wording is colloquial, informal, or uses synonyms. Partial filters
are valuable: extracting just "subject" for a query like "tell me about
Stormboyz" is much better than returning {} and falling back to unfiltered
search.

Only return {} when the query is genuinely generic (e.g. "what is a
stratagem?", "how does the game work?") with no extractable faction,
rulebook, or document-type signal.

=============================================================================
AVAILABLE doc_type VALUES — pick the ONE that best matches
=============================================================================
combat_patrol            a specific named combat patrol box
combat_patrol_rules      the universal combat patrol rulebook
faction_pack             a faction's rules (detachments, stratagems, datasheets)
core_rules               the main core rules
core_rules_updates       rolling patch document to core rules
core_rules_quickstart    beginner's short rules
points_costs             the Munitorum Field Manual (points for all factions)
balance_dataslate        the rolling balance patch
crusade_rules            narrative/campaign rules
boarding_actions_rules   boarding-actions game mode
tournament_rules         Chapter Approved / Pariah Nexus tournament packs
imperial_armour          Imperial Armour / Forge World rules
army_roster              army roster templates
other                    anything else

=============================================================================
AVAILABLE subject VALUES — always lowercase, always the codex faction
=============================================================================
NEVER use sub-chapters like 'ultramarines', 'raven guard', 'iron hands',
'salamanders', 'white scars', 'imperial fists' — those all roll up to
'space marines'.

space marines, grey knights, blood angels, dark angels, black templars,
space wolves, deathwatch, adepta sororitas, adeptus custodes,
adeptus mechanicus, astra militarum, imperial knights, imperial agents,
chaos space marines, death guard, thousand sons, world eaters,
emperor's children, chaos knights, chaos daemons, aeldari, drukhari,
genestealer cults, leagues of votann, necrons, orks, t'au empire, tyranids,
core rules, munitorum field manual, balance dataslate, combat patrol rules,
crusade rules, boarding actions

=============================================================================
OUTPUT FIELDS (all optional)
=============================================================================
doc_type             from the doc_type list above
subject              from the subject list above
patrol_name          ONLY for a specific named combat patrol box, e.g.
                     "aurellios banishers". Use together with subject.
munitorum_faction    ONLY for points-cost queries; the faction whose points
                     the user wants, e.g. "grey knights"

=============================================================================
ROUTING GUIDE — when you see these signals, pick these fields
=============================================================================
POINTS-COST queries — ALWAYS set doc_type=points_costs AND munitorum_faction.
  Signals: "points", "cost", "pts", "how many points", "how much does X cost",
  "what's the point value", "how expensive", "points value", "points for",
  "cost of", "price of", numeric questions about army-building costs.
  The munitorum_faction is the faction the *unit* belongs to, not the player.

FACTION-RULES queries — set subject, and usually doc_type=faction_pack.
  Signals: a faction name + any of: "stratagems", "detachment", "detachment
  rule", "enhancements", "army rule", "datasheet", "weapons", "abilities",
  "keyword", "what does X do". Also: named detachments ("Librarius Conclave",
  "Warpbane Task Force") → their parent faction, doc_type=faction_pack.
  Named universal abilities ("Teleport Assault", "Oath of Moment") → their
  parent faction (subject only; skip doc_type if unsure).

COMBAT-PATROL queries — set subject, patrol_name, doc_type=combat_patrol.
  Signals: "combat patrol" + faction, or a specific patrol box name
  ("Aurellios Banishers", "Warpbane Task Force" is NOT a patrol — it's a
  detachment). Patrol names typically end in "Banishers", "Host", "Cadre",
  "Guardians", "Brood", "Strike Team", "Kill Team", etc.

CORE-RULES queries — set doc_type=core_rules.
  Signals: universal rules language ("overwatch", "charge", "morale",
  "objective control", "how does X phase work", "what's the rule for X").

UNIT-BY-NAME queries — set subject if the unit is unambiguous.
  "Stormboyz" → orks. "Nemesis Dreadknight" → grey knights. "Leman Russ"
  → astra militarum. "Wraithlord" → aeldari.

=============================================================================
EXAMPLES — showing PHRASING VARIANTS for the same underlying intent
=============================================================================

Points costs (all these must extract both doc_type AND munitorum_faction):
  "Points cost for a Leman Russ"
    -> {"doc_type": "points_costs", "munitorum_faction": "astra militarum"}
  "How many points is a Nemesis Dreadknight?"
    -> {"doc_type": "points_costs", "munitorum_faction": "grey knights"}
  "How much does a Carnifex cost?"
    -> {"doc_type": "points_costs", "munitorum_faction": "tyranids"}
  "What's the points value of a Ghazghkull?"
    -> {"doc_type": "points_costs", "munitorum_faction": "orks"}
  "Baneblade pts"
    -> {"doc_type": "points_costs", "munitorum_faction": "astra militarum"}
  "cost of a wraithknight"
    -> {"doc_type": "points_costs", "munitorum_faction": "aeldari"}

Faction rules (all extract both subject AND doc_type):
  "Grey Knights stratagems"
    -> {"subject": "grey knights", "doc_type": "faction_pack"}
  "What are the Grey Knights stratagems?"
    -> {"subject": "grey knights", "doc_type": "faction_pack"}
  "Show me Tyranid enhancements"
    -> {"subject": "tyranids", "doc_type": "faction_pack"}
  "Librarius Conclave detachment rule"
    -> {"subject": "space marines", "doc_type": "faction_pack"}
  "Warpbane Task Force"
    -> {"subject": "grey knights", "doc_type": "faction_pack"}
  "What does Oath of Moment do?"
    -> {"subject": "space marines", "doc_type": "faction_pack"}

Unit by name (subject only when doc_type is ambiguous):
  "Teleport Assault rule"
    -> {"subject": "grey knights"}
  "What weapons does a Strike Squad have?"
    -> {"subject": "grey knights", "doc_type": "faction_pack"}
  "Tell me about Stormboyz"
    -> {"subject": "orks"}

Combat patrols (subject + patrol_name + doc_type):
  "Aurellios Banishers combat patrol"
    -> {"subject": "grey knights", "patrol_name": "aurellios banishers", "doc_type": "combat_patrol"}
  "Grey Knights combat patrol"
    -> {"subject": "grey knights", "doc_type": "combat_patrol"}
  "Sanctuary Guardians"
    -> {"subject": "adepta sororitas", "patrol_name": "sanctuary guardians", "doc_type": "combat_patrol"}

Core rules:
  "How does overwatch work?"
    -> {"doc_type": "core_rules"}
  "What's the rule for charging?"
    -> {"doc_type": "core_rules"}

Genuinely generic — these return {}:
  "What is a stratagem?"
    -> {}
  "How do I play Warhammer 40k?"
    -> {}
  "What's the best faction?"
    -> {}

Return ONLY valid JSON. No preamble, no code fences, no explanation.
"""


def extract_filters(query: str) -> tuple[dict, dict]:
    """
    Use Claude to extract metadata filters from a natural-language query.

    Returns a (where, raw) tuple:
      where -- ChromaDB-formatted filter dict, or {} on failure / no fields.
      raw   -- the raw extracted fields keyed by name (used for telemetry).

    Never raises.
    """
    raw: dict = {}
    try:
        resp = anthropic_client.messages.create(
            model=MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": f"{FILTER_PROMPT}\n\nQuery: {query}"}]
        )
        data = json.loads(resp.content[0].text.strip())
    except Exception:
        return {}, raw

    if not isinstance(data, dict):
        return {}, raw

    filters = []
    for field in ("subject", "doc_type", "patrol_name", "munitorum_faction"):
        val = data.get(field)
        if isinstance(val, str) and val.strip():
            normalized = val.strip().lower()
            filters.append({field: {"$eq": normalized}})
            raw[field] = normalized

    if not filters:
        return {}, raw
    if len(filters) == 1:
        return filters[0], raw
    return {"$and": filters}, raw


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


def search_context(query: str):
    """
    Retrieve context for the query using a three-tier fallback:
      1. exact filter        — best precision
      2. subject-only filter — if tier 1 returns nothing
      3. unfiltered          — if tiers 1-2 return nothing

    Returns (context_string, source_records, telemetry) where
      source_records is a list of { "filename": str, "page": int, "section": str }
      telemetry is a dict of routing/timing info for the UI panel.
    """
    where, raw_filters = extract_filters(query)

    tier_fired = None
    candidates: list = []
    metas: list = []

    # Tier 1: exact
    if where:
        candidates, metas = _chroma_query(query, where)
        if candidates:
            tier_fired = 1

    # Tier 2: subject-only
    if not candidates and where:
        subject = _subject_from_filter(where)
        if subject:
            candidates, metas = _chroma_query(query, {"subject": {"$eq": subject}})
            if candidates:
                tier_fired = 2

    # Tier 3: unfiltered
    if not candidates:
        candidates, metas = _chroma_query(query, None)
        if candidates:
            tier_fired = 3

    # Drop tiny noise chunks; cap to N_FINAL most relevant
    filtered = [(c, m) for c, m in zip(candidates, metas) if len(c.split()) >= 5][:N_FINAL]

    context = "\n\n".join(
        f"[{(m.get('source') or 'unknown')}]\n{c}" for c, m in filtered
    )

    # Build deduplicated source records for the UI. Use Path().name so the
    # raw filesystem path stored in 'source' is reduced to just the PDF name.
    seen = {}
    for _, m in filtered:
        raw = m.get("source") or "unknown"
        fname = Path(raw).name if raw != "unknown" else "unknown"
        page = int(m.get("page_number") or 0)
        section = (m.get("section_identifier_clean") or m.get("section_identifier") or "").strip()
        key = (fname, page)
        if key not in seen:
            seen[key] = {"filename": fname, "page": page, "section": section}
    sources = list(seen.values())

    telemetry = {
        "filters": raw_filters,
        "tier_fired": tier_fired,
        "candidates": len(candidates),
        "selected": len(filtered),
        "unique_tomes": len(sources),
    }

    return context, sources, telemetry


@app.post("/chat")
async def chat(request: ChatRequest):
    user_message = request.messages[-1]["content"]
    t_start = time.perf_counter()
    context, sources, telemetry = search_context(user_message)

    system_prompt = f"""You are a Warhammer 40,000 rules expert assistant.
Answer using only the context provided below. If the answer is not in the context, say so clearly.
Always cite the source PDF name when referencing rules or stats.

CONTEXT:
{context}"""

    def stream():
        output_chars = 0
        with anthropic_client.messages.stream(
            model=MODEL, max_tokens=MAX_TOKENS,
            system=system_prompt, messages=request.messages,
        ) as s:
            for text in s.text_stream:
                output_chars += len(text)
                yield f"data: {json.dumps({'text': text})}\n\n"

        # Compute final telemetry now that generation is done. Token count
        # is approximate (chars / 4 is the conventional rough estimate);
        # we surface it as "~N tokens" in the UI to be honest about that.
        duration_ms = int((time.perf_counter() - t_start) * 1000)
        telemetry["duration_ms"] = duration_ms
        telemetry["model"] = MODEL
        telemetry["output_tokens"] = f"~{max(1, output_chars // 4)}"

        # Emit sources first, then telemetry, then DONE. Order matters only
        # for renderer simplicity — the UI handles both fields whichever
        # arrives first.
        yield f"data: {json.dumps({'sources': sources})}\n\n"
        yield f"data: {json.dumps({'telemetry': telemetry})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/pdf/{filename}")
async def get_pdf(filename: str):
    """
    Serve a PDF from PDF_FOLDER inline so the browser renders it natively.
    Path-traversal protection: only the leaf filename is honored, and the
    resolved path must stay under PDF_FOLDER.
    """
    # Reject anything that looks like a path. We only accept a leaf filename.
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    base = Path(PDF_FOLDER).resolve()
    target = (base / filename).resolve()

    # Defence in depth: make sure resolved path is still under PDF_FOLDER
    try:
        target.relative_to(base)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="PDF not found")
    if target.suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Not a PDF")

    return FileResponse(
        target,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="{filename}"'},
    )


@app.get("/db-info")
async def db_info():
    total = collection.count()
    all_meta = collection.get(include=["metadatas"])
    sources = sorted({m.get("source", "unknown") for m in all_meta["metadatas"]})
    return {"total_chunks": total, "sources": sources}


@app.post("/shutdown")
async def shutdown():
    """
    Graceful shutdown triggered by the in-app shutdown button.

    We return a 200 response immediately, then schedule the actual process
    exit on a background thread with a short delay. This gives the browser
    time to receive the response and render the goodbye state before the
    connection drops.

    The launcher (launch.bat) detects the uvicorn exit and runs its own
    cleanup — stopping Ollama if it started it.
    """
    def _delayed_exit():
        time.sleep(0.5)
        # SIGTERM gives uvicorn a chance to close gracefully on POSIX.
        # On Windows, signal.SIGTERM behaves like SIGINT and uvicorn
        # handles it. Fallback to os._exit if signals are unavailable
        # for any reason.
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        except Exception:
            os._exit(0)

    threading.Thread(target=_delayed_exit, daemon=True).start()
    return {"status": "shutting_down"}


app.mount("/", StaticFiles(directory="static", html=True), name="static")

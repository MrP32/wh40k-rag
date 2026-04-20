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

CHROMA_PATH     = r"C:\Projects\wh40k-app\chroma_db"
COLLECTION_NAME = "warhammer40k"
N_RESULTS       = 40
N_FINAL         = 8

embedding_fn = OllamaEmbeddingFunction(
    url="http://127.0.0.1:11434/api/embeddings",
    model_name="nomic-embed-text"
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

Available content_category values:
  combat_patrol, faction_rules, core_rules, core_rules_updates,
  balance_rules, points_costs, tournament_rules, crusade_rules,
  boarding_actions_rules, imperial_armour, army_roster, other

Available subject values (lowercase):
  space marines, grey knights, blood angels, dark angels, ultramarines,
  raven guard, iron hands, salamanders, white scars, imperial fists,
  black templars, deathwatch, adeptus custodes, adeptus mechanicus,
  adeptus titanicus, astra militarum, adepta sororitas, chaos space marines,
  death guard, thousand sons, world eaters, chaos daemons, necrons, orks,
  tyranids, genestealer cults, tau empire, aeldari, drukhari, harlequins,
  leagues of votann, imperial knights, chaos knights, titans,
  balance dataslate, munitorum field manual, core rules,
  combat patrol rules, crusade rules, boarding actions companion

Return ONLY valid JSON with optional fields: subject, category.
Omit any field you cannot confidently determine. Return {} if unsure.

Examples:
  "Grey Knights stratagems" -> {"subject": "grey knights", "category": "faction_rules"}
  "Librarius Conclave rules" -> {"subject": "space marines", "category": "faction_rules"}
  "How does overwatch work?" -> {"category": "core_rules"}
  "Points cost for a Leman Russ" -> {"category": "points_costs"}
  "Aurellios Banishers combat patrol" -> {"subject": "grey knights - aurellios banishers", "category": "combat_patrol"}
  "What is a stratagem?" -> {}
"""

def extract_filters(query: str) -> dict:
    try:
        resp = anthropic_client.messages.create(
            model=MODEL,
            max_tokens=100,
            messages=[{"role": "user", "content": f"{FILTER_PROMPT}\n\nQuery: {query}"}]
        )
        data    = json.loads(resp.content[0].text.strip())
        filters = []
        if "subject" in data:
            filters.append({"subject": {"$eq": data["subject"]}})
        if "category" in data:
            filters.append({"content_category": {"$eq": data["category"]}})
        if len(filters) == 2:
            return {"$and": filters}
        elif len(filters) == 1:
            return filters[0]
    except Exception:
        pass
    return {}

def search_context(query: str) -> str:
    where = extract_filters(query)
    try:
        results = collection.query(
            query_texts=[query], n_results=N_RESULTS, where=where
        ) if where else collection.query(
            query_texts=[query], n_results=N_RESULTS
        )
    except Exception:
        results = collection.query(query_texts=[query], n_results=N_RESULTS)

    chunks   = results["documents"][0]
    metas    = results["metadatas"][0]
    filtered = [(c, m) for c, m in zip(chunks, metas) if len(c.split()) >= 5][:N_FINAL]
    sources  = [m.get("source", "unknown") for _, m in filtered]
    return "\n\n".join(f"[{sources[i]}]\n{filtered[i][0]}" for i in range(len(filtered)))

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
    sources  = sorted(set(m["source"] for m in all_meta["metadatas"]))
    return {"total_chunks": total, "sources": sources}

app.mount("/", StaticFiles(directory="static", html=True), name="static")

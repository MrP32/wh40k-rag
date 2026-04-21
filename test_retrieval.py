"""
test_retrieval.py
=================
Retrieval quality test suite for the Warhammer 40K RAG system.
Location: C:\Projects\wh40k-app\test_retrieval.py
Command:  python test_retrieval.py

Tests cover:
  - Faction-specific queries (combat patrol, faction rules)
  - General rules queries (core rules, stratagems)
  - Specific unit/detachment lookups
  - Points/balance queries
  - Filter extraction accuracy
"""

import os
import json
from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from anthropic import Anthropic

load_dotenv()

# ── Config ────────────────────────────────────────────────────
CHROMA_PATH     = r"C:\Projects\wh40k-app\chroma_db"
COLLECTION_NAME = "warhammer40k"
N_RESULTS       = 40
N_FINAL         = 8

embedding_fn = OllamaEmbeddingFunction(
    url="http://127.0.0.1:11434/api/embeddings",
    model_name="nomic-embed-text"
)
client     = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ── Filter extraction (mirrors main.py) ──────────────────────
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
            model="claude-sonnet-4-20250514",
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
            return {"$and": filters}, data
        elif len(filters) == 1:
            return filters[0], data
    except Exception:
        pass
    return {}, {}

def run_query(query: str) -> list:
    where, filter_data = extract_filters(query)
    try:
        results = collection.query(
            query_texts=[query], n_results=N_RESULTS, where=where
        ) if where else collection.query(
            query_texts=[query], n_results=N_RESULTS
        )
    except Exception:
        results = collection.query(query_texts=[query], n_results=N_RESULTS)

    chunks    = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    filtered  = [
        (c, m, d) for c, m, d in zip(chunks, metadatas, distances)
        if len(c.split()) >= 5
    ][:N_FINAL]
    return filtered, filter_data

def print_results(query, results, filter_data):
    print(f"\n{'='*65}")
    print(f"  QUERY: {query}")
    print(f"  FILTERS EXTRACTED: {json.dumps(filter_data)}")
    print(f"{'='*65}")
    if not results:
        print("  ✗ No results returned")
        return
    for i, (chunk, meta, dist) in enumerate(results):
        print(f"\n  ── Result {i+1} | dist={dist:.4f} | "
              f"section={meta.get('section_type','?')} | "
              f"words={len(chunk.split())}")
        print(f"  Source:  {meta.get('source','?').split(chr(92))[-1]}")
        print(f"  Subject: {meta.get('subject','?')}")
        print(f"  Text:    {chunk[:250].replace(chr(10), ' ')}")

# ── Test cases ────────────────────────────────────────────────
TEST_CASES = [
    # Faction-specific — combat patrol
    "What can you tell me about the Grey Knights combat patrol?",
    "Tell me about the Aurellios Banishers",

    # Faction-specific — faction rules
    "What is the Librarius Conclave detachment rule?",
    "What are the Grey Knights stratagems?",
    "What is the Warpbane Task Force detachment?",

    # General rules
    "How does the Overwatch stratagem work?",
    "What is the Teleport Assault ability?",

    # Points / balance
    "What are the points costs for a Nemesis Dreadknight?",

    # Cross-faction
    "What weapons does a Strike Squad have?",
]

if __name__ == "__main__":
    print(f"\nWarhammer 40K RAG — Retrieval Test Suite")
    print(f"Collection: {COLLECTION_NAME} ({collection.count():,} chunks)")
    print(f"Running {len(TEST_CASES)} test cases...\n")

    for query in TEST_CASES:
        results, filter_data = run_query(query)
        print_results(query, results, filter_data)

    print(f"\n{'='*65}")
    print("  Test suite complete.")
    print(f"{'='*65}\n")
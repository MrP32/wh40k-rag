"""
test_retrieval.py
=================
Retrieval quality test suite for the Warhammer 40K RAG system.
Location: C:\\Projects\\wh40k-app\\test_retrieval.py
Command:  python test_retrieval.py    (or: .\\deploy.ps1 -Action test ...)

Two jobs:
  1. Diagnostic: run a curated set of queries and print what came back, so a
     human can eyeball whether the retrieved chunks look reasonable.
  2. Gate: assert the four target improvements from HANDOFF_v2 actually hold
     against the live collection. Non-zero exit on any failure — deploy.ps1
     -Action test treats this as a pass/fail signal.

The four target assertions:
  A. Grey Knights combat patrol queries return combat_patrol chunks for
     subject=grey knights (used to return 0 under the old content_category
     filter).
  B. "Librarius Conclave" returns space marines chunks, and does NOT bleed
     "Spearpoint Paragon" (Grey Knights) text into the top results.
  C. "Nemesis Dreadknight points" returns points_costs chunks tagged with
     munitorum_faction=grey knights.
  D. "Teleport Assault" returns subject=grey knights chunks.

Note: this file intentionally duplicates extract_filters / _subject_from_filter
/ _chroma_query / search logic from main.py rather than importing them.
main.py constructs a FastAPI app and mounts static files at import time, which
would break this script. If main.py's FILTER_PROMPT or fallback logic changes,
this test MUST be updated in lockstep — otherwise it tests a different path
than users hit.
"""

import os
import sys
import json
from pathlib import Path

from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from anthropic import Anthropic

load_dotenv()

# ---------------------------------------------------------------------------
# Config — mirrors main.py. Keep in sync.
# ---------------------------------------------------------------------------

CHROMA_PATH     = os.getenv("CHROMA_PATH",     r"C:\Projects\wh40k-app\chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "warhammer40k")
OLLAMA_URL      = os.getenv("OLLAMA_URL",      "http://127.0.0.1:11434/api/embeddings")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "nomic-embed-text")
MODEL           = "claude-sonnet-4-6"
N_RESULTS       = 40
N_FINAL         = 8

embedding_fn = OllamaEmbeddingFunction(url=OLLAMA_URL, model_name=OLLAMA_MODEL)
client       = chromadb.PersistentClient(path=CHROMA_PATH)
collection   = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# Filter prompt — MUST stay in sync with main.py::FILTER_PROMPT
# ---------------------------------------------------------------------------

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
  doc_type           from the doc_type list above
  subject            from the subject list above
  patrol_name        ONLY for a specific named combat patrol box, e.g.
                     "aurellios banishers". Use together with subject.
  munitorum_faction  ONLY for points-cost queries; the faction whose points
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


# ---------------------------------------------------------------------------
# Retrieval — mirrors main.py exactly, plus a retrieval_tier report
# ---------------------------------------------------------------------------

def extract_filters(query: str) -> tuple[dict, dict]:
    """
    Returns (where_clause, raw_extracted_dict). raw_extracted_dict is the
    JSON Claude actually produced — useful when an assertion fails and we
    want to know whether the filter itself was wrong vs. the retrieval.
    """
    try:
        resp = anthropic_client.messages.create(
            model=MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": f"{FILTER_PROMPT}\n\nQuery: {query}"}],
        )
        data = json.loads(resp.content[0].text.strip())
    except Exception:
        return {}, {}

    if not isinstance(data, dict):
        return {}, {}

    filters = []
    for field in ("subject", "doc_type", "patrol_name", "munitorum_faction"):
        val = data.get(field)
        if isinstance(val, str) and val.strip():
            filters.append({field: {"$eq": val.strip().lower()}})

    if not filters:
        return {}, data
    if len(filters) == 1:
        return filters[0], data
    return {"$and": filters}, data


def _subject_from_filter(where: dict):
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
    try:
        if where:
            r = collection.query(query_texts=[query], n_results=N_RESULTS, where=where)
        else:
            r = collection.query(query_texts=[query], n_results=N_RESULTS)
    except Exception:
        return [], [], []
    return (r.get("documents", [[]])[0] or [],
            r.get("metadatas", [[]])[0] or [],
            r.get("distances", [[]])[0] or [])


def run_query(query: str) -> dict:
    """
    Three-tier fallback. Reports which tier actually produced the result set.
    "exact" covers both a real filter matching AND an empty-filter pass-through
    (i.e. the query produced {} and we went unfiltered from the start). If you
    need to distinguish those, check the filter_data field.
    """
    where, filter_data = extract_filters(query)

    # Tier 1
    chunks, metas, dists = _chroma_query(query, where) if where else _chroma_query(query, None)
    tier = "exact"

    # Tier 2
    if not chunks and where:
        subject = _subject_from_filter(where)
        if subject:
            chunks, metas, dists = _chroma_query(query, {"subject": {"$eq": subject}})
            if chunks:
                tier = "subject"

    # Tier 3
    if not chunks:
        chunks, metas, dists = _chroma_query(query, None)
        tier = "unfiltered"

    # Drop tiny noise and cap
    triples = [
        (c, m, d) for c, m, d in zip(chunks, metas, dists) if len(c.split()) >= 5
    ][:N_FINAL]
    return {
        "query":           query,
        "filter_data":     filter_data,
        "where":           where,
        "retrieval_tier":  tier,
        "results":         triples,
    }


# ---------------------------------------------------------------------------
# Diagnostic output
# ---------------------------------------------------------------------------

def print_results(r: dict) -> None:
    print(f"\n{'=' * 70}")
    print(f"  QUERY: {r['query']}")
    print(f"  FILTER JSON : {json.dumps(r['filter_data'])}")
    print(f"  WHERE CLAUSE: {json.dumps(r['where'])}")
    print(f"  TIER FIRED  : {r['retrieval_tier']}")
    print(f"{'=' * 70}")
    if not r["results"]:
        print("  (no results)")
        return
    for i, (chunk, meta, dist) in enumerate(r["results"], 1):
        source = meta.get("source", "?")
        # Windows paths — show just the filename
        source_short = source.split("\\")[-1].split("/")[-1]
        print(
            f"\n  -- Result {i} | dist={dist:.4f} | "
            f"type={meta.get('section_type', '?')} | "
            f"doc_type={meta.get('doc_type', '?')}"
        )
        print(f"     source  : {source_short}  (p.{meta.get('page_number', '?')})")
        print(f"     subject : {meta.get('subject', '?')}")
        if meta.get("patrol_name"):
            print(f"     patrol  : {meta['patrol_name']}")
        if meta.get("munitorum_faction"):
            print(f"     muni    : {meta['munitorum_faction']}")
        heading = meta.get("section_identifier", "")
        if heading:
            print(f"     heading : {heading}")
        preview = chunk[:240].replace("\n", " ")
        print(f"     text    : {preview}")


# ---------------------------------------------------------------------------
# Diagnostic queries — kept for eyeballing
# ---------------------------------------------------------------------------

DIAGNOSTIC_QUERIES = [
    # Combat patrol
    "What can you tell me about the Grey Knights combat patrol?",
    "Tell me about the Aurellios Banishers",
    # Faction rules
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


# ---------------------------------------------------------------------------
# Assertions — the four target improvements from HANDOFF_v2
# ---------------------------------------------------------------------------

class AssertionFailure(Exception):
    pass


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionFailure(msg)


def assert_A_grey_knights_combat_patrol():
    """
    At least one top-N_FINAL result should be a grey-knights combat_patrol
    chunk. Previously this query returned 0 under content_category='faction_rules'.
    """
    print("\n>>> Assertion A: Grey Knights combat patrol retrieval")
    r = run_query("What can you tell me about the Grey Knights combat patrol?")
    print_results(r)
    _assert(bool(r["results"]), "  FAIL: zero results returned")
    matches = [
        m for (_, m, _) in r["results"]
        if m.get("subject") == "grey knights" and m.get("doc_type") == "combat_patrol"
    ]
    _assert(
        len(matches) >= 1,
        f"  FAIL: no result had subject=grey knights AND doc_type=combat_patrol "
        f"(got subjects={[m.get('subject') for _, m, _ in r['results']]}, "
        f"doc_types={[m.get('doc_type') for _, m, _ in r['results']]})",
    )
    print(f"  PASS: {len(matches)}/{len(r['results'])} results match")


def assert_B_librarius_conclave_no_bleed():
    """
    "Librarius Conclave" is a Space Marines detachment. Under the old
    carry-forward bug, Grey Knights 'Spearpoint Paragon' heading bled onto
    many unrelated chunks and sometimes outranked the real answer.
    """
    print("\n>>> Assertion B: Librarius Conclave — no Spearpoint Paragon bleed")
    r = run_query("What is the Librarius Conclave detachment rule?")
    print_results(r)
    _assert(bool(r["results"]), "  FAIL: zero results returned")

    top5 = r["results"][:5]
    sm_count = sum(1 for (_, m, _) in top5 if m.get("subject") == "space marines")
    spearpoint_bleed = [
        (c, m) for (c, m, _) in top5 if "spearpoint paragon" in (c or "").lower()
    ]

    _assert(
        sm_count >= 3,
        f"  FAIL: only {sm_count}/5 top results are space marines subject "
        f"(got {[m.get('subject') for _, m, _ in top5]})",
    )
    _assert(
        not spearpoint_bleed,
        f"  FAIL: 'Spearpoint Paragon' text bled into {len(spearpoint_bleed)} top-5 results",
    )
    print(f"  PASS: {sm_count}/5 space marines in top-5, 0 Spearpoint Paragon bleed")


def assert_C_nemesis_dreadknight_points():
    """
    Nemesis Dreadknight points should retrieve Munitorum chunks tagged
    with munitorum_faction=grey knights.
    """
    print("\n>>> Assertion C: Nemesis Dreadknight points from Munitorum")
    r = run_query("How many points is a Nemesis Dreadknight?")
    print_results(r)
    _assert(bool(r["results"]), "  FAIL: zero results returned")
    matches = [
        m for (_, m, _) in r["results"]
        if m.get("doc_type") == "points_costs"
        and m.get("munitorum_faction") == "grey knights"
    ]
    _assert(
        len(matches) >= 1,
        f"  FAIL: no result had doc_type=points_costs AND munitorum_faction=grey knights "
        f"(got doc_types={[m.get('doc_type') for _, m, _ in r['results']]}, "
        f"muni={[m.get('munitorum_faction') for _, m, _ in r['results']]})",
    )
    print(f"  PASS: {len(matches)}/{len(r['results'])} Munitorum-Grey Knights matches")


def assert_D_teleport_assault_is_grey_knights():
    """
    "Teleport Assault" is the Grey Knights universal keyword ability. Should
    route to grey knights subject.
    """
    print("\n>>> Assertion D: Teleport Assault → Grey Knights")
    r = run_query("What is the Teleport Assault ability?")
    print_results(r)
    _assert(bool(r["results"]), "  FAIL: zero results returned")
    gk_count = sum(1 for (_, m, _) in r["results"] if m.get("subject") == "grey knights")
    _assert(
        gk_count >= 1,
        f"  FAIL: no result had subject=grey knights "
        f"(got {[m.get('subject') for _, m, _ in r['results']]})",
    )
    print(f"  PASS: {gk_count}/{len(r['results'])} Grey Knights matches")


ASSERTIONS = [
    ("A - GK combat patrol retrieves",     assert_A_grey_knights_combat_patrol),
    ("B - Librarius Conclave no bleed",    assert_B_librarius_conclave_no_bleed),
    ("C - Dreadknight points → Munitorum", assert_C_nemesis_dreadknight_points),
    ("D - Teleport Assault → Grey Knights", assert_D_teleport_assault_is_grey_knights),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    total = collection.count()
    print(f"\nWarhammer 40K RAG — Retrieval Test Suite")
    print(f"Collection: {COLLECTION_NAME} ({total:,} chunks)")

    if total == 0:
        print("\n  FAIL: collection is empty. Did you run `deploy.ps1 -Action ingest` yet?")
        sys.exit(2)

    # --- Diagnostics ---
    print(f"\n{'#' * 70}")
    print(f"#  DIAGNOSTIC QUERIES  ({len(DIAGNOSTIC_QUERIES)} total)")
    print(f"{'#' * 70}")
    for q in DIAGNOSTIC_QUERIES:
        print_results(run_query(q))

    # --- Assertions ---
    print(f"\n\n{'#' * 70}")
    print(f"#  TARGET ASSERTIONS  ({len(ASSERTIONS)} total)")
    print(f"{'#' * 70}")

    passed = []
    failed = []
    for name, fn in ASSERTIONS:
        try:
            fn()
            passed.append(name)
        except AssertionFailure as e:
            failed.append((name, str(e)))
            print(str(e))
        except Exception as e:
            failed.append((name, f"  EXCEPTION: {type(e).__name__}: {e}"))
            print(f"  EXCEPTION: {type(e).__name__}: {e}")

    # --- Summary ---
    print(f"\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Passed: {len(passed)}/{len(ASSERTIONS)}")
    for name in passed:
        print(f"    ok  -  {name}")
    for name, msg in failed:
        print(f"    FAIL - {name}")
        print(f"             {msg}")
    print(f"{'=' * 70}\n")

    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()

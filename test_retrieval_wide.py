"""
test_retrieval_wide.py
======================
Diagnostic retrieval sweep across 40 queries. Pure observation tool — no
assertions, always exits 0. The goal is to surface issues for backlog
prioritization, not to gate deployments.

Location: C:\\Projects\\wh40k-app\\test_retrieval_wide.py
Run:       python test_retrieval_wide.py

Coverage axes:
  1. Per-faction stratagems (8 queries) — probes subject routing across
     the full faction lineup, not just Imperium.
  2. Per-faction points costs (10 queries) — quantifies Munitorum faction
     detection per faction. Expected: Grey Knights works, Dark Angels works
     (probably wrong reason), most others fail.
  3. Phrasing variants (6 queries) — same intent in 3 different wordings,
     measures filter-prompt generalization.
  4. Named detachments (4 queries) — tests whether detachment names route
     to the right parent faction.
  5. Universal abilities (3 queries) — cross-faction keywords.
  6. Combat patrols (3 queries) — patrol_name detection for non-GK.
  7. Core rules / generic (3 queries) — negative cases that should not
     extract a faction.
  8. Edge cases (3 queries) — typos, colloquialisms, ambiguity.

Each result line ends with a status flag:
  [OK]          — filter succeeded; top result appears on-topic
  [WEAK]        — filter succeeded but top result looks like boilerplate
                  (whitespace-heavy / KEYWORDS banner / <20 real words)
  [FILTER-MISS] — Claude returned {} when it should have routed
  [UNFILTERED]  — tier 3 fell through (correct if query was generic)

Drift risk: FILTER_PROMPT is duplicated from main.py. Keep in sync.
"""

import os
import sys
import json
import re

from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from anthropic import Anthropic

load_dotenv()

# ---------------------------------------------------------------------------
# Config — mirrors main.py
# ---------------------------------------------------------------------------

CHROMA_PATH     = os.getenv("CHROMA_PATH",     r"C:\Projects\wh40k-app\chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "warhammer40k")
OLLAMA_URL      = os.getenv("OLLAMA_URL",      "http://127.0.0.1:11434/api/embeddings")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "nomic-embed-text")
MODEL           = "claude-sonnet-4-6"
N_RESULTS       = 40
N_FINAL         = 8
N_DISPLAY       = 3      # top-N to show in sweep (less visual noise than 8)

embedding_fn = OllamaEmbeddingFunction(url=OLLAMA_URL, model_name=OLLAMA_MODEL)
client       = chromadb.PersistentClient(path=CHROMA_PATH)
collection   = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# Filter prompt — duplicated from main.py. Keep in sync.
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

AVAILABLE doc_type VALUES:
  combat_patrol, combat_patrol_rules, faction_pack, core_rules,
  core_rules_updates, core_rules_quickstart, points_costs, balance_dataslate,
  crusade_rules, boarding_actions_rules, tournament_rules, imperial_armour,
  army_roster, other

AVAILABLE subject VALUES (always lowercase, always the codex faction; NEVER
use sub-chapters like 'ultramarines'):
  space marines, grey knights, blood angels, dark angels, black templars,
  space wolves, deathwatch, adepta sororitas, adeptus custodes,
  adeptus mechanicus, astra militarum, imperial knights, imperial agents,
  chaos space marines, death guard, thousand sons, world eaters,
  emperor's children, chaos knights, chaos daemons, aeldari, drukhari,
  genestealer cults, leagues of votann, necrons, orks, t'au empire, tyranids,
  core rules, munitorum field manual, balance dataslate, combat patrol rules,
  crusade rules, boarding actions

OUTPUT FIELDS (all optional):
  doc_type, subject, patrol_name, munitorum_faction

ROUTING GUIDE:
POINTS-COST queries — ALWAYS set doc_type=points_costs AND munitorum_faction.
  Signals: "points", "cost", "pts", "how many points", "how much does X cost".
FACTION-RULES queries — set subject + doc_type=faction_pack.
  Signals: faction + "stratagems" / "detachment" / "enhancement" / "army rule".
COMBAT-PATROL queries — set subject + patrol_name + doc_type=combat_patrol.
UNIT-BY-NAME queries — set subject if unambiguous ("Stormboyz"→orks).
CORE-RULES queries — set doc_type=core_rules for universal rules language.

EXAMPLES (showing phrasing variants):
  "Points cost for a Leman Russ"     -> {"doc_type": "points_costs", "munitorum_faction": "astra militarum"}
  "How many points is a Nemesis Dreadknight?" -> {"doc_type": "points_costs", "munitorum_faction": "grey knights"}
  "How much does a Carnifex cost?"   -> {"doc_type": "points_costs", "munitorum_faction": "tyranids"}
  "Baneblade pts"                    -> {"doc_type": "points_costs", "munitorum_faction": "astra militarum"}
  "Grey Knights stratagems"          -> {"subject": "grey knights", "doc_type": "faction_pack"}
  "Show me Tyranid enhancements"     -> {"subject": "tyranids", "doc_type": "faction_pack"}
  "Librarius Conclave detachment rule" -> {"subject": "space marines", "doc_type": "faction_pack"}
  "What does Oath of Moment do?"     -> {"subject": "space marines", "doc_type": "faction_pack"}
  "Teleport Assault rule"            -> {"subject": "grey knights"}
  "Tell me about Stormboyz"          -> {"subject": "orks"}
  "Aurellios Banishers combat patrol" -> {"subject": "grey knights", "patrol_name": "aurellios banishers", "doc_type": "combat_patrol"}
  "Sanctuary Guardians"              -> {"subject": "adepta sororitas", "patrol_name": "sanctuary guardians", "doc_type": "combat_patrol"}
  "How does overwatch work?"         -> {"doc_type": "core_rules"}
  "What is a stratagem?"             -> {}

Return ONLY valid JSON. No preamble, no code fences, no explanation.
"""


# ---------------------------------------------------------------------------
# Retrieval — mirrors main.py, reports tier
# ---------------------------------------------------------------------------

def extract_filters(query):
    try:
        resp = anthropic_client.messages.create(
            model=MODEL,
            max_tokens=300,
            messages=[{"role": "user", "content": f"{FILTER_PROMPT}\n\nQuery: {query}"}],
        )
        data = json.loads(resp.content[0].text.strip())
    except Exception as e:
        return {}, {}, f"(filter call error: {type(e).__name__})"

    if not isinstance(data, dict):
        return {}, {}, "(filter returned non-dict)"

    filters = []
    for field in ("subject", "doc_type", "patrol_name", "munitorum_faction"):
        val = data.get(field)
        if isinstance(val, str) and val.strip():
            filters.append({field: {"$eq": val.strip().lower()}})

    if not filters:
        return {}, data, ""
    if len(filters) == 1:
        return filters[0], data, ""
    return {"$and": filters}, data, ""


def _subject_from_filter(where):
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


def _chroma_query(query, where):
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


def run_query(query):
    where, filter_data, filter_note = extract_filters(query)

    chunks, metas, dists = _chroma_query(query, where) if where else _chroma_query(query, None)
    tier = "exact" if where else "unfiltered"

    if not chunks and where:
        subject = _subject_from_filter(where)
        if subject:
            chunks, metas, dists = _chroma_query(query, {"subject": {"$eq": subject}})
            if chunks:
                tier = "subject"

    if not chunks:
        chunks, metas, dists = _chroma_query(query, None)
        if where:
            tier = "unfiltered"

    triples = [
        (c, m, d) for c, m, d in zip(chunks, metas, dists) if len(c.split()) >= 5
    ][:N_FINAL]
    return {
        "query":        query,
        "filter_data":  filter_data,
        "where":        where,
        "retrieval_tier": tier,
        "results":      triples,
        "filter_note":  filter_note,
    }


# ---------------------------------------------------------------------------
# Quality classification — crude heuristic, just for the scan flag
# ---------------------------------------------------------------------------

_BOILERPLATE_HINTS = ("FACTION KEYWORDS:", "WARHAMMER   LEGENDS", "| | --- |")


def _effective_word_count(text):
    """Word count excluding tokens that are only punctuation / whitespace noise."""
    stripped = re.sub(r"[|\-=\s]+", " ", text or "").strip()
    return len(stripped.split()) if stripped else 0


def _top_result_quality(results):
    """Return a crude 'on-topic' vs 'boilerplate' flag for the top result."""
    if not results:
        return "empty"
    doc = results[0][0] or ""
    if any(hint in doc for hint in _BOILERPLATE_HINTS) and _effective_word_count(doc) < 25:
        return "boilerplate"
    if _effective_word_count(doc) < 20:
        return "boilerplate"
    return "substantive"


def _scan_flag(query_result, expect_filter):
    """
    One-line status flag for scanning the diagnostic output.
      OK           — filter extracted something AND top result is substantive
      WEAK         — filter worked but top result looks like boilerplate
      FILTER-MISS  — expected a non-empty filter, got {}
      UNFILTERED   — query fell through to tier 3 (fine for generic queries)
      NO-RESULTS   — no chunks survived the ≥5-word filter
    """
    fd = query_result["filter_data"]
    tier = query_result["retrieval_tier"]
    quality = _top_result_quality(query_result["results"])

    if not query_result["results"]:
        return "NO-RESULTS"
    if expect_filter and not fd:
        return "FILTER-MISS"
    if tier == "unfiltered" and expect_filter:
        return "UNFILTERED"
    if quality == "boilerplate":
        return "WEAK"
    return "OK"


# ---------------------------------------------------------------------------
# Test corpus — 40 queries organized by coverage axis
# ---------------------------------------------------------------------------

# Format: (category, query, expect_filter_extraction)
#   expect_filter_extraction=True means we EXPECT filter_data to be non-empty.
#   Set False for genuinely-generic queries so FILTER-MISS doesn't fire on them.

TEST_CORPUS = [
    # --- 1. Per-faction stratagems (8) ---------------------------------------
    ("per-faction stratagems", "What are the Blood Angels stratagems?", True),
    ("per-faction stratagems", "Tau Empire stratagems", True),
    ("per-faction stratagems", "Show me Necron stratagems", True),
    ("per-faction stratagems", "Aeldari stratagems and how they work", True),
    ("per-faction stratagems", "Thousand Sons stratagems", True),
    ("per-faction stratagems", "Genestealer Cults stratagem options", True),
    ("per-faction stratagems", "Chaos Daemons battle tactic stratagems", True),
    ("per-faction stratagems", "What stratagems do World Eaters have?", True),

    # --- 2. Per-faction points costs (10) ------------------------------------
    ("per-faction points", "How many points is a Leman Russ Battle Tank?", True),
    ("per-faction points", "Carnifex points cost", True),
    ("per-faction points", "How much does a Wraithknight cost?", True),
    ("per-faction points", "Ghazghkull Mag Uruk Thraka pts", True),
    ("per-faction points", "Space Marine Captain points", True),
    ("per-faction points", "Terminator squad cost in points", True),
    ("per-faction points", "How many points is a Custodian Guard squad?", True),
    ("per-faction points", "Canoptek Wraiths points", True),
    ("per-faction points", "What's the point cost of a Knight Paladin?", True),
    ("per-faction points", "Death Guard Plague Marines points", True),

    # --- 3. Phrasing variants — same topic, different wordings (6) ----------
    ("phrasing variants", "Grey Knights detachment rules", True),
    ("phrasing variants", "What detachments do Grey Knights have?", True),
    ("phrasing variants", "detachments for grey knights army", True),
    ("phrasing variants", "Ork weapons and ranged profiles", True),
    ("phrasing variants", "What ranged weapons do Orks use?", True),
    ("phrasing variants", "show me the orks' shootas and rokkits", True),

    # --- 4. Named detachments — should route to parent faction (4) ----------
    ("named detachments", "Warpbane Task Force", True),
    ("named detachments", "Librarius Conclave detachment rule", True),
    ("named detachments", "Gladius Task Force", True),       # Space Marines
    ("named detachments", "Mephrit Dynasty", True),          # Necrons

    # --- 5. Universal abilities / cross-faction keywords (3) ----------------
    ("universal abilities", "What does Oath of Moment do?", True),
    ("universal abilities", "Deep Strike rules", False),     # could be core, could be faction
    ("universal abilities", "Armour of Contempt stratagem", True),

    # --- 6. Combat patrols — variety beyond Grey Knights (3) ----------------
    ("combat patrols", "Sanctuary Guardians", True),           # Adepta Sororitas
    ("combat patrols", "Twilight Cavalcade combat patrol", True),  # Drukhari
    ("combat patrols", "Na'pok's Hunters", True),              # Tau

    # --- 7. Core rules / generic — negative cases (3) -----------------------
    ("core rules generic", "How does overwatch work?", True),     # core_rules
    ("core rules generic", "What is a stratagem?", False),        # should be {}
    ("core rules generic", "How do I play Warhammer 40k?", False),# should be {}

    # --- 8. Edge cases — typos, colloquialisms, ambiguity (3) ---------------
    ("edge cases", "wat are grey kn1ghts rules lol", True),  # typo-ridden
    ("edge cases", "Bloodthirster rules",           True),   # unit name only, no faction word
    ("edge cases", "wraithguard stats",             True),   # ambiguous (aeldari)
]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _shorten_source(source):
    if not source:
        return "?"
    return source.replace("\\", "/").split("/")[-1]


def render_result(query_result, expect_filter):
    flag = _scan_flag(query_result, expect_filter)
    flag_padded = f"[{flag}]".ljust(14)

    print(f"\n  {flag_padded} {query_result['query']}")
    print(f"      filter_json  : {json.dumps(query_result['filter_data'])}")
    print(f"      where        : {json.dumps(query_result['where']) if query_result['where'] else '(no filter)'}")
    print(f"      tier         : {query_result['retrieval_tier']}")
    if query_result.get("filter_note"):
        print(f"      note         : {query_result['filter_note']}")
    if not query_result["results"]:
        print(f"      results      : (none)")
        return flag
    for i, (chunk, meta, dist) in enumerate(query_result["results"][:N_DISPLAY], 1):
        src = _shorten_source(meta.get("source", "?"))
        subj = meta.get("subject", "?")
        dt = meta.get("doc_type", "?")
        muni = meta.get("munitorum_faction") or ""
        extra = f" muni={muni}" if muni else ""
        preview = (chunk[:160] or "").replace("\n", " ")
        print(f"      [{i}] dist={dist:.3f} subj={subj} dt={dt}{extra} src={src}")
        print(f"           {preview!r}")
    return flag


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    total = collection.count()
    print(f"\n{'=' * 70}")
    print(f"  Warhammer 40K RAG — Wide Diagnostic Sweep")
    print(f"  Collection: {COLLECTION_NAME} ({total:,} chunks)")
    print(f"  Queries:    {len(TEST_CORPUS)}")
    print(f"{'=' * 70}")

    if total == 0:
        print("\n  FAIL: collection is empty.")
        sys.exit(2)

    # Counters
    category_flags = {}          # {category: {flag: count}}
    overall_flags  = {}          # {flag: count}

    # Category order preserves what's in the corpus
    seen_categories = []
    for cat, _, _ in TEST_CORPUS:
        if cat not in seen_categories:
            seen_categories.append(cat)

    for category in seen_categories:
        print(f"\n\n{'#' * 70}")
        print(f"#  {category.upper()}")
        print(f"{'#' * 70}")
        for cat, query, expect in TEST_CORPUS:
            if cat != category:
                continue
            r = run_query(query)
            flag = render_result(r, expect)
            overall_flags[flag] = overall_flags.get(flag, 0) + 1
            category_flags.setdefault(category, {})
            category_flags[category][flag] = category_flags[category].get(flag, 0) + 1

    # ---- Summary ----
    print(f"\n\n{'=' * 70}")
    print(f"  SUMMARY")
    print(f"{'=' * 70}\n")

    flag_order = ["OK", "WEAK", "UNFILTERED", "FILTER-MISS", "NO-RESULTS"]
    total_q = sum(overall_flags.values()) or 1

    print(f"  Overall:")
    for flag in flag_order:
        n = overall_flags.get(flag, 0)
        if n:
            pct = n * 100 // total_q
            print(f"    {flag:<13}  {n:>3}  ({pct}%)")
    print()
    print(f"  By category:")
    for cat in seen_categories:
        flags = category_flags.get(cat, {})
        total_cat = sum(flags.values())
        parts = []
        for flag in flag_order:
            n = flags.get(flag, 0)
            if n:
                parts.append(f"{flag}={n}")
        print(f"    {cat:<28}  ({total_cat:>2} q)  {' '.join(parts)}")

    print(f"\n{'=' * 70}")
    print(f"  Flag meanings")
    print(f"{'=' * 70}")
    print("    OK          - filter extracted + top result is substantive")
    print("    WEAK        - filter worked but top result is boilerplate/whitespace")
    print("    UNFILTERED  - tier 3 fallback (OK for generic queries)")
    print("    FILTER-MISS - expected a filter, got {} (filter-prompt bug)")
    print("    NO-RESULTS  - query returned nothing after noise filter")
    print()

    # Always exit 0 — this is a diagnostic, not a gate
    sys.exit(0)


if __name__ == "__main__":
    main()

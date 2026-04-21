"""
test_retrieval_filter.py
========================
Unit tests for the new retrieval filter parsing + fallback logic.

We don't require ChromaDB — the fallback tests use a minimal mock
collection so we can exercise the three-tier cascade deterministically.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from retrieval_filter import parse_filter, _extract_subject, query_with_fallback

passed = 0
failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✓ {name}")
    else:
        failed += 1
        print(f"  ✗ {name}   {detail}")


# ---------------------------------------------------------------------------
# parse_filter
# ---------------------------------------------------------------------------

def test_parse_filter():
    print("\n=== parse_filter ===")
    # Bare JSON
    r = parse_filter('{"doc_type": "faction_pack"}')
    check("bare JSON", r == {"doc_type": "faction_pack"}, f"got {r}")

    # JSON in code fences
    r = parse_filter('```json\n{"doc_type": "faction_pack"}\n```')
    check("JSON in code fences", r == {"doc_type": "faction_pack"}, f"got {r}")

    # Fences without language tag
    r = parse_filter('```\n{"subject": "grey knights"}\n```')
    check("JSON in bare fences", r == {"subject": "grey knights"}, f"got {r}")

    # JSON with preamble prose
    r = parse_filter('Here is the filter you requested:\n{"doc_type": "core_rules"}')
    check("JSON with preamble", r == {"doc_type": "core_rules"}, f"got {r}")

    # $and nested
    r = parse_filter('{"$and": [{"doc_type": "faction_pack"}, {"subject": "grey knights"}]}')
    check("$and nested", r == {"$and": [{"doc_type": "faction_pack"}, {"subject": "grey knights"}]})

    # Empty response → empty dict (triggers unfiltered fallback)
    check("empty response → {}", parse_filter("") == {})
    check("None-ish → {}", parse_filter(None) == {})

    # Malformed JSON → empty dict, don't crash
    check("malformed JSON → {}", parse_filter("{not valid json") == {})

    # A plain string is not a dict → {}
    check("non-object JSON → {}", parse_filter('"just a string"') == {})

    # Empty object → empty dict
    check("{} → {}", parse_filter('{}') == {})


# ---------------------------------------------------------------------------
# _extract_subject
# ---------------------------------------------------------------------------

def test_extract_subject():
    print("\n=== _extract_subject ===")
    # Flat
    check("flat subject", _extract_subject({"subject": "grey knights"}) == "grey knights")

    # Nested in $and
    f = {"$and": [{"doc_type": "faction_pack"}, {"subject": "grey knights"}]}
    check("subject in $and", _extract_subject(f) == "grey knights")

    # Nested deep
    f = {"$and": [{"doc_type": "faction_pack"},
                  {"$and": [{"subject": "tyranids"}, {"section_type": "stratagem"}]}]}
    check("subject in deep $and", _extract_subject(f) == "tyranids")

    # $eq form
    check("$eq form", _extract_subject({"subject": {"$eq": "orks"}}) == "orks")

    # Absent
    check("no subject → None", _extract_subject({"doc_type": "core_rules"}) is None)
    check("empty → None", _extract_subject({}) is None)


# ---------------------------------------------------------------------------
# query_with_fallback — using a mock collection
# ---------------------------------------------------------------------------

class MockCollection:
    """
    A ChromaDB-shaped mock that returns a configured number of hits per
    (where-dict-key-tuple) request. Lets tests assert which tier fired.
    """
    def __init__(self, responses):
        # responses is a list of dicts like:
        #   [{"match": <where_or_None>, "hits": <int>}, ...]
        # evaluated in order; first match wins.
        self.responses = responses
        self.calls = []

    def query(self, query_texts, n_results=10, where=None):
        self.calls.append({"where": where, "n_results": n_results})
        n = 0
        for rule in self.responses:
            if rule["match"] == where or (rule["match"] == "*"):
                n = rule["hits"]
                break
        ids = [[f"id{i}" for i in range(n)]]
        return {"ids": ids, "documents": [["doc"] * n], "metadatas": [[{}] * n]}


def test_fallback_tier1_fires():
    print("\n=== query_with_fallback: tier 1 (exact) ===")
    filt = {"$and": [{"doc_type": "faction_pack"}, {"subject": "grey knights"}]}
    coll = MockCollection([
        {"match": filt, "hits": 5},   # exact filter has results
        {"match": {"subject": "grey knights"}, "hits": 20},
        {"match": None, "hits": 100},
    ])
    result = query_with_fallback(coll, "test query", filt, n_results=10)
    check("tier 1 returns exact-filter results", result["retrieval_tier"] == "exact")
    check("tier 1 stops at tier 1", len(coll.calls) == 1,
          f"made {len(coll.calls)} calls; expected 1")


def test_fallback_tier2_fires():
    print("\n=== query_with_fallback: tier 2 (subject-only) ===")
    filt = {"$and": [{"doc_type": "faction_pack"}, {"subject": "grey knights"},
                     {"section_type": "stratagem"}]}
    coll = MockCollection([
        {"match": filt, "hits": 0},                           # tier 1 empty
        {"match": {"subject": "grey knights"}, "hits": 8},    # tier 2 has hits
        {"match": None, "hits": 100},
    ])
    result = query_with_fallback(coll, "test query", filt, n_results=10)
    check("tier 2 fires when tier 1 empty", result["retrieval_tier"] == "subject")
    check("tier 2 stops at tier 2", len(coll.calls) == 2,
          f"made {len(coll.calls)} calls; expected 2")


def test_fallback_tier3_fires():
    print("\n=== query_with_fallback: tier 3 (unfiltered) ===")
    filt = {"$and": [{"doc_type": "faction_pack"}, {"subject": "nonexistent"}]}
    coll = MockCollection([
        {"match": filt, "hits": 0},
        {"match": {"subject": "nonexistent"}, "hits": 0},
        {"match": None, "hits": 25},
    ])
    result = query_with_fallback(coll, "test query", filt, n_results=10)
    check("tier 3 fires when tiers 1-2 empty", result["retrieval_tier"] == "unfiltered")
    check("tier 3 stops at tier 3", len(coll.calls) == 3,
          f"made {len(coll.calls)} calls; expected 3")


def test_fallback_no_subject_skips_tier2():
    print("\n=== query_with_fallback: no subject → tier 2 skipped ===")
    filt = {"doc_type": "core_rules"}
    coll = MockCollection([
        {"match": filt, "hits": 0},    # tier 1 empty
        {"match": None, "hits": 50},   # tier 3 has hits
    ])
    result = query_with_fallback(coll, "test query", filt, n_results=10)
    check("no subject → tier 2 skipped, tier 3 fires",
          result["retrieval_tier"] == "unfiltered")
    check("only 2 calls made (tier 1 + tier 3)", len(coll.calls) == 2,
          f"made {len(coll.calls)} calls; expected 2")


def test_fallback_empty_filter():
    print("\n=== query_with_fallback: empty filter ===")
    coll = MockCollection([
        {"match": None, "hits": 50},
    ])
    result = query_with_fallback(coll, "test query", {}, n_results=10)
    # An empty filter → where=None → this is tier 1 unfiltered, gets results
    check("empty filter → unfiltered tier 1, gets results",
          result["retrieval_tier"] == "exact")


if __name__ == "__main__":
    test_parse_filter()
    test_extract_subject()
    test_fallback_tier1_fires()
    test_fallback_tier2_fires()
    test_fallback_tier3_fires()
    test_fallback_no_subject_skips_tier2()
    test_fallback_empty_filter()
    print(f"\n{'='*50}\n  SUMMARY: {passed} passed, {failed} failed\n{'='*50}")
    sys.exit(0 if failed == 0 else 1)

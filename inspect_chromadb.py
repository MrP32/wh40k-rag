"""
inspect_chromadb.py
===================
Direct inspection of the ChromaDB collection — no Claude, no embeddings,
no retrieval pipeline. Just: "what's actually in here?"

This is the answer to 'I have no way to check whether the chunks exist'.

Location: place at repo root next to test_retrieval.py
Command:  python inspect_chromadb.py [subcommand] [args]

Subcommands
-----------
  stats                          High-level: total chunks, distinct subjects,
                                 distinct doc_types, distinct section_types.

  facets                         Counts per value for each metadata facet.
                                 (How many chunks per subject? per doc_type?)

  find --subject grey knights    Show first N matching chunks (default 5).
       --doc-type combat_patrol  Combine filters with --and (AND) or --or (OR).
       --source-contains banish  Substring match on filename.
       --text-contains teleport  Substring match on chunk text.
       --limit 10

  sample --subject grey knights  Random N chunks matching the filter,
         --n 3                   useful for spot-checking extraction quality.

  page --source aurellios.pdf    All chunks from a specific source PDF,
                                 in page order. Useful to see what survived
                                 ingestion for a single file.

Why no embedding function?
--------------------------
We never call .query() here, only .get() with where-clauses. That means
ChromaDB doesn't need to embed anything, and you can run this even when
Ollama isn't running.
"""

import os
import sys
import json
import random
import argparse
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
import chromadb

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", r"C:\Projects\wh40k-app\chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "warhammer40k")

# Opening WITHOUT an embedding function. We only do .get(), never .query().
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_collection(name=COLLECTION_NAME)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FACET_FIELDS = ["subject", "doc_type", "section_type",
                "munitorum_faction", "patrol_name", "extraction_method"]


def _short_source(s: str | None) -> str:
    if not s:
        return "?"
    return s.split("\\")[-1].split("/")[-1]


def _build_where(args) -> dict | None:
    """Build a Chroma where-clause from CLI args. Returns None for no filter."""
    clauses = []
    if args.subject:
        clauses.append({"subject": {"$eq": args.subject.lower()}})
    if args.doc_type:
        clauses.append({"doc_type": {"$eq": args.doc_type.lower()}})
    if args.section_type:
        clauses.append({"section_type": {"$eq": args.section_type.lower()}})
    if args.munitorum_faction:
        clauses.append({"munitorum_faction": {"$eq": args.munitorum_faction.lower()}})
    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    op = "$or" if getattr(args, "or_mode", False) else "$and"
    return {op: clauses}


def _fetch_all_metadatas() -> list[dict]:
    """Pull every chunk's metadata. Used by stats/facets commands."""
    # Chroma's .get() with no IDs returns everything. We don't need documents,
    # just metadatas.
    result = collection.get(include=["metadatas"])
    return result.get("metadatas", []) or []


def _print_chunk(i: int, doc: str, meta: dict, preview_chars: int = 280) -> None:
    print(f"\n--- [{i}] {_short_source(meta.get('source'))} "
          f"(p.{meta.get('page_number', '?')})")
    print(f"    subject     : {meta.get('subject')}")
    print(f"    doc_type    : {meta.get('doc_type')}")
    print(f"    section_type: {meta.get('section_type')}")
    if meta.get("patrol_name"):
        print(f"    patrol_name : {meta['patrol_name']}")
    if meta.get("munitorum_faction"):
        print(f"    munitorum   : {meta['munitorum_faction']}")
    if meta.get("section_identifier"):
        print(f"    heading     : {meta['section_identifier']}")
    if meta.get("extraction_method"):
        print(f"    extracted_by: {meta['extraction_method']}")
    preview = (doc or "")[:preview_chars].replace("\n", " ")
    print(f"    text        : {preview}{'...' if len(doc or '') > preview_chars else ''}")


# ---------------------------------------------------------------------------
# Subcommands
# ---------------------------------------------------------------------------

def cmd_stats(args):
    total = collection.count()
    metas = _fetch_all_metadatas()
    print(f"\nCollection: {COLLECTION_NAME}")
    print(f"Total chunks: {total:,}")
    print(f"\nDistinct values per facet:")
    for f in FACET_FIELDS:
        values = {m.get(f) for m in metas if m.get(f)}
        print(f"  {f:20s} {len(values):4d} distinct")


def cmd_facets(args):
    metas = _fetch_all_metadatas()
    print(f"\nFacet breakdown ({len(metas):,} chunks)")
    for f in FACET_FIELDS:
        counts = Counter(m.get(f) for m in metas if m.get(f))
        if not counts:
            continue
        print(f"\n{f}")
        print("-" * (len(f) + 2))
        for value, n in counts.most_common(args.top):
            print(f"  {n:6,d}  {value}")
        remaining = len(counts) - args.top
        if remaining > 0:
            print(f"  ... {remaining} more")


def cmd_find(args):
    where = _build_where(args)
    print(f"\nWHERE: {json.dumps(where) if where else '(none)'}")

    # CRITICAL: when text-/source-contains is used, we must fetch ALL matching
    # chunks first, then filter in Python, then apply the limit. Applying the
    # limit at the Chroma level means we only ever scan that many chunks —
    # the substring filter has nothing to look through. This was a real bug
    # in the v1 inspector that made every text-contains search return zero.
    needs_python_filter = bool(args.text_contains or args.source_contains)

    if needs_python_filter:
        result = collection.get(where=where,
                                include=["documents", "metadatas"])
    else:
        result = collection.get(where=where, limit=args.limit,
                                include=["documents", "metadatas"])

    docs = result.get("documents", []) or []
    metas = result.get("metadatas", []) or []
    pairs = list(zip(docs, metas))

    total_before_filter = len(pairs)

    if args.source_contains:
        needle = args.source_contains.lower()
        pairs = [(d, m) for d, m in pairs
                 if needle in (m.get("source") or "").lower()]
    if args.text_contains:
        needle = args.text_contains.lower()
        pairs = [(d, m) for d, m in pairs if needle in (d or "").lower()]

    total_after_filter = len(pairs)

    # Tell the user what the funnel looked like — important for understanding
    # whether a zero-result really means "doesn't exist" vs "filter too narrow"
    if needs_python_filter:
        print(f"Scanned {total_before_filter:,} chunks matching the WHERE clause; "
              f"{total_after_filter:,} matched the text/source filter.")

    pairs = pairs[: args.limit]

    if not pairs:
        print("(no matching chunks)")
        print("\nTip: try `facets` to see what values actually exist.")
        return

    print(f"Showing {len(pairs)} of {total_after_filter:,} matches:")
    for i, (doc, meta) in enumerate(pairs, 1):
        _print_chunk(i, doc, meta)


def cmd_sample(args):
    where = _build_where(args)
    # Pull a larger pool then random.sample, since Chroma .get() doesn't
    # support random ordering.
    pool_size = max(args.n * 20, 100)
    result = collection.get(where=where, limit=pool_size,
                            include=["documents", "metadatas"])
    docs = result.get("documents", []) or []
    metas = result.get("metadatas", []) or []
    pairs = list(zip(docs, metas))
    if not pairs:
        print("(no matching chunks)")
        return
    chosen = random.sample(pairs, min(args.n, len(pairs)))
    print(f"\nRandom {len(chosen)} of {len(pairs)} matching chunks:")
    for i, (doc, meta) in enumerate(chosen, 1):
        _print_chunk(i, doc, meta)


def cmd_page(args):
    """All chunks from a specific source file, sorted by page number."""
    metas = _fetch_all_metadatas()
    needle = args.source.lower()
    matching_ids = []
    # Match against the short filename
    for i, m in enumerate(metas):
        if needle in (_short_source(m.get("source"))).lower():
            matching_ids.append(i)
    if not matching_ids:
        print(f"(no chunks found for source matching '{args.source}')")
        return

    # Re-fetch with documents this time, filtered by source
    # We use the actual full source string from the first match
    full_source = metas[matching_ids[0]].get("source")
    result = collection.get(where={"source": {"$eq": full_source}},
                            include=["documents", "metadatas"])
    docs = result.get("documents", []) or []
    metas2 = result.get("metadatas", []) or []
    pairs = sorted(zip(docs, metas2),
                   key=lambda p: (p[1].get("page_number") or 0))
    print(f"\n{len(pairs)} chunks from {_short_source(full_source)}:")
    for i, (doc, meta) in enumerate(pairs, 1):
        _print_chunk(i, doc, meta, preview_chars=160)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _add_filter_args(p):
    p.add_argument("--subject")
    p.add_argument("--doc-type", dest="doc_type")
    p.add_argument("--section-type", dest="section_type")
    p.add_argument("--munitorum-faction", dest="munitorum_faction")
    p.add_argument("--or", dest="or_mode", action="store_true",
                   help="Combine filters with OR instead of AND")


def main():
    parser = argparse.ArgumentParser(description="Inspect the WH40K ChromaDB collection")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_stats = sub.add_parser("stats", help="High-level collection stats")

    p_facets = sub.add_parser("facets", help="Counts per value for each facet")
    p_facets.add_argument("--top", type=int, default=15,
                          help="How many top values per facet (default 15)")

    p_find = sub.add_parser("find", help="Find chunks by metadata")
    _add_filter_args(p_find)
    p_find.add_argument("--source-contains", dest="source_contains")
    p_find.add_argument("--text-contains", dest="text_contains")
    p_find.add_argument("--limit", type=int, default=5)

    p_sample = sub.add_parser("sample", help="Random sample for spot-checking")
    _add_filter_args(p_sample)
    p_sample.add_argument("--n", type=int, default=3)

    p_page = sub.add_parser("page", help="All chunks from one source PDF")
    p_page.add_argument("--source", required=True,
                        help="Substring of filename, e.g. 'aurellios'")

    args = parser.parse_args()
    {"stats": cmd_stats, "facets": cmd_facets, "find": cmd_find,
     "sample": cmd_sample, "page": cmd_page}[args.cmd](args)


if __name__ == "__main__":
    main()
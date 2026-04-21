"""
diagnose.py — Read-only diagnostic against the warhammer40k ChromaDB collection.

Run from C:\\Projects\\wh40k-app\\ with the venv activated:
    python diagnose.py

No writes. No ingestion. Just prints what's actually stored.

Updated for the refactored schema. Queries use doc_type / patrol_name /
munitorum_faction (not the old content_category). The Spearpoint Paragon
bleed check reads section_identifier from metadata rather than looking for
a [SPEARPOINT PARAGON] prefix in the embedded document, because the new
ingest only prefixes when the heading classifier is confident.
"""

import os
from collections import Counter

from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

load_dotenv()

CHROMA_PATH     = os.getenv("CHROMA_PATH",     r"C:\Projects\wh40k-app\chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "warhammer40k")
OLLAMA_URL      = os.getenv("OLLAMA_URL",      "http://127.0.0.1:11434/api/embeddings")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "nomic-embed-text")

embedding_fn = OllamaEmbeddingFunction(url=OLLAMA_URL, model_name=OLLAMA_MODEL)
client       = chromadb.PersistentClient(path=CHROMA_PATH)
collection   = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


def hr(title: str) -> None:
    print(f"\n-- {title} " + "-" * max(0, 65 - len(title)))


print(f"\n{'=' * 70}")
print(f"  Collection: {COLLECTION_NAME}  |  total chunks: {collection.count():,}")
print(f"{'=' * 70}")


# ---------------------------------------------------------------------------
# 1. doc_type distribution
# ---------------------------------------------------------------------------
hr("1. doc_type distribution")
all_meta = collection.get(include=["metadatas"])["metadatas"]
doc_types = Counter(m.get("doc_type", "<missing>") for m in all_meta)
for dt, count in sorted(doc_types.items(), key=lambda x: -x[1]):
    print(f"  {count:>6,}  {dt}")


# ---------------------------------------------------------------------------
# 2. subject distribution (top 20)
# ---------------------------------------------------------------------------
hr("2. subject distribution (top 20)")
subjects = Counter(m.get("subject", "<missing>") for m in all_meta)
for subj, count in sorted(subjects.items(), key=lambda x: -x[1])[:20]:
    print(f"  {count:>6,}  {subj}")
print(f"  (total unique subjects: {len(subjects)})")


# ---------------------------------------------------------------------------
# 3. patrol_name distribution (non-empty only)
# ---------------------------------------------------------------------------
hr("3. patrol_name distribution (non-empty)")
patrols = Counter(m.get("patrol_name", "") for m in all_meta if m.get("patrol_name"))
if not patrols:
    print("  (no chunks carry a patrol_name)")
else:
    for name, count in sorted(patrols.items(), key=lambda x: -x[1]):
        print(f"  {count:>6,}  {name}")


# ---------------------------------------------------------------------------
# 4. munitorum_faction distribution (non-empty only)
# ---------------------------------------------------------------------------
hr("4. munitorum_faction distribution (non-empty)")
muni = Counter(m.get("munitorum_faction", "") for m in all_meta if m.get("munitorum_faction"))
if not muni:
    print("  (no chunks carry a munitorum_faction — did Munitorum PDF ingest?)")
else:
    for name, count in sorted(muni.items(), key=lambda x: -x[1]):
        print(f"  {count:>6,}  {name}")


# ---------------------------------------------------------------------------
# 5. Do the filter combos main.py builds actually return chunks?
# ---------------------------------------------------------------------------
hr("5. testing filter combinations main.py would build")

test_filters = [
    {"subject": "grey knights", "doc_type": "combat_patrol"},
    {"doc_type": "combat_patrol"},
    {"subject": "grey knights", "doc_type": "faction_pack"},
    {"doc_type": "points_costs", "munitorum_faction": "grey knights"},
    {"doc_type": "points_costs"},
    {"subject": "grey knights"},
    {"subject": "space marines"},
    {"patrol_name": "aurellios banishers"},
]

for f in test_filters:
    # main.py wraps single-clause filters in $eq; this mirrors that
    if len(f) > 1:
        where = {"$and": [{k: {"$eq": v}} for k, v in f.items()]}
    else:
        k, v = next(iter(f.items()))
        where = {k: {"$eq": v}}
    try:
        hits = collection.get(where=where, limit=1, include=["metadatas"])
        all_hits = collection.get(where=where, include=["metadatas"])
        total = len(all_hits["ids"])
        if total == 0:
            print(f"  [x] {f} -> ZERO matches")
        else:
            sample_source = hits["metadatas"][0].get("source", "") or ""
            fname = sample_source.replace("\\", "/").split("/")[-1]
            print(f"  [ok] {f} -> {total:,} matches  (sample: {fname})")
    except Exception as e:
        print(f"  [!] {f} -> error: {e}")


# ---------------------------------------------------------------------------
# 6. SPEARPOINT PARAGON bleed — heading should NOT appear on dozens of chunks
# ---------------------------------------------------------------------------
hr("6. SPEARPOINT PARAGON bleed check")
sm_chunks = collection.get(
    where={"subject": {"$eq": "space marines"}},
    include=["documents", "metadatas"],
)

# Post-refactor: the bleed manifests in metadata.section_identifier, not in
# an embedded-doc prefix. Count how many SM chunks have this identifier.
bleed = [
    (doc, meta)
    for doc, meta in zip(sm_chunks["documents"], sm_chunks["metadatas"])
    if (meta.get("section_identifier") or "").upper() == "SPEARPOINT PARAGON"
]
print(f"  Space Marines chunks with section_identifier='SPEARPOINT PARAGON': {len(bleed)}")
print(f"  (Spearpoint Paragon is a Grey Knights detachment — any count > 0")
print(f"   here means the heading-classifier reset isn't working. A healthy")
print(f"   ingest should be 0 for this specific cross-faction bleed.)")
for doc, meta in bleed[:3]:
    print(f"\n  page={meta.get('page_number')} region={meta.get('region_index')} "
          f"section_type={meta.get('section_type')} "
          f"confident={meta.get('classification_confident')}")
    print(f"  first 200 chars of doc: {doc[:200]!r}")


# ---------------------------------------------------------------------------
# 7. Does "Librarius Conclave" appear in Space Marines chunks?
# ---------------------------------------------------------------------------
hr("7. 'Librarius Conclave' presence in Space Marines chunks")
librarius_hits = [
    (doc, meta)
    for doc, meta in zip(sm_chunks["documents"], sm_chunks["metadatas"])
    if "librarius conclave" in (doc or "").lower()
]
print(f"  Space Marines chunks containing 'Librarius Conclave': {len(librarius_hits)}")
for doc, meta in librarius_hits[:3]:
    print(f"\n  page={meta.get('page_number')} region={meta.get('region_index')} "
          f"section_identifier={meta.get('section_identifier')!r}")
    print(f"  first 300 chars: {doc[:300]!r}")


# ---------------------------------------------------------------------------
# 8. Top section_identifier values per faction — health check for
#    carry-forward (you want lots of distinct identifiers, not one or two
#    bleeding across thousands of chunks)
# ---------------------------------------------------------------------------
hr("8. Top 20 section_identifier values in Space Marines chunks")
sm_ids = Counter((m.get("section_identifier") or "") for m in sm_chunks["metadatas"])
for ident, count in sorted(sm_ids.items(), key=lambda x: -x[1])[:20]:
    label = ident if ident else "(empty)"
    print(f"  {count:>4}  {label!r}")


# ---------------------------------------------------------------------------
# 9. Confidence distribution — how many chunks have a confident heading?
# ---------------------------------------------------------------------------
hr("9. classification_confident distribution (all chunks)")
conf = Counter(bool(m.get("classification_confident")) for m in all_meta)
total = sum(conf.values()) or 1
print(f"  confident=True  : {conf[True]:>6,}  ({conf[True] * 100 // total}%)")
print(f"  confident=False : {conf[False]:>6,}  ({conf[False] * 100 // total}%)")


print(f"\n{'=' * 70}")
print("  Diagnostic complete.")
print(f"{'=' * 70}\n")

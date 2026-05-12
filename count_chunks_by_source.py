"""
count_chunks_by_source.py
=========================
Lists every distinct source PDF in the ChromaDB collection and how many
chunks each one produced. Sorted by chunk count, highest first.

Read-only — does not modify the database.

Location: place at repo root next to inspect_chromadb.py
Command:  python count_chunks_by_source.py

What to look for in the output
------------------------------
- Suspiciously small counts (1-5 chunks) — usually a template, blank
  form, or PDF where extraction mostly failed
- Suspiciously large counts on small PDFs — over-segmentation
- Source filenames you don't recognize
- Near-duplicates (same PDF re-ingested with slightly different names)
"""

import os
from collections import Counter
from dotenv import load_dotenv
import chromadb

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", r"C:\Projects\wh40k-app\chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "warhammer40k")


def short_source(s):
    """Strip directory paths — keep just the filename."""
    if not s:
        return "(no source)"
    return s.split("\\")[-1].split("/")[-1]


def main():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    # Pull every chunk's metadata. We don't need the document text here,
    # just the source field.
    metadatas = collection.get(include=["metadatas"])["metadatas"] or []

    counts = Counter(short_source(m.get("source")) for m in metadatas)

    total_chunks = sum(counts.values())
    total_sources = len(counts)

    print(f"\nCollection: {COLLECTION_NAME}")
    print(f"Total chunks: {total_chunks:,}")
    print(f"Distinct source PDFs: {total_sources}")
    print(f"\n{'Chunks':>7s}  Source")
    print("-" * 70)

    for source, n in sorted(counts.items(), key=lambda x: -x[1]):
        # Flag suspicious cases inline so you can spot them at a glance
        flag = ""
        if n <= 5:
            flag = "  <-- very few chunks, possible template or failed extraction"
        elif n >= 500:
            flag = "  <-- many chunks, possible over-segmentation"
        print(f"{n:7,d}  {source}{flag}")


if __name__ == "__main__":
    main()

"""
delete_pdfs.py
==============
Delete all chunks from specified source PDFs in the ChromaDB collection.

Defaults to DRY-RUN. You must pass --execute to actually delete anything.
This is intentional. Destructive scripts should never destroy by default.

Location: place at repo root next to inspect_chromadb.py
Commands:
  python delete_pdfs.py                  # dry-run: shows what would be deleted
  python delete_pdfs.py --execute        # actually performs the deletion

How rollback works
------------------
This script does not back anything up. Before running with --execute, you
should manually copy the ChromaDB directory:

  Copy-Item -Recurse C:\\Projects\\wh40k-app\\chroma_db C:\\Projects\\wh40k-app\\chroma_db_backup_YYYYMMDD

If you skip the backup, the only recovery path is to re-ingest from the
source PDFs (which is slow but possible — the PDFs themselves live in your
source folder, not in ChromaDB).

What it deletes
---------------
All chunks whose `source` metadata field's filename matches one of the
PDFs listed in PDFS_TO_DELETE below. Match is case-insensitive on the
filename portion of the path.
"""

import os
import sys
import argparse
from dotenv import load_dotenv
import chromadb

load_dotenv()

CHROMA_PATH = os.getenv("CHROMA_PATH", r"C:\Projects\wh40k-app\chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "warhammer40k")

# The list of PDFs whose chunks should be deleted from the collection.
# Match is case-insensitive on the short filename (last path component).
# Add a PDF here to mark it for deletion in the next run.
PDFS_TO_DELETE = [
    "Army Roster.pdf",
    "Boarding Actions Companion.pdf",
]


def short_source(s):
    if not s:
        return ""
    return s.split("\\")[-1].split("/")[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true",
                   help="Actually perform the deletion. Without this flag, "
                        "the script is a dry-run that only reports what "
                        "would be deleted.")
    args = p.parse_args()

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=COLLECTION_NAME)

    before_count = collection.count()
    print(f"\nCollection: {COLLECTION_NAME}")
    print(f"Current chunk count: {before_count:,}")
    print(f"\nTargeting {len(PDFS_TO_DELETE)} PDF(s) for deletion:")
    for name in PDFS_TO_DELETE:
        print(f"  - {name}")

    # Pull every chunk's ID and source so we can find matches.
    # We don't need documents — just IDs and metadata.
    result = collection.get(include=["metadatas"])
    ids = result.get("ids", []) or []
    metas = result.get("metadatas", []) or []

    # Build lowercase set for case-insensitive matching
    targets_lower = {name.lower() for name in PDFS_TO_DELETE}

    ids_to_delete = []
    per_pdf_counts = {name: 0 for name in PDFS_TO_DELETE}

    for chunk_id, meta in zip(ids, metas):
        src_short = short_source(meta.get("source"))
        if src_short.lower() in targets_lower:
            ids_to_delete.append(chunk_id)
            # Find which target this matched (case-insensitive)
            for name in PDFS_TO_DELETE:
                if src_short.lower() == name.lower():
                    per_pdf_counts[name] += 1
                    break

    print(f"\nFound {len(ids_to_delete)} chunk(s) matching the target list:")
    for name, n in per_pdf_counts.items():
        marker = "" if n > 0 else "  <-- WARNING: no chunks found"
        print(f"  {n:5d}  {name}{marker}")

    if not ids_to_delete:
        print("\nNothing to delete. Exiting.")
        sys.exit(0)

    if not args.execute:
        print(f"\nDRY-RUN: no chunks were actually deleted.")
        print(f"To actually delete: python delete_pdfs.py --execute")
        sys.exit(0)

    # Past this point we're actually deleting.
    print(f"\n--execute flag detected. Deleting {len(ids_to_delete)} chunks...")
    collection.delete(ids=ids_to_delete)

    after_count = collection.count()
    print(f"\nDone.")
    print(f"Before: {before_count:,} chunks")
    print(f"After:  {after_count:,} chunks")
    print(f"Removed: {before_count - after_count:,} chunks")

    # Sanity check: chunk count should drop by exactly len(ids_to_delete)
    if before_count - after_count != len(ids_to_delete):
        print(f"\nWARNING: expected to remove {len(ids_to_delete)} chunks, "
              f"actually removed {before_count - after_count}. "
              f"Something may have gone wrong.")
        sys.exit(1)


if __name__ == "__main__":
    main()

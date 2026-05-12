"""
test_munitorum_fix.py
=====================
Run the fixed munitorum_parser against the Munitorum Field Manual PDF only,
without touching ChromaDB. Print proposed faction labels and counts.

Use this BEFORE re-ingesting the full corpus to verify the fix is working.

Location: place at repo root (C:\\Projects\\wh40k-app\\test_munitorum_fix.py)
Command:  python test_munitorum_fix.py

What this does NOT do
---------------------
- Write to ChromaDB. Pure read+analysis, completely safe.
- Re-ingest other PDFs. Only the Munitorum.

What you're looking for in the output
-------------------------------------
1. Dark Angels count: was 163, should drop substantially (probably 10-30)
2. Grey Knights count: was 4, should rise (probably 15-40)
3. "DETECTION CHANGES" section: chunks that *would* get a different label
   under the fix vs. the buggy v2 detector. Spot-check a few.
"""

import sys
from pathlib import Path
from collections import Counter

# Add pdf_agent to path so we can import the parser and segmenter
sys.path.insert(0, str(Path(__file__).parent / "pdf_agent"))

from pdf_agent import assess_pdf
from pdf_region_segmenter import segment_document_into_regions
from munitorum_parser import (
    tag_chunks_with_faction,
    detect_faction_in_text,
)


PDF_FOLDER = r"C:\Personal Projects\warhammer_40k_pdfs"
MUNITORUM_PDF = "Munitorum Field Manual.pdf"


def short(text, n=80):
    return (text or "").replace("\n", " ")[:n]


def main():
    pdf_path = Path(PDF_FOLDER) / MUNITORUM_PDF
    if not pdf_path.exists():
        print(f"FATAL: {pdf_path} not found")
        print(f"Check PDF_FOLDER at the top of this script")
        sys.exit(1)

    print(f"Running fixed munitorum_parser against {MUNITORUM_PDF}")
    print(f"(this only re-extracts ONE PDF, not the whole corpus)\n")

    # Run the same extraction pipeline ingest.py uses
    print("[1/3] Assessing PDF...")
    assessment = assess_pdf(str(pdf_path))
    print(f"      pdf_type={assessment.pdf_type}, "
          f"pages={assessment.total_pages}")

    print("\n[2/3] Segmenting into chunks...")
    chunks, _ = segment_document_into_regions(str(pdf_path), assessment)
    print(f"      Produced {len(chunks)} chunks")

    print("\n[3/3] Applying fixed faction tagging...")
    tag_chunks_with_faction(chunks)

    # --- Summary: per-faction counts under the fix ---
    print("\n" + "=" * 70)
    print(" PROPOSED FACTION DISTRIBUTION (after fix)")
    print("=" * 70)
    counts = Counter(c["metadata"].get("munitorum_faction", "") for c in chunks)
    total = sum(counts.values())
    untagged = counts.get("", 0)
    for faction, n in sorted(counts.items(), key=lambda x: -x[1]):
        label = faction if faction else "(no faction — intro/TOC)"
        pct = (n / total * 100) if total else 0
        print(f"  {n:5d}  ({pct:5.1f}%)  {label}")
    print(f"\n  Total: {total} chunks, {untagged} untagged")

    # --- Highlight chunks where the detector fired ---
    print("\n" + "=" * 70)
    print(" CHUNKS WHERE A FACTION HEADER WAS DETECTED (top of section)")
    print("=" * 70)
    print(" These are the 'anchor' chunks that set the carry-forward.")
    print(" If any look wrong, the fix has a remaining issue.\n")
    for chunk in chunks:
        detected = detect_faction_in_text(chunk.get("text", ""))
        if detected:
            page = chunk.get("page_number", "?")
            text_preview = short(chunk.get("text", ""), 120)
            print(f"  p.{page:>3}  detected={detected}")
            print(f"         text: {text_preview}")
            print()

    # --- Spot-check chunks that mention specific units ---
    print("\n" + "=" * 70)
    print(" SPOT-CHECK: chunks mentioning specific Grey Knights units")
    print("=" * 70)
    print(" These should now be tagged 'grey knights', not 'dark angels'.\n")
    gk_unit_terms = ["nemesis dreadknight", "brother-captain",
                     "grand master", "purifier", "strike squad"]
    for chunk in chunks:
        text_lower = chunk.get("text", "").lower()
        for term in gk_unit_terms:
            if term in text_lower:
                page = chunk.get("page_number", "?")
                tag = chunk["metadata"].get("munitorum_faction", "(none)")
                preview = short(chunk.get("text", ""), 100)
                marker = "OK " if tag == "grey knights" else "??? "
                print(f"  {marker}p.{page:>3}  tag={tag}")
                print(f"         text: {preview}")
                print()
                break  # one match per chunk is enough

    print("\nDone. If labels look right, you can proceed with full re-ingest.")
    print("If labels look wrong, do NOT re-ingest — iterate on the parser first.")


if __name__ == "__main__":
    main()

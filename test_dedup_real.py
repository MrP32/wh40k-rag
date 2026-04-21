"""
test_dedup_real.py
==================
Runs the real segmenter against Grey Knights Faction Pack page 3 (Warpbane
Task Force stratagems) to verify the _deduplicate_regions() fix.

Expected outcome per HANDOFF.md:
  Before fix:  page 3 produces multiple duplicate-content chunks per stratagem
               because pdfplumber reports overlapping decorative image bboxes
               as multiple regions.
  After fix:   page 3 produces one chunk per stratagem (6 stratagems visible
               on that page: SANCTIFIED KILL ZONE, FLAMES OF SANCTITY,
               HALLOWED BEACON, FIRES OF COVENANT, AEGIS ETERNAL,
               REPELLING SPHERE).

We run the full segment_document_into_regions but cap to just page 3 and
inspect the output.
"""

import sys
import logging
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

# Verbose logging so we can see what the pipeline is doing
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

from pdf_agent import assess_pdf
from pdf_region_segmenter import segment_document_into_regions

PDF = "/home/claude/40k/testpdfs/Faction_Pack_-_Grey_Knights.pdf"
TARGET_PAGE = 3

def main():
    print(f"\n{'='*70}\n  Testing dedup on {Path(PDF).name} page {TARGET_PAGE}\n{'='*70}")

    assessment = assess_pdf(PDF)
    print(f"\n[ASSESS] type={assessment.pdf_type}  pages={assessment.total_pages}")

    # Only run on pages 1..3 to keep it fast; we only inspect page 3 output
    chunks, statlines = segment_document_into_regions(PDF, assessment, max_pages=TARGET_PAGE)
    print(f"\n[TOTAL] {len(chunks)} chunks across first {TARGET_PAGE} pages")

    # Inspect page 3 only
    pg3 = [c for c in chunks if c["page_number"] == TARGET_PAGE]
    print(f"[PAGE {TARGET_PAGE}] {len(pg3)} chunks extracted")
    print()

    # Print a compact summary
    for i, c in enumerate(pg3):
        preview = c["text"].strip().replace("\n", " ")[:100]
        print(f"  region {c['region_index']:2d} col={c['column_label']:<6} "
              f"src={c['geometric_source']:<14} "
              f"wc={c['word_count']:<4} "
              f"type={c['content_type']:<9} "
              f"{preview!r}")

    # Duplicate-content detection: look for chunks whose (column + first-80-chars)
    # match. These are what the bug used to produce.
    signature_counter = Counter()
    for c in pg3:
        sig = (c["column_label"], c["text"].strip()[:80])
        signature_counter[sig] += 1

    dupes = [(sig, n) for sig, n in signature_counter.items() if n > 1]
    print(f"\n[DEDUP-CHECK] duplicate (column + first-80-chars) signatures: {len(dupes)}")
    for (col, preview), n in dupes:
        print(f"   x{n}  col={col}  preview={preview!r}")

    # Check for expected stratagem names
    expected_stratagems = [
        "SANCTIFIED KILL ZONE",
        "FLAMES OF SANCTITY",
        "HALLOWED BEACON",
        "FIRES OF COVENANT",
        "AEGIS ETERNAL",
        "REPELLING SPHERE",
    ]
    all_text = " ".join(c["text"] for c in pg3)
    found = [s for s in expected_stratagems if s in all_text]
    missing = [s for s in expected_stratagems if s not in all_text]

    print(f"\n[COVERAGE] stratagems found: {len(found)}/{len(expected_stratagems)}")
    for s in found:
        print(f"   ✓ {s}")
    for s in missing:
        print(f"   ✗ MISSING: {s}")

    # Final verdict
    print(f"\n{'='*70}")
    if not dupes and len(found) == len(expected_stratagems):
        print("  ✓ DEDUP VERIFIED: no duplicate-content chunks, all stratagems present")
    elif dupes:
        print(f"  ✗ DEDUP FAILED: {len(dupes)} duplicate signature(s) still present")
    elif missing:
        print(f"  ⚠ COVERAGE GAP: {len(missing)} expected stratagem(s) not found in text")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

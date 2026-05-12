"""
diagnose_page23.py
==================
Print the raw, unmodified text of chunks from specific pages of the Munitorum
PDF, with explicit line markers so we can see exactly what the parser sees.

No ChromaDB. No tagging. Pure diagnostic.

Run: python diagnose_page23.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "pdf_agent"))

from pdf_agent import assess_pdf
from pdf_region_segmenter import segment_document_into_regions


PDF_FOLDER = r"C:\Personal Projects\warhammer_40k_pdfs"
MUNITORUM_PDF = "Munitorum Field Manual.pdf"
PAGES_OF_INTEREST = [23, 41, 42, 43, 49]  # 23 = broken case, 49 = working case


def main():
    pdf_path = Path(PDF_FOLDER) / MUNITORUM_PDF
    if not pdf_path.exists():
        print(f"FATAL: {pdf_path} not found")
        sys.exit(1)

    print(f"Extracting chunks from {MUNITORUM_PDF}...\n")
    assessment = assess_pdf(str(pdf_path))
    chunks, _ = segment_document_into_regions(str(pdf_path), assessment)

    for target_page in PAGES_OF_INTEREST:
        page_chunks = [c for c in chunks
                       if c.get("page_number") == target_page]

        print("=" * 70)
        print(f" PAGE {target_page}: {len(page_chunks)} chunks")
        print("=" * 70)

        for i, chunk in enumerate(page_chunks[:5], 1):
            print(f"\n--- chunk {i} (region_index={chunk.get('region_index')}) ---")
            text = chunk.get("text", "")
            lines = text.split("\n")
            print(f"    total lines: {len(lines)}")
            print(f"    first 8 lines, marked:")
            for line_no, line in enumerate(lines[:8], 1):
                # Show explicit empty-line markers and trailing whitespace
                marker = "(empty)" if not line.strip() else ""
                # Repr makes whitespace and special chars visible
                print(f"      [{line_no:2d}] {marker} {repr(line)}")
            if len(lines) > 8:
                print(f"      ... {len(lines) - 8} more lines")
        print()


if __name__ == "__main__":
    main()

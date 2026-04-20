"""
assess_to_csv.py
================
Diagnostic tool — runs PDF assessment across all source PDFs and writes
a timestamped CSV report to C:\Projects\wh40k-app\AnalysisResults\

Location: C:\Projects\wh40k-app\pdf_agent\assess_to_csv.py

Usage:
    python pdf_agent\assess_to_csv.py "C:\Personal Projects\warhammer_40k_pdfs"
"""

import csv
import sys
import logging
from pathlib import Path
from datetime import datetime
from dataclasses import asdict

sys.path.insert(0, str(Path(__file__).parent))
from pdf_agent import assess_pdf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

COLUMNS = [
    "file_name", "file_size_kb", "total_pages", "pdf_type",
    "extraction_strategy", "has_extractable_text", "text_coverage_pct",
    "has_tables", "has_embedded_images", "fonts_embedded",
    "garbled_text", "title", "author", "path",
]


def run_assessment_export(pdf_folder: str, output_root: str = None):
    folder = Path(pdf_folder)
    if not folder.is_dir():
        print(f"Error: '{pdf_folder}' is not a valid directory.")
        sys.exit(1)

    pdfs = sorted(folder.glob("**/*.pdf"))
    if not pdfs:
        print(f"No PDF files found in '{pdf_folder}'.")
        sys.exit(1)

    log.info(f"Found {len(pdfs)} PDFs in {folder}")

    output_root = Path(output_root) if output_root else Path(r"C:\Projects\wh40k-app\AnalysisResults")
    output_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path  = output_root / f"PDFAssessmentMetadata_{timestamp}.csv"

    rows, failed = [], []

    for i, pdf in enumerate(pdfs, start=1):
        log.info(f"[{i}/{len(pdfs)}] Assessing: {pdf.name}")
        try:
            a = asdict(assess_pdf(str(pdf)))
            rows.append({
                "file_name":            pdf.name,
                "file_size_kb":         round(pdf.stat().st_size / 1024, 1),
                "total_pages":          a["total_pages"],
                "pdf_type":             a["pdf_type"],
                "extraction_strategy":  a["extraction_strategy"],
                "has_extractable_text": a["has_extractable_text"],
                "text_coverage_pct":    f"{a['text_coverage'] * 100:.1f}%",
                "has_tables":           a["has_tables"],
                "has_embedded_images":  a["has_embedded_images"],
                "fonts_embedded":       a["fonts_embedded"],
                "garbled_text":         a["garbled_text"],
                "title":                a["title"] or "",
                "author":               a["author"] or "",
                "path":                 str(pdf),
            })
        except Exception as e:
            log.warning(f"Failed to assess {pdf.name}: {e}")
            failed.append({"file_name": pdf.name, "error": str(e)})

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    if failed:
        fail_path = output_root / f"PDFAssessmentMetadata_{timestamp}_ERRORS.csv"
        with open(fail_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["file_name", "error"])
            writer.writeheader()
            writer.writerows(failed)
        log.warning(f"{len(failed)} PDFs failed — see {fail_path.name}")

    print("\n" + "="*60)
    print("  ASSESSMENT COMPLETE")
    print("="*60)
    print(f"  Total PDFs assessed : {len(rows)}")
    print(f"  Failed              : {len(failed)}")

    type_counts = {}
    for r in rows:
        t = r["pdf_type"]
        type_counts[t] = type_counts.get(t, 0) + 1
    print("\n  PDF Type Breakdown:")
    for t, count in sorted(type_counts.items()):
        print(f"    {t:<20} {count} PDFs")

    print(f"\n  Output CSV : {csv_path}")
    print("="*60 + "\n")
    return str(csv_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Path to folder containing PDFs")
    parser.add_argument("--output-root", default=None)
    args = parser.parse_args()
    run_assessment_export(args.folder, output_root=args.output_root)

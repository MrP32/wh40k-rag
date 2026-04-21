"""
pdf_agent.py
============
PDF assessment and shared extraction helpers.

Location: C:\\Projects\\wh40k-app\\pdf_agent\\pdf_agent.py

Responsibilities AFTER the refactor:
  - assess_pdf() — classify PDFs by type (text/table-heavy/mixed/scanned/graphic-stats)
  - Shared OCR preprocessing + statline parsing + table-to-markdown helpers
  - Constants used across the pipeline

REMOVED (moved or deleted):
  - Five strategy handlers (extract_pdfplumber_text, etc.) — dead code, never called
  - _chunk_text — replaced by text_chunker.chunk_text
  - _classify_chunk — replaced by heading_classifier.classify_chunk
  - enrich_with_unit_names — dead code, never called
  - STRATEGY_HANDLERS dict — dead registry

The segmenter now does all extraction. pdf_agent.py is just assessment +
shared utilities.
"""

import re
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import pdfplumber
from pypdf import PdfReader
from PIL import Image, ImageFilter, ImageEnhance

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants (exported to segmenter and other modules)
# ---------------------------------------------------------------------------

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
OCR_DPI_STD   = 200
OCR_DPI_DEEP  = 300

WH40K_STAT_HEADERS = [
    "M", "T", "SV", "W", "LD", "OC",
    "RANGE", "A", "BS", "WS", "S", "AP", "D",
]

STAT_BLOCK_SIGNAL_THRESHOLD = 2


# ---------------------------------------------------------------------------
# Assessment data structure
# ---------------------------------------------------------------------------

@dataclass
class PDFAssessment:
    path: str
    total_pages: int
    has_extractable_text: bool
    text_coverage: float
    has_tables: bool
    has_embedded_images: bool
    fonts_embedded: bool
    garbled_text: bool
    has_stat_blocks: bool
    pdf_type: str
    extraction_strategy: str
    title: Optional[str] = None
    author: Optional[str] = None


# ---------------------------------------------------------------------------
# Assessment logic
# ---------------------------------------------------------------------------

def _run(cmd: list) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return result.stdout
    except Exception:
        return ""


def _is_garbled(text: str) -> bool:
    if not text:
        return False
    non_printable = sum(
        1 for c in text
        if ord(c) > 127 or (ord(c) < 32 and c not in "\n\t\r"))
    return (non_printable / max(len(text), 1)) > 0.15


def _has_stat_block_indicators(text: str, pdf_path: str) -> bool:
    signals = 0
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Signal 1: stat header lines
    stat_header_lines = [
        line for line in lines
        if 2 <= len(line.upper().split()) <= len(WH40K_STAT_HEADERS)
        and all(t in WH40K_STAT_HEADERS for t in line.upper().split())
        and len(line) < 50
    ]
    if len(stat_header_lines) >= 3:
        signals += 1

    # Signal 2: paired header + values
    stat_val_re = re.compile(r'^[\d\+\-\*\"/\s]+$')
    paired = 0
    for i, line in enumerate(lines):
        tokens = line.upper().split()
        if sum(1 for t in tokens if t in WH40K_STAT_HEADERS) >= 3 and i + 1 < len(lines):
            nxt = lines[i + 1]
            if stat_val_re.match(nxt) and len(nxt.split()) >= 3:
                paired += 1
    if paired >= 2:
        signals += 1

    # Signal 3: lots of images + little text
    img_info = _run(["pdfimages", "-list", pdf_path])
    if img_info:
        img_lines = [l for l in img_info.strip().split("\n")
                     if l and not l.startswith("page") and not l.startswith("-")]
        try:
            pages = len(PdfReader(pdf_path).pages)
            images_per_page = len(img_lines) / max(pages, 1)
            text_density = len(text.strip()) / max(pages, 1)
            if images_per_page > 3 and text_density < 150:
                signals += 1
        except Exception:
            pass

    return signals >= STAT_BLOCK_SIGNAL_THRESHOLD


def assess_pdf(pdf_path: str) -> PDFAssessment:
    """Classify a PDF into one of: text, table-heavy, mixed, scanned, graphic-stats."""
    path = Path(pdf_path)
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    meta = reader.metadata or {}
    title = meta.get("/Title") or meta.get("Title")
    author = meta.get("/Author") or meta.get("Author")

    pages_with_text, sample_text = 0, ""
    for page in reader.pages[:min(5, total_pages)]:
        t = page.extract_text() or ""
        if len(t.strip()) >= 50:
            pages_with_text += 1
        sample_text += t

    text_coverage = pages_with_text / min(5, total_pages)
    has_extractable_text = text_coverage > 0.3
    garbled = _is_garbled(sample_text)

    has_tables = False
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:min(3, total_pages)]:
                if page.extract_tables():
                    has_tables = True
                    break
    except Exception:
        pass

    has_embedded_images = bool(_run(["pdfimages", "-list", pdf_path]).strip())
    font_info = _run(["pdffonts", pdf_path])
    fonts_embedded = "yes" in font_info.lower() if font_info else True
    has_stat_blocks = _has_stat_block_indicators(sample_text, pdf_path)

    if has_stat_blocks:                                pdf_type = "graphic-stats"
    elif not has_extractable_text or garbled:          pdf_type = "scanned"
    elif has_tables and text_coverage > 0.7:           pdf_type = "table-heavy"
    elif has_embedded_images and text_coverage < 0.6:  pdf_type = "mixed"
    else:                                              pdf_type = "text"

    # strategy field retained for compat, though the actual strategy now lives
    # in the segmenter's routing logic rather than in a dispatch dict
    strategy_map = {
        "text":          "column_aware_text",
        "table-heavy":   "column_aware_tables",
        "mixed":         "column_aware_mixed",
        "scanned":       "column_aware_ocr",
        "graphic-stats": "column_aware_hybrid",
    }

    log.info(f"[ASSESS] {path.name} — type={pdf_type}")
    return PDFAssessment(
        path=pdf_path, total_pages=total_pages,
        has_extractable_text=has_extractable_text, text_coverage=text_coverage,
        has_tables=has_tables, has_embedded_images=has_embedded_images,
        fonts_embedded=fonts_embedded, garbled_text=garbled,
        has_stat_blocks=has_stat_blocks, pdf_type=pdf_type,
        extraction_strategy=strategy_map[pdf_type],
        title=str(title) if title else None,
        author=str(author) if author else None,
    )


# ---------------------------------------------------------------------------
# Shared helpers used by the segmenter
# ---------------------------------------------------------------------------

def _table_to_markdown(table: list) -> str:
    """Convert a pdfplumber table (list of rows) to pipe-delimited markdown."""
    if not table or not table[0]:
        return ""
    rows = [[str(cell or "").strip() for cell in row] for row in table]
    header, body = rows[0], rows[1:]
    sep = ["---"] * len(header)
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(sep) + " |"]
    for row in body:
        row = row[:len(header)] + [""] * max(0, len(header) - len(row))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """Increase contrast + sharpen for better OCR results."""
    image = image.convert("RGB")
    image = ImageEnhance.Contrast(image).enhance(2.0)
    image = image.filter(ImageFilter.SHARPEN)
    return image.convert("L")


def _score_text(text: str) -> int:
    """Heuristic quality score for text — longer + more stat-headers = better."""
    if not text:
        return 0
    return len(text.strip()) + sum(500 for h in WH40K_STAT_HEADERS if h in text.upper())


def _parse_statlines(text: str, page_num: int) -> list:
    """Extract (unit_name, stat_dict) records from text containing stat rows."""
    records = []
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    stat_val_re = re.compile(r'\b(\d+[\+\-\/]?\"?|\*)\b')
    for i, line in enumerate(lines):
        tokens = line.upper().split()
        hits = sum(1 for t in tokens if t in WH40K_STAT_HEADERS)
        if hits >= 3 and i + 1 < len(lines):
            values = stat_val_re.findall(lines[i + 1])
            if len(values) >= 3:
                headers = [t for t in tokens if t in WH40K_STAT_HEADERS]
                record = {
                    "page": page_num,
                    "raw_header_line": line,
                    "raw_value_line": lines[i + 1],
                    "stats": dict(zip(headers, values)),
                }
                for back in range(1, 4):
                    candidate = lines[i - back] if i - back >= 0 else ""
                    if len(candidate) > 2 and re.match(r"^[A-Za-z\s'\-]+$", candidate):
                        record["unit_name"] = candidate.strip()
                        break
                records.append(record)
    return records

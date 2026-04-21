"""
pdf_region_segmenter.py
=======================
Sub-page region detection and extraction — COLUMN-AWARE refactor.

Location: C:\\Projects\\wh40k-app\\pdf_agent\\pdf_region_segmenter.py

Changes from previous version:
  - NEW: per-page column detection runs FIRST, producing ColumnRegion crops
  - NEW: geometric region detection runs INSIDE each column, not across the page
  - NEW: classification uses the new heading_classifier module
  - REMOVED: enrich_with_unit_names dead-code path
  - REMOVED: the bleed-prone in-segmenter carry-forward (handled later in ingest)

Three-pass per column:
  Pass 1 — Geometric boundary detection (image bboxes, rect dividers, whitespace)
  Pass 2 — Content signal verification (crops + re-classifies)
  Pass 3 — Per-region extraction routing (best method per section type)

Chunk output schema (populated by this module):
  chunk_id              — set later by ingest.py via make_chunk_id()
  source_file           — PDF path
  page_number           — 1-based
  region_index          — within page
  column_label          — "single" | "left" | "right"
  chunk_index           — within region
  text                  — the extracted text
  content_type          — "text" | "table" | "ocr_text" | "artwork"
  extraction_method     — how the text was produced
  bbox                  — (x0, y0, x1, y1) on the source page
  geometric_source      — how the region was detected
  content_confirmed     — whether extraction confirmed classification
  ocr_confidence        — 0..1 for OCR chunks, None otherwise
  table_count           — number of tables in the region
  word_count            — whitespace-delimited word count
  is_pure_artwork       — True for image-only regions
  statlines             — list of parsed stat rows for datasheets
  classification        — HeadingResult from heading_classifier (for later carry-forward)
  metadata              — pdf-level metadata dict
"""

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_OK = True
except ImportError:
    PDF2IMAGE_OK = False

try:
    import pytesseract
    from PIL import Image
    TESSERACT_OK = True
except ImportError:
    TESSERACT_OK = False

try:
    import fitz
    PYMUPDF_OK = True
except ImportError:
    PYMUPDF_OK = False

import sys
sys.path.insert(0, str(Path(__file__).parent))

from pdf_agent import (
    PDFAssessment, _table_to_markdown, _preprocess_for_ocr,
    _parse_statlines, _score_text, WH40K_STAT_HEADERS,
    OCR_DPI_DEEP, OCR_DPI_STD,
)
from column_detection import detect_column_layout, ColumnRegion, format_layout_summary
from heading_classifier import classify_chunk, HeadingResult
from text_chunker import chunk_text, CHUNK_SIZE, CHUNK_OVERLAP

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tuning constants (within-column region detection)
# ---------------------------------------------------------------------------

WHITESPACE_GAP_THRESHOLD = 12.0
RECT_WIDTH_FRACTION      = 0.70
RECT_MIN_HEIGHT          = 4.0
REGION_MIN_HEIGHT        = 20.0
CROP_PADDING             = 2.0
ARTWORK_WORD_THRESHOLD   = 8


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RegionBoundary:
    """A vertical (y-axis) region within a column."""
    y0: float
    y1: float
    x0: float
    x1: float
    geometric_source: str

    @property
    def bbox(self):
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def height(self):
        return self.y1 - self.y0


@dataclass
class PageRegion:
    """A single extracted region ready for chunking."""
    source_file: str
    page_number: int
    region_index: int
    column_label: str
    bbox: tuple
    geometric_source: str
    section_type: str
    classification: HeadingResult
    extraction_method: str
    content_type: str
    text: str
    is_pure_artwork: bool = False
    content_confirmed: bool = False
    ocr_confidence: Optional[float] = None
    table_count: int = 0
    word_count: int = 0
    statlines: list = field(default_factory=list)

    def to_chunk_dicts(self, stem: str, assessment: PDFAssessment) -> list:
        """Split this region's text into chunk dicts. Chunk IDs assigned later by ingest."""
        if not self.text.strip():
            return []
        records = []
        for chunk_idx, chunk_str in enumerate(chunk_text(self.text)):
            # Skip whitespace-only pieces. chunk_text() can emit these when a
            # region is mostly empty (e.g. a large datasheet with sparse text
            # and big whitespace gaps). They carry no information, they bloat
            # the collection, and because make_chunk_id() hashes the first
            # 100 chars they can collide across regions.
            if not chunk_str.strip():
                continue
            records.append({
                # chunk_id will be assigned later by make_chunk_id()
                "source_file":       self.source_file,
                "page_number":       self.page_number,
                "region_index":      self.region_index,
                "column_label":      self.column_label,
                "chunk_index":       chunk_idx,
                "text":              chunk_str,
                "content_type":      self.content_type,
                "extraction_method": self.extraction_method,
                "bbox":              list(self.bbox),
                "geometric_source":  self.geometric_source,
                "content_confirmed": self.content_confirmed,
                "ocr_confidence":    self.ocr_confidence,
                "table_count":       self.table_count,
                "word_count":        len(chunk_str.split()),
                "is_pure_artwork":   self.is_pure_artwork,
                "statlines":         self.statlines,
                # Classification attached here; apply_carry_forward will refine section_type
                "classification":    self.classification,
                "section_type":      self.section_type,        # overwritten later by carry-forward
                "section_identifier": "",                      # filled by carry-forward
                # PDF-level metadata
                "metadata": {
                    "pdf_type":          assessment.pdf_type,
                    "total_pages":       assessment.total_pages,
                    "char_count":        len(chunk_str),
                    "has_tables":        self.table_count > 0,
                    "has_images":        self.is_pure_artwork,
                    "title":             assessment.title,
                    "author":            assessment.author,
                    "chunk_strategy":    f"column_aware size={CHUNK_SIZE} overlap={CHUNK_OVERLAP}",
                },
            })
        return records


# ---------------------------------------------------------------------------
# Pass 1 — Geometric boundary detection (within a column region)
# ---------------------------------------------------------------------------

def _detect_image_regions(page, column: ColumnRegion) -> list:
    """Image bboxes that overlap this column."""
    out = []
    for img in (page.images or []):
        x0, y0 = float(img.get("x0", 0)), float(img.get("y0", 0))
        x1, y1 = float(img.get("x1", page.width)), float(img.get("y1", 0))
        if y0 > y1: y0, y1 = y1, y0
        # Only include if overlaps vertically with column range
        if y1 < column.y0 or y0 > column.y1:
            continue
        if y1 - y0 < REGION_MIN_HEIGHT:
            continue
        # Clip to column bounds
        cx0 = max(column.x0, x0)
        cx1 = min(column.x1, x1)
        if cx1 <= cx0:
            continue
        out.append(RegionBoundary(
            y0=max(column.y0, y0), y1=min(column.y1, y1),
            x0=cx0, x1=cx1, geometric_source="image_bbox"))
    return out


def _detect_rect_dividers(page, column: ColumnRegion) -> list:
    """Rectangles that likely divide sections within the column."""
    page_h = float(page.height)
    col_w = column.x1 - column.x0
    min_w = col_w * RECT_WIDTH_FRACTION
    out = []
    for rect in (page.rects or []):
        w = float(rect.get("width", 0))
        h = float(rect.get("height", 0))
        x0 = float(rect.get("x0", 0))
        y0 = float(rect.get("y0", 0))
        x1 = float(rect.get("x1", page.width))
        y1 = float(rect.get("y1", 0))
        if y0 > y1: y0, y1 = y1, y0
        # Must be roughly within the column
        if x0 < column.x0 - 5 or x1 > column.x1 + 5:
            continue
        if y1 < column.y0 or y0 > column.y1:
            continue
        if w >= min_w and h >= RECT_MIN_HEIGHT and h <= page_h * 0.5:
            out.append(RegionBoundary(
                y0=max(column.y0, y0), y1=min(column.y1, y1),
                x0=max(column.x0, x0), x1=min(column.x1, x1),
                geometric_source="rect_divider"))
    return out


def _detect_whitespace_cuts_in_column(page, column: ColumnRegion) -> list:
    """Find y-coordinate cuts where large vertical whitespace separates content."""
    words = page.extract_words() or []
    # Restrict to words within this column
    col_words = [w for w in words
                 if (w["x0"] + w["x1"]) / 2 >= column.x0
                 and (w["x0"] + w["x1"]) / 2 <= column.x1
                 and float(w["top"]) >= column.y0
                 and float(w["bottom"]) <= column.y1]
    if not col_words:
        return []
    y_mids = sorted((float(w["top"]) + float(w["bottom"])) / 2.0 for w in col_words)
    return [
        (y_mids[i - 1] + y_mids[i]) / 2.0
        for i in range(1, len(y_mids))
        if y_mids[i] - y_mids[i - 1] >= WHITESPACE_GAP_THRESHOLD
    ]


def _cuts_to_regions(cuts: list, column: ColumnRegion) -> list:
    """Convert y-cut points to RegionBoundary objects within the column."""
    boundaries = sorted(set([column.y0] + cuts + [column.y1]))
    return [
        RegionBoundary(y0=boundaries[i], y1=boundaries[i+1],
                       x0=column.x0, x1=column.x1,
                       geometric_source="whitespace")
        for i in range(len(boundaries) - 1)
        if boundaries[i+1] - boundaries[i] >= REGION_MIN_HEIGHT
    ]


def _merge_boundaries_in_column(image_regions, rect_regions, whitespace_regions,
                                 column: ColumnRegion):
    """Combine hard boundaries (image/rect) with whitespace filler regions."""
    hard = sorted(image_regions + rect_regions, key=lambda r: r.y0)

    if not hard:
        return whitespace_regions if whitespace_regions else [
            RegionBoundary(y0=column.y0, y1=column.y1,
                           x0=column.x0, x1=column.x1,
                           geometric_source="full_column")
        ]

    cursor = column.y0
    final = []
    for h in hard:
        gap_y0, gap_y1 = cursor, h.y0
        if gap_y1 - gap_y0 >= REGION_MIN_HEIGHT:
            ws_in_gap = [r for r in whitespace_regions
                         if r.y0 >= gap_y0 and r.y1 <= gap_y1]
            final.extend(ws_in_gap if ws_in_gap else [
                RegionBoundary(y0=gap_y0, y1=gap_y1,
                               x0=column.x0, x1=column.x1,
                               geometric_source="whitespace")
            ])
        final.append(h)
        cursor = h.y1

    if column.y1 - cursor >= REGION_MIN_HEIGHT:
        ws_trailing = [r for r in whitespace_regions
                       if r.y0 >= cursor and r.y1 <= column.y1]
        final.extend(ws_trailing if ws_trailing else [
            RegionBoundary(y0=cursor, y1=column.y1,
                           x0=column.x0, x1=column.x1,
                           geometric_source="whitespace")
        ])

    return sorted(final, key=lambda r: r.y0)


DEDUP_AREA_OVERLAP_THRESHOLD = 0.80  # fraction of smaller region's 2D area

def _area(r) -> float:
    w = max(0.0, r.x1 - r.x0)
    h = max(0.0, r.y1 - r.y0)
    return w * h


def _rect_overlap_area(a, b) -> float:
    x0 = max(a.x0, b.x0); x1 = min(a.x1, b.x1)
    y0 = max(a.y0, b.y0); y1 = min(a.y1, b.y1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _deduplicate_regions(regions: list) -> list:
    """
    Remove near-duplicate regions produced by overlapping PDF image bboxes.

    Why this exists: WH40K PDFs use decorative background images that
    pdfplumber reports as multiple overlapping image bboxes — full-page
    backgrounds, mid-width sidebar ornaments, and content-area rectangles
    all stacked on top of one another. A crop of any of them pulls in
    substantially the same text from the column below, producing duplicate
    chunks.

    Heuristic: if one region's 2D area overlaps ≥80% with a region we've
    already kept, drop it. The sort (earliest y0, then widest) means we
    prefer keeping the region that covers the most content, and discard the
    tighter near-duplicates inside it.

    This is broader than the original "same x range + y overlap" rule, which
    missed the common case where a narrow sidebar-image bbox (x=[21,111])
    sits against a wide content-image bbox (x=[45,349]) with near-identical
    y extents — different x ranges, but crops yield the same body text.
    """
    if len(regions) <= 1:
        return regions
    # Sort: earliest y0 first, then widest (largest area) first on ties.
    # That way the "biggest container" is kept and tighter overlappers that
    # come later are dropped. Within the same y0, widest-first means the
    # full-page background gets kept ahead of the inner content rectangle.
    ordered = sorted(regions, key=lambda r: (r.y0, -_area(r)))
    result = []
    for r in ordered:
        r_area = _area(r)
        if r_area <= 0:
            continue
        is_dup = False
        for kept in result:
            overlap = _rect_overlap_area(r, kept)
            if overlap <= 0:
                continue
            smaller_area = min(r_area, _area(kept))
            if smaller_area > 0 and (overlap / smaller_area) >= DEDUP_AREA_OVERLAP_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            result.append(r)
    return result


def detect_regions_in_column(page, column: ColumnRegion) -> list:
    """Run all three region-detection strategies within a single column."""
    image_regions = _detect_image_regions(page, column)
    rect_regions = _detect_rect_dividers(page, column)
    whitespace_cuts = _detect_whitespace_cuts_in_column(page, column)
    whitespace_regions = _cuts_to_regions(whitespace_cuts, column)
    merged = _merge_boundaries_in_column(image_regions, rect_regions,
                                          whitespace_regions, column)
    return _deduplicate_regions(merged)


# ---------------------------------------------------------------------------
# Pass 2 — Content signal verification
# ---------------------------------------------------------------------------

_STAT_HEADER_SET = set(WH40K_STAT_HEADERS)
_STAT_VAL_RE     = re.compile(r'^[\d\+\-\*\"\/\s]+$')


def _count_stat_header_lines(text: str) -> int:
    return sum(
        1 for line in text.split("\n")
        if 2 <= len(line.strip().upper().split()) <= len(WH40K_STAT_HEADERS)
        and all(t in _STAT_HEADER_SET for t in line.strip().upper().split())
        and len(line.strip()) < 60
    )


def _count_stat_value_pairs(text: str) -> int:
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    pairs = 0
    for i, line in enumerate(lines):
        tokens = line.upper().split()
        if sum(1 for t in tokens if t in _STAT_HEADER_SET) >= 3 and i + 1 < len(lines):
            nxt = lines[i + 1]
            if _STAT_VAL_RE.match(nxt) and len(nxt.split()) >= 3:
                pairs += 1
    return pairs


def _verify_region_content(page, region: RegionBoundary):
    """Extract a preview and use it to classify the region's content type."""
    page_w, page_h = float(page.width), float(page.height)
    x0 = max(0.0, region.x0 - CROP_PADDING)
    y0 = max(0.0, region.y0 - CROP_PADDING)
    x1 = min(page_w, region.x1 + CROP_PADDING)
    y1 = min(page_h, region.y1 + CROP_PADDING)
    try:
        crop = page.crop((x0, y0, x1, y1))
        text = crop.extract_text(layout=True) or ""
        tables = crop.extract_tables() or []
        table_count = len(tables)
        word_count = len(text.split()) if text else 0
    except Exception as e:
        log.warning(f"    [VERIFY] crop failed: {e}")
        return "no_text_layer", False, "", 0, 0

    # Classify based on content signals
    if region.geometric_source == "image_bbox" and word_count < ARTWORK_WORD_THRESHOLD:
        return "artwork", True, text, table_count, word_count
    if not text or word_count < 3:
        return "no_text_layer", True, text, table_count, word_count

    stat_signals = (
        (1 if _count_stat_header_lines(text) >= 2 else 0)
        + (1 if _count_stat_value_pairs(text) >= 1 else 0)
        + (1 if table_count > 0 else 0)
    )
    if stat_signals >= 2:
        return "unit_datasheet", True, text, table_count, word_count
    if stat_signals >= 1 or table_count > 0:
        return "stat_block", True, text, table_count, word_count
    if region.geometric_source == "rect_divider" and word_count < 5:
        return "artwork", True, text, table_count, word_count

    # Default for prose-ish content
    return "general", True, text, table_count, word_count


# ---------------------------------------------------------------------------
# Pass 3 — Per-region extraction routing
# ---------------------------------------------------------------------------

def _ocr_region_crop(pdf_path, page_number, bbox, page_w, page_h, dpi=OCR_DPI_STD):
    """OCR a region from the PDF. Returns (text, confidence)."""
    x0, y0, x1, y1 = bbox
    if PYMUPDF_OK:
        try:
            doc = fitz.open(pdf_path)
            pg = doc[page_number - 1]
            pix = pg.get_pixmap(matrix=fitz.Matrix(dpi/72.0, dpi/72.0),
                                clip=fitz.Rect(x0, y0, x1, y1))
            img = _preprocess_for_ocr(Image.frombytes(
                "RGB", [pix.width, pix.height], pix.samples))
            if TESSERACT_OK:
                data = pytesseract.image_to_data(
                    img, output_type=pytesseract.Output.DICT, timeout=60)
                confs = [int(c) for c in data["conf"]
                         if str(c).lstrip("-").isdigit() and int(c) >= 0]
                conf = sum(confs) / len(confs) / 100.0 if confs else 0.0
                return pytesseract.image_to_string(img, timeout=60), conf
        except Exception as e:
            log.warning(f"    [OCR-CROP] PyMuPDF failed: {e}")

    if PDF2IMAGE_OK and TESSERACT_OK:
        try:
            images = convert_from_path(pdf_path, first_page=page_number,
                                        last_page=page_number, dpi=dpi)
            if images:
                full_img = images[0]
                sx, sy = full_img.width / page_w, full_img.height / page_h
                region_img = _preprocess_for_ocr(
                    full_img.crop((int(x0*sx), int(y0*sy),
                                   int(x1*sx), int(y1*sy))))
                data = pytesseract.image_to_data(
                    region_img, output_type=pytesseract.Output.DICT, timeout=60)
                confs = [int(c) for c in data["conf"]
                         if str(c).lstrip("-").isdigit() and int(c) >= 0]
                conf = sum(confs) / len(confs) / 100.0 if confs else 0.0
                return pytesseract.image_to_string(region_img, timeout=60), conf
        except Exception as e:
            log.warning(f"    [OCR-CROP] pdf2image fallback failed: {e}")

    return "", 0.0


def _extract_region(pdf_path, page, page_number, region_idx, column: ColumnRegion,
                    boundary: RegionBoundary, verified_type: str, confirmed: bool,
                    base_text: str, table_count: int, word_count: int,
                    assessment: PDFAssessment) -> PageRegion:
    """Run the right extraction strategy for this region's content type."""
    page_w, page_h = float(page.width), float(page.height)
    x0 = max(0.0, boundary.x0 - CROP_PADDING)
    y0 = max(0.0, boundary.y0 - CROP_PADDING)
    x1 = min(page_w, boundary.x1 + CROP_PADDING)
    y1 = min(page_h, boundary.y1 + CROP_PADDING)
    bbox = (x0, y0, x1, y1)

    extracted_text = base_text
    extraction_method = "pdfplumber_text"
    content_type = "text"
    ocr_confidence = None
    statlines = []
    is_pure_artwork = False
    final_table_count = table_count

    if verified_type == "artwork":
        is_pure_artwork = word_count < ARTWORK_WORD_THRESHOLD
        extracted_text = ""
        extraction_method = "skipped_artwork"
        content_type = "artwork"

    elif verified_type == "no_text_layer":
        text, conf = _ocr_region_crop(pdf_path, page_number, bbox, page_w, page_h)
        extracted_text = text
        ocr_confidence = round(conf, 3)
        extraction_method = f"ocr_crop_{OCR_DPI_STD}dpi"
        content_type = "ocr_text"

    elif verified_type == "unit_datasheet":
        # Deep hybrid: compare plumber vs OCR, take the winner
        try:
            crop = page.crop(bbox)
            plumber_text = crop.extract_text(layout=True) or ""
            plumber_tables = crop.extract_tables() or []
            plumber_score = _score_text(plumber_text) + sum(
                _score_text(" ".join(str(c) for row in t for c in row))
                for t in plumber_tables)
        except Exception:
            plumber_text, plumber_tables, plumber_score = base_text, [], _score_text(base_text)

        ocr_text, ocr_conf = _ocr_region_crop(
            pdf_path, page_number, bbox, page_w, page_h, dpi=OCR_DPI_DEEP)
        if _score_text(ocr_text) > plumber_score and ocr_text:
            extracted_text = ocr_text
            extraction_method = "deep_hybrid_ocr_winner"
            content_type = "ocr_text"
            ocr_confidence = round(ocr_conf, 3)
        else:
            parts = [_table_to_markdown(t) for t in plumber_tables
                     if _table_to_markdown(t)]
            if plumber_text.strip():
                parts.append(plumber_text)
            extracted_text = "\n\n".join(parts)
            extraction_method = "deep_hybrid_plumber_winner"
            content_type = "table" if plumber_tables else "text"
            final_table_count = len(plumber_tables)
        statlines = _parse_statlines(extracted_text, page_number)

    elif verified_type == "stat_block":
        try:
            crop = page.crop(bbox)
            tables = crop.extract_tables() or []
            parts = [_table_to_markdown(t) for t in tables if _table_to_markdown(t)]
            raw = crop.extract_text(layout=True) or ""
            if raw.strip():
                parts.append(raw)
            extracted_text = "\n\n".join(parts) if parts else base_text
            final_table_count = len(tables)
        except Exception:
            extracted_text = base_text
        extraction_method = "pdfplumber_table_and_text"
        content_type = "table" if final_table_count > 0 else "text"

    else:  # "general" or anything else — prose extraction, COLUMN-AWARE
        try:
            # NEW: use layout=False when inside a column crop. We're already
            # inside a single-column region, so we don't need layout preservation
            # that causes the column-interleaving problem.
            extracted_text = page.crop(bbox).extract_text(layout=False) or base_text
        except Exception:
            extracted_text = base_text
        extraction_method = "pdfplumber_column_text"

    # Classify AFTER extraction — use the new classifier on clean text
    classification = classify_chunk(extracted_text)

    # Prefer the classifier's section_type if it produced a confident one,
    # otherwise use the content-verification type mapped to our taxonomy.
    if classification.confident:
        section_type = classification.section_type
    else:
        # Map verification types to our canonical taxonomy
        type_map = {
            "unit_datasheet": "unit_datasheet",
            "stat_block":     "general",  # unlabeled stat-like content
            "artwork":        "general",
            "no_text_layer":  "general",
            "general":        classification.section_type,  # use classifier's best guess
        }
        section_type = type_map.get(verified_type, "general")

    return PageRegion(
        source_file=pdf_path, page_number=page_number, region_index=region_idx,
        column_label=column.column_label,
        bbox=bbox, geometric_source=boundary.geometric_source,
        section_type=section_type,
        classification=classification,
        extraction_method=extraction_method, content_type=content_type,
        text=extracted_text, is_pure_artwork=is_pure_artwork,
        content_confirmed=confirmed, ocr_confidence=ocr_confidence,
        table_count=final_table_count,
        word_count=len(extracted_text.split()) if extracted_text else 0,
        statlines=statlines,
    )


# ---------------------------------------------------------------------------
# Page and document entry points
# ---------------------------------------------------------------------------

def segment_page_into_regions(pdf_path, page, page_number, assessment):
    """
    Segment a page with column-awareness.

    For each column:
      1. Detect regions within the column
      2. Verify content of each region
      3. Extract each region
    Regions across all columns are combined and returned in reading order
    (top-to-bottom across columns: full single, then left, then right).
    """
    # Pass 0 (new): detect column layout for this page
    layout = detect_column_layout(page)
    log.debug(f"  page {page_number}: {format_layout_summary(layout)}")

    all_regions = []
    region_idx = 0

    for column in layout.regions:
        # Pass 1: detect y-regions within this column
        boundaries = detect_regions_in_column(page, column)
        # Pass 2: verify content of each region
        verified = [(b, *_verify_region_content(page, b)) for b in boundaries]
        # Pass 3: extract each region
        for boundary, vtype, confirmed, base_text, tcount, wcount in verified:
            try:
                region = _extract_region(
                    pdf_path=pdf_path, page=page, page_number=page_number,
                    region_idx=region_idx, column=column, boundary=boundary,
                    verified_type=vtype, confirmed=confirmed, base_text=base_text,
                    table_count=tcount, word_count=wcount, assessment=assessment,
                )
                all_regions.append(region)
                region_idx += 1
            except Exception as e:
                log.warning(f"  [REGION] skipping region on page {page_number}: "
                            f"{type(e).__name__}: {e}")
                continue

    return all_regions


def segment_document_into_regions(pdf_path, assessment, max_pages=None):
    """
    Segment all pages of a PDF. Returns (chunks, all_statlines).

    chunks is a list of dicts matching the schema at the top of this file.
    The chunks do NOT yet have chunk_id set — ingest.py assigns those.
    They also do not have final section_identifier values — those come from
    apply_carry_forward() which ingest.py runs after all chunks are collected.
    """
    stem = Path(pdf_path).stem
    pages_to_do = min(max_pages, assessment.total_pages) if max_pages else assessment.total_pages
    all_chunks, all_statlines = [], []

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx in range(pages_to_do):
            page_number = page_idx + 1
            try:
                page = pdf.pages[page_idx]
                regions = segment_page_into_regions(pdf_path, page, page_number, assessment)
                for region in regions:
                    if region.is_pure_artwork or not region.text.strip():
                        continue
                    all_chunks.extend(region.to_chunk_dicts(stem, assessment))
                    all_statlines.extend(region.statlines)
            except Exception as e:
                log.warning(f"  [REGION-DOC] skipping page {page_number} of {stem}: "
                            f"{type(e).__name__}: {e}")
                continue

    log.info(f"[REGION-DOC] {stem}: {len(all_chunks)} chunks from {pages_to_do} pages")
    return all_chunks, all_statlines

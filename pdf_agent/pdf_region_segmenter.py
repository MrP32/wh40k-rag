"""
pdf_region_segmenter.py
=======================
Sub-page region detection and extraction.
Location: C:\Projects\wh40k-app\pdf_agent\pdf_region_segmenter.py

Three-pass approach per page:
  Pass 1 — Geometric boundary detection (image bboxes, rect dividers, whitespace gaps)
  Pass 2 — Content signal verification (crops + re-classifies each region)
  Pass 3 — Per-region extraction routing (best method per section type)

Chunk ID format: {stem}_p{page:04d}_r{region:02d}_c{chunk:03d}
"""

import re
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber
from pypdf import PdfReader

try:
    from pdf2image import convert_from_path
    PDF2IMAGE_OK = True
except ImportError:
    PDF2IMAGE_OK = False

try:
    import pytesseract
    from PIL import Image, ImageFilter, ImageEnhance
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
    PDFAssessment, _chunk_text, _table_to_markdown, _classify_chunk,
    _preprocess_for_ocr, _parse_statlines, _score_text, _to_title,
    WH40K_STAT_HEADERS, OCR_DPI_DEEP, OCR_DPI_STD, CHUNK_SIZE, CHUNK_OVERLAP,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
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
    source_file: str
    page_number: int
    region_index: int
    bbox: tuple
    geometric_source: str
    section_type: str
    section_identifier: str
    section_identifier_clean: str
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
        if not self.text.strip():
            return []
        records = []
        for chunk_idx, chunk_text in enumerate(_chunk_text(self.text)):
            records.append({
                "chunk_id":                 f"{stem}_p{self.page_number:04d}_r{self.region_index:02d}_c{chunk_idx:03d}",
                "source_file":              self.source_file,
                "page_number":              self.page_number,
                "region_index":             self.region_index,
                "chunk_index":              chunk_idx,
                "section_type":             self.section_type,
                "section_identifier":       self.section_identifier,
                "section_identifier_clean": self.section_identifier_clean,
                "extraction_method":        self.extraction_method,
                "content_type":             self.content_type,
                "text":                     chunk_text,
                "bbox":                     list(self.bbox),
                "metadata": {
                    "pdf_type":          assessment.pdf_type,
                    "total_pages":       assessment.total_pages,
                    "char_count":        len(chunk_text),
                    "word_count":        len(chunk_text.split()),
                    "has_tables":        self.table_count > 0,
                    "has_images":        self.is_pure_artwork,
                    "ocr_confidence":    self.ocr_confidence,
                    "title":             assessment.title,
                    "author":            assessment.author,
                    "geometric_source":  self.geometric_source,
                    "content_confirmed": self.content_confirmed,
                    "is_pure_artwork":   self.is_pure_artwork,
                    "chunk_strategy":    f"sub_page_region size={CHUNK_SIZE} overlap={CHUNK_OVERLAP}",
                },
            })
        return records


# ---------------------------------------------------------------------------
# Pass 1 — Geometric boundary detection
# ---------------------------------------------------------------------------

def _detect_image_regions(page) -> list:
    regions, page_w = [], float(page.width)
    for img in (page.images or []):
        x0, y0 = float(img.get("x0", 0)), float(img.get("y0", 0))
        x1, y1 = float(img.get("x1", page_w)), float(img.get("y1", 0))
        if y0 > y1: y0, y1 = y1, y0
        if y1 - y0 >= REGION_MIN_HEIGHT:
            regions.append(RegionBoundary(y0=y0, y1=y1, x0=x0, x1=x1, geometric_source="image_bbox"))
    return regions


def _detect_rect_dividers(page) -> list:
    regions, page_w, page_h = [], float(page.width), float(page.height)
    min_w = page_w * RECT_WIDTH_FRACTION
    for rect in (page.rects or []):
        w  = float(rect.get("width",  0))
        h  = float(rect.get("height", 0))
        x0 = float(rect.get("x0",    0))
        y0 = float(rect.get("y0",    0))
        x1 = float(rect.get("x1",    page_w))
        y1 = float(rect.get("y1",    0))
        if y0 > y1: y0, y1 = y1, y0
        if w >= min_w and h >= RECT_MIN_HEIGHT and h <= page_h * 0.5:
            regions.append(RegionBoundary(y0=y0, y1=y1, x0=x0, x1=x1, geometric_source="rect_divider"))
    return regions


def _detect_whitespace_gaps(page) -> list:
    words = page.extract_words() or []
    if not words:
        return []
    y_mids = sorted((float(w["top"]) + float(w["bottom"])) / 2.0 for w in words)
    return [
        (y_mids[i - 1] + y_mids[i]) / 2.0
        for i in range(1, len(y_mids))
        if y_mids[i] - y_mids[i - 1] >= WHITESPACE_GAP_THRESHOLD
    ]


def _gaps_to_regions(cut_points: list, page_w: float, page_h: float) -> list:
    boundaries = sorted(set([0.0] + cut_points + [page_h]))
    return [
        RegionBoundary(y0=boundaries[i], y1=boundaries[i+1], x0=0.0, x1=page_w,
                       geometric_source="whitespace")
        for i in range(len(boundaries) - 1)
        if boundaries[i+1] - boundaries[i] >= REGION_MIN_HEIGHT
    ]


def _merge_boundaries(image_regions, rect_regions, whitespace_regions, page_w, page_h):
    hard   = sorted(image_regions + rect_regions, key=lambda r: r.y0)
    cursor, final = 0.0, []

    for h_region in hard:
        gap_y0, gap_y1 = cursor, h_region.y0
        if gap_y1 - gap_y0 >= REGION_MIN_HEIGHT:
            ws_in_gap = [r for r in whitespace_regions if r.y0 >= gap_y0 and r.y1 <= gap_y1]
            final.extend(ws_in_gap if ws_in_gap else [
                RegionBoundary(y0=gap_y0, y1=gap_y1, x0=0.0, x1=page_w, geometric_source="whitespace")])
        final.append(h_region)
        cursor = h_region.y1

    if page_h - cursor >= REGION_MIN_HEIGHT:
        ws_trailing = [r for r in whitespace_regions if r.y0 >= cursor and r.y1 <= page_h]
        final.extend(ws_trailing if ws_trailing else [
            RegionBoundary(y0=cursor, y1=page_h, x0=0.0, x1=page_w, geometric_source="whitespace")])

    if not hard:
        final = whitespace_regions if whitespace_regions else [
            RegionBoundary(y0=0.0, y1=page_h, x0=0.0, x1=page_w, geometric_source="full_page")]

    return sorted(final, key=lambda r: r.y0)


def detect_regions_geometric(page) -> list:
    page_w, page_h     = float(page.width), float(page.height)
    image_regions      = _detect_image_regions(page)
    rect_regions       = _detect_rect_dividers(page)
    whitespace_cuts    = _detect_whitespace_gaps(page)
    whitespace_regions = _gaps_to_regions(whitespace_cuts, page_w, page_h)
    return _merge_boundaries(image_regions, rect_regions, whitespace_regions, page_w, page_h)


# ---------------------------------------------------------------------------
# Pass 2 — Content signal verification
# ---------------------------------------------------------------------------

_STAT_HEADER_SET = set(WH40K_STAT_HEADERS)
_RULES_RE        = re.compile(
    r'\b(stratagem|command point|detachment rule|army rule|enhancement|warlord trait|'
    r'psychic|litany|aura|special rule|phase|ability|abilities)\b', re.IGNORECASE)
_NARRATIVE_RE    = re.compile(
    r'\b(lore|legend|history|ancient|century|millennia|saga|myth|'
    r'imperium|chaos|xenos|heresy|galaxy|battle.worn)\b', re.IGNORECASE)
_STAT_VAL_RE     = re.compile(r'^[\d\+\-\*\"\/\s]+$')


def _count_stat_header_lines(text: str) -> int:
    return sum(
        1 for line in text.split("\n")
        if 2 <= len(line.strip().upper().split()) <= len(WH40K_STAT_HEADERS)
        and all(t in _STAT_HEADER_SET for t in line.strip().upper().split())
        and len(line.strip()) < 60
    )


def _count_stat_value_pairs(text: str) -> int:
    lines, pairs = [l.strip() for l in text.split("\n") if l.strip()], 0
    for i, line in enumerate(lines):
        tokens = line.upper().split()
        if sum(1 for t in tokens if t in _STAT_HEADER_SET) >= 3 and i + 1 < len(lines):
            nxt = lines[i + 1]
            if _STAT_VAL_RE.match(nxt) and len(nxt.split()) >= 3:
                pairs += 1
    return pairs


def _classify_region_text(text, table_count, geometric_source, word_count):
    if geometric_source == "image_bbox" and word_count < ARTWORK_WORD_THRESHOLD:
        return "artwork", True
    confirmed_by_geo = geometric_source != "image_bbox"
    if not text or word_count < 3:
        return "no_text_layer", True

    stat_signals = (
        (1 if _count_stat_header_lines(text) >= 2 else 0)
        + (1 if _count_stat_value_pairs(text)  >= 1 else 0)
        + (1 if table_count > 0 else 0)
    )
    rules_hits     = len(_RULES_RE.findall(text))
    narrative_hits = len(_NARRATIVE_RE.findall(text))
    long_paras     = len([l for l in text.split("\n") if len(l.split()) > 15])

    if stat_signals >= 2:   return "unit_datasheet", confirmed_by_geo
    if stat_signals >= 1 or table_count > 0: return "stat_block", confirmed_by_geo
    if rules_hits >= 2:     return "rules_text",    confirmed_by_geo
    if narrative_hits >= 1 or long_paras >= 2: return "narrative", confirmed_by_geo
    if geometric_source == "rect_divider" and word_count < 5: return "artwork", True

    return "general", confirmed_by_geo


def verify_regions(page, candidates):
    results, page_w, page_h = [], float(page.width), float(page.height)
    for i, region in enumerate(candidates):
        x0 = max(0.0, region.x0 - CROP_PADDING)
        y0 = max(0.0, region.y0 - CROP_PADDING)
        x1 = min(page_w, region.x1 + CROP_PADDING)
        y1 = min(page_h, region.y1 + CROP_PADDING)
        try:
            crop        = page.crop((x0, y0, x1, y1))
            text        = crop.extract_text(layout=True) or ""
            tables      = crop.extract_tables() or []
            table_count = len(tables)
            word_count  = len(text.split()) if text else 0
        except Exception as e:
            log.warning(f"    [VERIFY] crop failed region {i}: {e}")
            text, table_count, word_count = "", 0, 0

        section_type, confirmed = _classify_region_text(
            text, table_count, region.geometric_source, word_count)
        results.append((region, section_type, confirmed, text, table_count, word_count))
    return results


# ---------------------------------------------------------------------------
# Pass 3 — Per-region extraction routing
# ---------------------------------------------------------------------------

def _ocr_region_crop(pdf_path, page_number, bbox, page_w, page_h, dpi=OCR_DPI_STD):
    x0, y0, x1, y1 = bbox
    if PYMUPDF_OK:
        try:
            doc  = fitz.open(pdf_path)
            pg   = doc[page_number - 1]
            pix  = pg.get_pixmap(matrix=fitz.Matrix(dpi/72.0, dpi/72.0), clip=fitz.Rect(x0,y0,x1,y1))
            img  = _preprocess_for_ocr(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
            if TESSERACT_OK:
                # timeout=30 prevents tesseract hanging on complex regions
                data  = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, timeout=30)
                confs = [int(c) for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) >= 0]
                conf  = sum(confs) / len(confs) / 100.0 if confs else 0.0
                return pytesseract.image_to_string(img, timeout=30), conf
        except Exception as e:
            log.warning(f"    [OCR-CROP] PyMuPDF failed: {e}")

    if PDF2IMAGE_OK and TESSERACT_OK:
        try:
            images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number, dpi=dpi)
            if images:
                full_img   = images[0]
                sx, sy     = full_img.width / page_w, full_img.height / page_h
                region_img = _preprocess_for_ocr(
                    full_img.crop((int(x0*sx), int(y0*sy), int(x1*sx), int(y1*sy))))
                data  = pytesseract.image_to_data(region_img, output_type=pytesseract.Output.DICT, timeout=30)
                confs = [int(c) for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) >= 0]
                conf  = sum(confs) / len(confs) / 100.0 if confs else 0.0
                return pytesseract.image_to_string(region_img, timeout=30), conf
        except Exception as e:
            log.warning(f"    [OCR-CROP] pdf2image fallback failed: {e}")

    return "", 0.0


def _extract_region(pdf_path, page, page_number, region_idx, boundary,
                    section_type, content_confirmed, base_text,
                    table_count, word_count, assessment):
    page_w, page_h = float(page.width), float(page.height)
    x0 = max(0.0,    boundary.x0 - CROP_PADDING)
    y0 = max(0.0,    boundary.y0 - CROP_PADDING)
    x1 = min(page_w, boundary.x1 + CROP_PADDING)
    y1 = min(page_h, boundary.y1 + CROP_PADDING)
    bbox = (x0, y0, x1, y1)

    extracted_text    = base_text
    extraction_method = "pdfplumber_text"
    content_type      = "text"
    ocr_confidence    = None
    statlines         = []
    is_pure_artwork   = False
    final_table_count = table_count
    section_type_out  = section_type

    if section_type == "artwork":
        is_pure_artwork   = word_count < ARTWORK_WORD_THRESHOLD
        extracted_text    = ""
        extraction_method = "skipped_artwork"
        content_type      = "artwork"

    elif section_type == "no_text_layer":
        text, conf        = _ocr_region_crop(pdf_path, page_number, bbox, page_w, page_h)
        extracted_text    = text
        ocr_confidence    = round(conf, 3)
        extraction_method = f"ocr_crop_{OCR_DPI_STD}dpi"
        content_type      = "ocr_text"

    elif section_type == "unit_datasheet":
        try:
            crop           = page.crop(bbox)
            plumber_text   = crop.extract_text(layout=True) or ""
            plumber_tables = crop.extract_tables() or []
            plumber_score  = _score_text(plumber_text) + sum(
                _score_text(" ".join(str(c) for row in t for c in row)) for t in plumber_tables)
        except Exception:
            plumber_text, plumber_tables, plumber_score = base_text, [], _score_text(base_text)

        ocr_text, ocr_conf = _ocr_region_crop(pdf_path, page_number, bbox, page_w, page_h, dpi=OCR_DPI_DEEP)
        if _score_text(ocr_text) > plumber_score and ocr_text:
            extracted_text    = ocr_text
            extraction_method = "deep_hybrid_ocr_winner"
            content_type      = "ocr_text"
            ocr_confidence    = round(ocr_conf, 3)
        else:
            parts = [_table_to_markdown(t) for t in plumber_tables if _table_to_markdown(t)]
            if plumber_text.strip():
                parts.append(plumber_text)
            extracted_text    = "\n\n".join(parts)
            extraction_method = "deep_hybrid_plumber_winner"
            content_type      = "table" if plumber_tables else "text"
            final_table_count = len(plumber_tables)
        statlines = _parse_statlines(extracted_text, page_number)

    elif section_type == "stat_block":
        try:
            crop   = page.crop(bbox)
            tables = crop.extract_tables() or []
            parts  = [_table_to_markdown(t) for t in tables if _table_to_markdown(t)]
            raw    = crop.extract_text(layout=True) or ""
            if raw.strip():
                parts.append(raw)
            extracted_text    = "\n\n".join(parts) if parts else base_text
            final_table_count = len(tables)
        except Exception:
            extracted_text = base_text
        extraction_method = "pdfplumber_table_and_text"
        content_type      = "table" if final_table_count > 0 else "text"

    elif section_type in ("rules_text", "narrative", "general"):
        try:
            extracted_text = page.crop(bbox).extract_text(layout=True) or base_text
        except Exception:
            extracted_text = base_text
        extraction_method = "pdfplumber_layout_text"

    # Enrich identifier
    identifier, identifier_clean = "", ""
    if extracted_text.strip():
        try:
            s_type, s_id, _ = _classify_chunk(extracted_text)
            section_type_out = s_type if s_type != "general" else section_type_out
            identifier       = s_id
            identifier_clean = _to_title(s_id) if s_id and s_id != "general" else s_id
        except Exception:
            pass

    return PageRegion(
        source_file=pdf_path, page_number=page_number, region_index=region_idx,
        bbox=bbox, geometric_source=boundary.geometric_source,
        section_type=section_type_out, section_identifier=identifier,
        section_identifier_clean=identifier_clean, extraction_method=extraction_method,
        content_type=content_type, text=extracted_text, is_pure_artwork=is_pure_artwork,
        content_confirmed=content_confirmed, ocr_confidence=ocr_confidence,
        table_count=final_table_count,
        word_count=len(extracted_text.split()) if extracted_text else 0,
        statlines=statlines,
    )


# ---------------------------------------------------------------------------
# Page and document entry points
# ---------------------------------------------------------------------------

def segment_page_into_regions(pdf_path, page, page_number, assessment):
    candidates = detect_regions_geometric(page)
    verified   = verify_regions(page, candidates)
    regions    = []
    for r_idx, (boundary, section_type, confirmed, base_text, tbl_cnt, word_cnt) in enumerate(verified):
        try:
            region = _extract_region(
                pdf_path=pdf_path, page=page, page_number=page_number,
                region_idx=r_idx, boundary=boundary, section_type=section_type,
                content_confirmed=confirmed, base_text=base_text,
                table_count=tbl_cnt, word_count=word_cnt, assessment=assessment,
            )
            regions.append(region)
        except BaseException as e:
            log.warning(f"  [REGION] Skipping region {r_idx} on page {page_number}: {type(e).__name__}: {e}")
            continue
    return regions


def segment_document_into_regions(pdf_path, assessment, max_pages=None):
    stem        = Path(pdf_path).stem
    pages_to_do = min(max_pages, assessment.total_pages) if max_pages else assessment.total_pages
    all_chunks, all_statlines = [], []

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx in range(pages_to_do):
            page_number = page_idx + 1
            try:
                page    = pdf.pages[page_idx]
                regions = segment_page_into_regions(pdf_path, page, page_number, assessment)
                for region in regions:
                    if region.is_pure_artwork or not region.text.strip():
                        continue
                    all_chunks.extend(region.to_chunk_dicts(stem, assessment))
                    all_statlines.extend(region.statlines)
            except BaseException as e:
                log.warning(f"  [REGION-DOC] Skipping page {page_number} of {stem}: {type(e).__name__}: {e}")
                continue

    log.info(f"[REGION-DOC] {stem}: {len(all_chunks)} chunks from {pages_to_do} pages")
    return all_chunks, all_statlines

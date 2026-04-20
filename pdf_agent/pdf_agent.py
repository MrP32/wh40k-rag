"""
pdf_agent.py
============
PDF extraction agent for the Warhammer 40K RAG system.
Location: C:\Projects\wh40k-app\pdf_agent\pdf_agent.py

Extraction strategies:
  pdfplumber_text            — clean text-based PDFs
  pdfplumber_tables          — table-heavy PDFs
  pdfplumber_text+ocr_images — mixed PDFs
  ocr_full                   — fully scanned PDFs
  deep_hybrid                — graphic-embedded stat tables (WH40K datasheets)

Section type classifications:
  unit_datasheet | stratagem | objective | ability |
  enhancement | rules | narrative | general
"""

import os
import re
import json
import logging
import subprocess
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import pdfplumber
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract
from PIL import Image, ImageFilter, ImageEnhance

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
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
# Data structures
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


@dataclass
class DocumentChunk:
    chunk_id: str
    source_file: str
    page_number: int
    chunk_index: int
    content_type: str
    text: str
    metadata: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


# ---------------------------------------------------------------------------
# Assessment
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
    non_printable = sum(1 for c in text if ord(c) > 127 or (ord(c) < 32 and c not in "\n\t\r"))
    return (non_printable / max(len(text), 1)) > 0.15


def _has_stat_block_indicators(text: str, pdf_path: str) -> bool:
    signals = 0
    lines   = [l.strip() for l in text.split("\n") if l.strip()]

    stat_header_lines = [
        line for line in lines
        if 2 <= len(line.upper().split()) <= len(WH40K_STAT_HEADERS)
        and all(t in WH40K_STAT_HEADERS for t in line.upper().split())
        and len(line) < 50
    ]
    if len(stat_header_lines) >= 3:
        signals += 1

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

    img_info = _run(["pdfimages", "-list", pdf_path])
    if img_info:
        img_lines = [l for l in img_info.strip().split("\n")
                     if l and not l.startswith("page") and not l.startswith("-")]
        try:
            pages           = len(PdfReader(pdf_path).pages)
            images_per_page = len(img_lines) / max(pages, 1)
            text_density    = len(text.strip()) / max(pages, 1)
            if images_per_page > 3 and text_density < 150:
                signals += 1
        except Exception:
            pass

    return signals >= STAT_BLOCK_SIGNAL_THRESHOLD


def assess_pdf(pdf_path: str) -> PDFAssessment:
    path   = Path(pdf_path)
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    meta   = reader.metadata or {}
    title  = meta.get("/Title")  or meta.get("Title")
    author = meta.get("/Author") or meta.get("Author")

    pages_with_text, sample_text = 0, ""
    for page in reader.pages[:min(5, total_pages)]:
        t = page.extract_text() or ""
        if len(t.strip()) >= 50:
            pages_with_text += 1
        sample_text += t

    text_coverage        = pages_with_text / min(5, total_pages)
    has_extractable_text = text_coverage > 0.3
    garbled              = _is_garbled(sample_text)

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
    font_info           = _run(["pdffonts", pdf_path])
    fonts_embedded      = "yes" in font_info.lower() if font_info else True
    has_stat_blocks     = _has_stat_block_indicators(sample_text, pdf_path)

    if has_stat_blocks:                                    pdf_type = "graphic-stats"
    elif not has_extractable_text or garbled:              pdf_type = "scanned"
    elif has_tables and text_coverage > 0.7:               pdf_type = "table-heavy"
    elif has_embedded_images and text_coverage < 0.6:      pdf_type = "mixed"
    else:                                                  pdf_type = "text"

    strategy_map = {
        "text":          "pdfplumber_text",
        "table-heavy":   "pdfplumber_tables",
        "mixed":         "pdfplumber_text+ocr_images",
        "scanned":       "ocr_full",
        "graphic-stats": "deep_hybrid",
    }

    log.info(f"[ASSESS] {path.name} — type={pdf_type} strategy={strategy_map[pdf_type]}")
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
# Shared helpers
# ---------------------------------------------------------------------------

def _chunk_text(text: str) -> list:
    text = text.strip()
    if not text:
        return []
    if len(text) <= CHUNK_SIZE:
        return [text]
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        if end < len(text):
            bp = text.rfind(" ", start, end)
            if bp > start:
                end = bp
        chunks.append(text[start:end].strip())
        start = end - CHUNK_OVERLAP
    return [c for c in chunks if c]


def _make_chunk(source_file, page_number, chunk_index, text, content_type,
                assessment, ocr_confidence=None, extraction_method_override=None):
    stem = Path(source_file).stem
    return DocumentChunk(
        chunk_id=f"{stem}_p{page_number:04d}_c{chunk_index:03d}",
        source_file=source_file, page_number=page_number,
        chunk_index=chunk_index, content_type=content_type, text=text,
        metadata={
            "pdf_type":          assessment.pdf_type,
            "extraction_method": extraction_method_override or assessment.extraction_strategy,
            "total_pages":       assessment.total_pages,
            "char_count":        len(text),
            "has_tables":        assessment.has_tables,
            "has_images":        assessment.has_embedded_images,
            "ocr_confidence":    ocr_confidence,
            "title":             assessment.title,
            "author":            assessment.author,
            "chunk_strategy":    f"size={CHUNK_SIZE},overlap={CHUNK_OVERLAP}",
        },
    )


def _table_to_markdown(table: list) -> str:
    if not table or not table[0]:
        return ""
    rows   = [[str(cell or "").strip() for cell in row] for row in table]
    header, body = rows[0], rows[1:]
    sep    = ["---"] * len(header)
    lines  = ["| " + " | ".join(header) + " |", "| " + " | ".join(sep) + " |"]
    for row in body:
        row = row[:len(header)] + [""] * max(0, len(header) - len(row))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Extraction strategies
# ---------------------------------------------------------------------------

def extract_pdfplumber_text(assessment: PDFAssessment, **_) -> list:
    chunks = []
    with pdfplumber.open(assessment.path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for idx, part in enumerate(_chunk_text(page.extract_text() or "")):
                chunks.append(_make_chunk(assessment.path, page_num, idx, part, "text", assessment))
    return chunks


def extract_pdfplumber_tables(assessment: PDFAssessment, **_) -> list:
    chunks = []
    with pdfplumber.open(assessment.path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            idx = 0
            for tbl in (page.extract_tables() or []):
                md = _table_to_markdown(tbl)
                if md:
                    chunks.append(_make_chunk(assessment.path, page_num, idx, md, "table", assessment))
                    idx += 1
            for part in _chunk_text(page.extract_text() or ""):
                chunks.append(_make_chunk(assessment.path, page_num, idx, part, "text", assessment))
                idx += 1
    return chunks


def _ocr_page(source, page_num, assessment, start_idx=0, dpi=OCR_DPI_STD):
    chunks = []
    try:
        images = convert_from_path(source, first_page=page_num, last_page=page_num, dpi=dpi)
        if not images:
            return chunks, start_idx
        img   = images[0]
        data  = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        confs = [int(c) for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) >= 0]
        conf  = sum(confs) / len(confs) / 100.0 if confs else 0.0
        text  = pytesseract.image_to_string(img)
        for idx, part in enumerate(_chunk_text(text), start=start_idx):
            chunks.append(_make_chunk(source, page_num, idx, part, "ocr_text", assessment,
                                      ocr_confidence=conf))
            start_idx = idx + 1
    except Exception as e:
        log.warning(f"OCR failed page {page_num}: {e}")
    return chunks, start_idx


def extract_mixed(assessment: PDFAssessment, **_) -> list:
    chunks = []
    with pdfplumber.open(assessment.path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            idx  = 0
            text = page.extract_text() or ""
            if len(text.strip()) >= 50:
                for tbl in (page.extract_tables() or []):
                    md = _table_to_markdown(tbl)
                    if md:
                        chunks.append(_make_chunk(assessment.path, page_num, idx, md, "table", assessment))
                        idx += 1
                for part in _chunk_text(text):
                    chunks.append(_make_chunk(assessment.path, page_num, idx, part, "text", assessment))
                    idx += 1
            else:
                ocr_chunks, _ = _ocr_page(assessment.path, page_num, assessment, start_idx=idx)
                chunks.extend(ocr_chunks)
    return chunks


def extract_ocr_full(assessment: PDFAssessment, **_) -> list:
    chunks = []
    for page_num in range(1, assessment.total_pages + 1):
        page_chunks, _ = _ocr_page(assessment.path, page_num, assessment)
        chunks.extend(page_chunks)
    return chunks


def _preprocess_for_ocr(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    image = ImageEnhance.Contrast(image).enhance(2.0)
    image = image.filter(ImageFilter.SHARPEN)
    return image.convert("L")


def _parse_statlines(text: str, page_num: int) -> list:
    records     = []
    lines       = [l.strip() for l in text.split("\n") if l.strip()]
    stat_val_re = re.compile(r'\b(\d+[\+\-\/]?\"?|\*)\b')
    for i, line in enumerate(lines):
        tokens = line.upper().split()
        hits   = sum(1 for t in tokens if t in WH40K_STAT_HEADERS)
        if hits >= 3 and i + 1 < len(lines):
            values = stat_val_re.findall(lines[i + 1])
            if len(values) >= 3:
                headers = [t for t in tokens if t in WH40K_STAT_HEADERS]
                record  = {"page": page_num, "raw_header_line": line,
                           "raw_value_line": lines[i + 1], "stats": dict(zip(headers, values))}
                for back in range(1, 4):
                    candidate = lines[i - back] if i - back >= 0 else ""
                    if len(candidate) > 2 and re.match(r"^[A-Za-z\s'\-]+$", candidate):
                        record["unit_name"] = candidate.strip()
                        break
                records.append(record)
    return records


def _score_text(text: str) -> int:
    if not text:
        return 0
    return len(text.strip()) + sum(500 for h in WH40K_STAT_HEADERS if h in text.upper())


def extract_deep_hybrid(assessment: PDFAssessment, output_dir: str = None, **_) -> list:
    source, stem  = assessment.path, Path(assessment.path).stem
    chunks, all_statlines = [], []

    with pdfplumber.open(source) as plumber_pdf:
        for page_num in range(1, assessment.total_pages + 1):
            plumber_text, plumber_tables = "", []
            try:
                page           = plumber_pdf.pages[page_num - 1]
                plumber_text   = page.extract_text() or ""
                plumber_tables = page.extract_tables() or []
            except Exception as e:
                log.warning(f"pdfplumber failed page {page_num}: {e}")

            ocr_text, ocr_confidence = "", 0.0
            try:
                images = convert_from_path(source, first_page=page_num,
                                           last_page=page_num, dpi=OCR_DPI_DEEP)
                if images:
                    img   = _preprocess_for_ocr(images[0])
                    data  = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
                    confs = [int(c) for c in data["conf"]
                             if str(c).lstrip("-").isdigit() and int(c) >= 0]
                    ocr_confidence = sum(confs) / len(confs) / 100.0 if confs else 0.0
                    ocr_text       = pytesseract.image_to_string(img)
            except Exception as e:
                log.warning(f"OCR failed page {page_num}: {e}")

            plumber_score = _score_text(plumber_text) + sum(
                _score_text(" ".join(str(c) for row in t for c in row)) for t in plumber_tables)
            winner = "ocr" if _score_text(ocr_text) > plumber_score else "plumber"
            all_statlines.extend(_parse_statlines(ocr_text if winner == "ocr" else plumber_text, page_num))

            idx = 0
            if winner == "ocr":
                for part in _chunk_text(ocr_text):
                    chunks.append(_make_chunk(source, page_num, idx, part, "ocr_text", assessment,
                        ocr_confidence=round(ocr_confidence, 3), extraction_method_override="ocr_300dpi"))
                    idx += 1
            else:
                for tbl in plumber_tables:
                    md = _table_to_markdown(tbl)
                    if md:
                        chunks.append(_make_chunk(source, page_num, idx, md, "table", assessment,
                            extraction_method_override="pdfplumber_table"))
                        idx += 1
                for part in _chunk_text(plumber_text):
                    chunks.append(_make_chunk(source, page_num, idx, part, "text", assessment,
                        extraction_method_override="pdfplumber_text"))
                    idx += 1

    if output_dir and all_statlines:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(output_dir) / f"{stem}_statlines.json", "w", encoding="utf-8") as f:
            json.dump(all_statlines, f, indent=2, ensure_ascii=False)

    return chunks


# ---------------------------------------------------------------------------
# Content enrichment
# ---------------------------------------------------------------------------

_DATASHEET_HDR_RE = re.compile(
    r"Combat Patrol Datasheet\s*\n([A-Z][A-Z0-9 '\-]+)", re.MULTILINE)
_STAT_ROW_RE = re.compile(
    r'^([A-Z][A-Z0-9 \'\-]{2,})\s*\n[\d]+["\']?\s+\d+\s+\d+\+?\s+\d+\s+\d+\+\s+\d+',
    re.MULTILINE)
_ALLCAPS_HDR_RE = re.compile(
    r'^((?:[A-Z][A-Z0-9]*\s+){1,5}[A-Z][A-Z0-9]*)\s*\n(?=[A-Z][a-z])', re.MULTILINE)

_STRATAGEM_KEYWORDS   = {"STRATAGEM", "STRATAGEMS", "CP", "COMMAND POINT"}
_OBJECTIVE_KEYWORDS   = {"OBJECTIVE", "OBJECTIVES", "SECONDARY OBJECTIVE", "PRIMARY OBJECTIVE", "MISSION"}
_ABILITY_KEYWORDS     = {"ABILITIES", "ABILITY", "SPECIAL RULE", "SPECIAL RULES"}
_ENHANCEMENT_KEYWORDS = {"ENHANCEMENT", "ENHANCEMENTS", "WARLORD TRAIT", "RELIC", "RELICS", "HEIRLOOM"}
_RULES_KEYWORDS       = {"CORE RULES", "UNIVERSAL SPECIAL RULES", "GLOSSARY", "APPENDIX", "REFERENCE"}
_NARRATIVE_SIGNALS    = re.compile(
    r'\b(lore|legend|history|ancient|century|millennia|saga|myth)\b', re.IGNORECASE)

_STRUCTURAL_NOISE = {
    "ABILITIES", "KEYWORDS", "FACTION", "LEADER", "CORE",
    "RANGED WEAPONS", "MELEE WEAPONS", "INVULNERABLE SAVE",
    "FACTION KEYWORDS", "PSYCHIC", "PRECISION", "HAZARDOUS",
    "SUSTAINED HITS", "RAPID FIRE", "DEEP STRIKE", "TELEPORT ASSAULT",
    "FEEL NO PAIN", "COMBAT PATROL", "DATASHEET", "INFANTRY",
    "CHARACTER", "TERMINATOR", "PSYKER", "IMPERIUM", "BATTLELINE",
}


def _to_title(name: str) -> str:
    abbrevs = {"ii", "iii", "iv", "vi", "vii", "viii"}
    return " ".join(w.upper() if w.lower() in abbrevs else w.capitalize() for w in name.split())


def _classify_chunk(text: str) -> tuple:
    text_upper, all_labels = text.upper(), []

    datasheet_names = [m.group(1).strip() for m in _DATASHEET_HDR_RE.finditer(text)
                       if m.group(1).strip() not in _STRUCTURAL_NOISE and len(m.group(1).strip()) > 2]
    stat_names      = [m.group(1).strip() for m in _STAT_ROW_RE.finditer(text)
                       if m.group(1).strip() not in _STRUCTURAL_NOISE and len(m.group(1).strip()) > 2]
    prose_headers   = [m.group(1).strip() for m in _ALLCAPS_HDR_RE.finditer(text)
                       if len(m.group(1).strip()) > 2]

    for name in datasheet_names + stat_names + prose_headers:
        if name not in all_labels:
            all_labels.append(name)

    if datasheet_names or stat_names:
        return "unit_datasheet", (datasheet_names + stat_names)[0], all_labels

    heading_tokens = set()
    for label in prose_headers:
        heading_tokens.update(label.upper().split())
    all_tokens = heading_tokens | set(re.findall(r'[A-Z]{3,}', text_upper[:200]))

    if all_tokens & _STRATAGEM_KEYWORDS:
        return "stratagem",   prose_headers[0] if prose_headers else "Stratagem",   all_labels
    if all_tokens & _OBJECTIVE_KEYWORDS:
        return "objective",   prose_headers[0] if prose_headers else "Objective",   all_labels
    if all_tokens & _ENHANCEMENT_KEYWORDS:
        return "enhancement", prose_headers[0] if prose_headers else "Enhancement", all_labels
    if all_tokens & _ABILITY_KEYWORDS and prose_headers:
        return "ability",     prose_headers[0], all_labels
    if all_tokens & _RULES_KEYWORDS:
        return "rules",       prose_headers[0] if prose_headers else "Rules",       all_labels
    if _NARRATIVE_SIGNALS.search(text) and not stat_names:
        return "narrative",   prose_headers[0] if prose_headers else "Narrative",   all_labels

    return "general", prose_headers[0] if prose_headers else "general", all_labels


def enrich_with_unit_names(records: list) -> list:
    page_sections_map, chunk_classifications = {}, []
    for rec in records:
        section_type, identifier, all_labels = _classify_chunk(rec["text"])
        chunk_classifications.append((section_type, identifier, all_labels))
        if all_labels:
            page     = rec["page_number"]
            existing = page_sections_map.get(page, [])
            for label in all_labels:
                if label not in existing:
                    existing.append(label)
            page_sections_map[page] = existing

    enriched, last_type, last_id = [], "general", "general"
    for rec, (section_type, identifier, all_labels) in zip(records, chunk_classifications):
        page = rec["page_number"]
        if identifier and identifier != "general":
            last_type, last_id = section_type, identifier
        else:
            section_type, identifier = last_type, last_id

        rec["metadata"]["section_type"]             = section_type
        rec["metadata"]["section_identifier"]       = identifier
        rec["metadata"]["section_identifier_clean"] = (
            _to_title(identifier) if identifier != "general" else identifier)
        rec["metadata"]["all_sections_on_page"]     = page_sections_map.get(page, [])
        enriched.append(rec)

    return enriched


# ---------------------------------------------------------------------------
# Strategy dispatch
# ---------------------------------------------------------------------------

STRATEGY_HANDLERS = {
    "pdfplumber_text":            extract_pdfplumber_text,
    "pdfplumber_tables":          extract_pdfplumber_tables,
    "pdfplumber_text+ocr_images": extract_mixed,
    "ocr_full":                   extract_ocr_full,
    "deep_hybrid":                extract_deep_hybrid,
}
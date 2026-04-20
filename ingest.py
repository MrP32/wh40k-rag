"""
ingest.py — Smart PDF ingestion pipeline
Location: C:\Projects\wh40k-app\ingest.py
Command:  python ingest.py

Orchestrates the full segmentation pipeline across all PDFs and
loads the resulting chunks directly into ChromaDB.

Pipeline per PDF:
  1. assess_pdf()                    — classify PDF type & select extraction strategy
  2. segment_document_into_regions() — sub-page region detection + smart extraction
  3. Enrich metadata                 — content_category, subject, doc_type from filename
  4. Upsert into ChromaDB            — warhammer40k collection via Ollama embeddings
"""

import os
import re
import sys
import logging
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# Add pdf_agent subfolder to path
sys.path.insert(0, str(Path(__file__).parent / "pdf_agent"))
from pdf_agent import assess_pdf
from pdf_region_segmenter import segment_document_into_regions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PDF_FOLDER      = r"C:\Personal Projects\warhammer_40k_pdfs"
CHROMA_PATH     = r"C:\Projects\wh40k-app\chroma_db"
COLLECTION_NAME = "warhammer40k"

# ---------------------------------------------------------------------------
# ChromaDB — delete and recreate for a clean ingest
# ---------------------------------------------------------------------------

embedding_fn = OllamaEmbeddingFunction(
    url="http://127.0.0.1:11434/api/embeddings",
    model_name="nomic-embed-text"
)

client = chromadb.PersistentClient(path=CHROMA_PATH)

existing = [c.name for c in client.list_collections()]
if COLLECTION_NAME in existing:
    log.info(f"Deleting existing collection '{COLLECTION_NAME}' for clean re-ingest...")
    client.delete_collection(name=COLLECTION_NAME)

collection = client.create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn
)
log.info(f"Created fresh collection '{COLLECTION_NAME}'")

# ---------------------------------------------------------------------------
# Filename metadata helpers
# ---------------------------------------------------------------------------

CATEGORY_RULES = [
    (r"core rules updates",           "core_rules_updates"),
    (r"core rules",                   "core_rules"),
    (r"quick start",                  "core_rules_quickstart"),
    (r"combat patrol rules",          "combat_patrol_rules"),
    (r"crusade rules",                "crusade_rules"),
    (r"boarding actions",             "boarding_actions_rules"),
    (r"chapter approved.*tournament", "tournament_rules"),
    (r"pariah nexus.*tournament",     "tournament_rules"),
    (r"balance dataslate",            "balance_rules"),
    (r"munitorum",                    "points_costs"),
    (r"army roster",                  "army_roster"),
    (r"combat patrol",                "combat_patrol"),
    (r"faction pack",                 "faction_rules"),
    (r"imperial armour",              "imperial_armour"),
]

def extract_content_category(filename: str) -> str:
    name = filename.lower()
    for pattern, category in CATEGORY_RULES:
        if re.search(pattern, name):
            return category
    return "other"

def extract_doc_type(filename: str) -> str:
    name = filename.lower()
    if re.search(r"combat patrol rules", name): return "combat_patrol_rules"
    if re.search(r"combat patrol",       name): return "combat_patrol"
    if re.search(r"faction pack",        name): return "faction_pack"
    if re.search(r"core rules",          name): return "core_rules"
    if re.search(r"balance dataslate",   name): return "balance_dataslate"
    if re.search(r"imperial armour",     name): return "imperial_armour"
    if re.search(r"army roster",         name): return "army_roster"
    if re.search(r"munitorum",           name): return "munitorum"
    return "other"

def extract_subject(filename: str) -> str:
    name = filename.lower().replace(".pdf", "").strip()
    for prefix in ("combat patrol - ", "faction pack - ", "imperial armour - "):
        if name.startswith(prefix):
            return name[len(prefix):].strip()
    return name

# ---------------------------------------------------------------------------
# ChromaDB metadata flattening
# ---------------------------------------------------------------------------

def flatten_metadata(chunk: dict, doc_type: str, content_category: str, subject: str) -> dict:
    """
    ChromaDB requires all metadata values to be str, int, float, or bool.
    Flatten the nested metadata dict and add filename-derived fields.
    """
    raw = chunk.get("metadata", {})
    return {
        # Top-level chunk fields
        "source":                   chunk.get("source_file", ""),
        "page_number":              int(chunk.get("page_number", 0)),
        "region_index":             int(chunk.get("region_index", 0)),
        "chunk_index":              int(chunk.get("chunk_index", 0)),
        "section_type":             chunk.get("section_type", "general"),
        "section_identifier":       chunk.get("section_identifier", ""),
        "section_identifier_clean": chunk.get("section_identifier_clean", ""),
        "extraction_method":        chunk.get("extraction_method", ""),
        "content_type":             chunk.get("content_type", "text"),
        # Filename-derived
        "doc_type":                 doc_type,
        "content_category":         content_category,
        "subject":                  subject,
        # From nested metadata
        "pdf_type":                 str(raw.get("pdf_type", "")),
        "total_pages":              int(raw.get("total_pages", 0)),
        "char_count":               int(raw.get("char_count", 0)),
        "word_count":               int(raw.get("word_count", 0)),
        "has_tables":               bool(raw.get("has_tables", False)),
        "has_images":               bool(raw.get("has_images", False)),
        "ocr_confidence":           float(raw.get("ocr_confidence") or 0.0),
        "geometric_source":         str(raw.get("geometric_source", "")),
        "content_confirmed":        bool(raw.get("content_confirmed", False)),
        "is_pure_artwork":          bool(raw.get("is_pure_artwork", False)),
        "title":                    str(raw.get("title") or ""),
        "author":                   str(raw.get("author") or ""),
    }

# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

def ingest_pdfs(folder: str):
    pdf_files = sorted([f for f in os.listdir(folder) if f.endswith(".pdf")])
    log.info(f"Found {len(pdf_files)} PDFs to ingest from {folder}")

    total_chunks = 0
    failed       = []

    for i, filename in enumerate(pdf_files, 1):
        filepath         = os.path.join(folder, filename)
        doc_type         = extract_doc_type(filename)
        content_category = extract_content_category(filename)
        subject          = extract_subject(filename)

        log.info(f"[{i}/{len(pdf_files)}] {filename}")
        log.info(f"  category={content_category} | subject={subject}")

        try:
            assessment        = assess_pdf(filepath)
            chunks, statlines = segment_document_into_regions(filepath, assessment)
        except BaseException as e:
            log.warning(f"  Segmentation failed ({type(e).__name__}: {e}) — trying pdfplumber fallback")
            try:
                import pdfplumber
                chunks, statlines = [], []
                with pdfplumber.open(filepath) as pdf:
                    for page_num, page in enumerate(pdf.pages, start=1):
                        text = page.extract_text() or ""
                        if text.strip():
                            chunks.append({
                                "chunk_id":   f"{Path(filename).stem}_p{page_num:04d}_r00_c000",
                                "source_file": filepath,
                                "page_number": page_num,
                                "region_index": 0,
                                "chunk_index": 0,
                                "section_type": "general",
                                "section_identifier": "",
                                "section_identifier_clean": "",
                                "extraction_method": "pdfplumber_fallback",
                                "content_type": "text",
                                "text": text,
                                "metadata": {},
                            })
                log.info(f"  Fallback produced {len(chunks)} chunks")
            except BaseException as e2:
                log.error(f"  Fallback also failed: {e2} — skipping {filename}")
                failed.append(filename)
                continue

        if not chunks:
            log.warning(f"  No chunks produced — skipping")
            continue

        # ── Forward-pass: carry section headings into every chunk ──
        current_heading = ""
        for c in chunks:
            sid = c.get("section_identifier", "").strip()
            if sid and sid.lower() not in ("general", "narrative", ""):
                current_heading = sid
            if current_heading:
                c["_heading"] = current_heading
            else:
                c["_heading"] = ""

        ids    = [c["chunk_id"] for c in chunks]
        # Embed with: [category | subject | heading] + text
        # This ensures every chunk is self-identifying in vector space
        documents = []
        for c in chunks:
            heading = c.get("_heading", "")
            prefix  = f"[{content_category} | {subject}]"
            if heading:
                prefix += f" [{heading}]"
            documents.append(prefix + "\n" + c["text"])
        metadatas = [
            flatten_metadata(c, doc_type, content_category, subject)
            for c in chunks
        ]

        # Upsert in batches of 100
        batch_size = 100
        for start in range(0, len(chunks), batch_size):
            end = start + batch_size
            collection.upsert(
                documents=documents[start:end],
                ids=ids[start:end],
                metadatas=metadatas[start:end],
            )

        log.info(f"  -> {len(chunks)} chunks upserted | {len(statlines)} statlines")
        total_chunks += len(chunks)

    print(f"\n{'='*60}")
    print(f"  INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"  PDFs processed : {len(pdf_files) - len(failed)}")
    print(f"  Failed         : {len(failed)}")
    if failed:
        for f in failed:
            print(f"    - {f}")
    print(f"  Total chunks   : {collection.count():,}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    ingest_pdfs(PDF_FOLDER)
"""
ingest.py — Smart PDF ingestion pipeline (refactored)
Location: C:\\Projects\\wh40k-app\\ingest.py
Command:  python ingest.py

Orchestrates the full segmentation pipeline across all PDFs and loads the
resulting chunks directly into ChromaDB using the refactored module stack.

Pipeline per PDF:
  1. classify_filename()             — doc_type, subject, patrol_name, is_legends
  2. assess_pdf()                    — classify PDF type & pick extraction strategy
  3. segment_document_into_regions() — column-aware sub-page region detection
  4. apply_carry_forward()           — section_type + section_identifier with
                                       page/column resets (no bleed-through)
  5. tag_chunks_with_faction()       — Munitorum-only: per-chunk faction tag
  6. make_chunk_id()                 — content-addressable + versioned ID
  7. flatten metadata                — schema main.py's filters expect
  8. Upsert into ChromaDB            — warhammer40k collection via Ollama embeddings

Config is read from env (see .env.example). The collection is nuked and
recreated every run — same as the old pipeline. Incremental ingestion is
deferred (see the TODO in chunk_ids.py).
"""

import os
import sys
import logging
from pathlib import Path
from collections import Counter

from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

# Add pdf_agent subfolder to path so we can import the refactored modules
sys.path.insert(0, str(Path(__file__).parent / "pdf_agent"))

from pdf_agent import assess_pdf
from pdf_region_segmenter import segment_document_into_regions
from filename_classifier import classify_filename, FilenameMetadata
from heading_classifier import apply_carry_forward
from chunk_ids import get_or_create_run_id, make_chunk_id
from munitorum_parser import tag_chunks_with_faction

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — all overridable via .env
# ---------------------------------------------------------------------------

PDF_FOLDER      = os.getenv("PDF_FOLDER",      r"C:\Personal Projects\warhammer_40k_pdfs")
CHROMA_PATH     = os.getenv("CHROMA_PATH",     r"C:\Projects\wh40k-app\chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "warhammer40k")
OLLAMA_URL      = os.getenv("OLLAMA_URL",      "http://127.0.0.1:11434/api/embeddings")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "nomic-embed-text")
BATCH_SIZE      = int(os.getenv("INGEST_BATCH_SIZE", "100"))


# ---------------------------------------------------------------------------
# ChromaDB — delete and recreate for a clean ingest
# ---------------------------------------------------------------------------

def make_collection():
    embedding_fn = OllamaEmbeddingFunction(url=OLLAMA_URL, model_name=OLLAMA_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        log.info(f"Deleting existing collection '{COLLECTION_NAME}' for clean re-ingest...")
        client.delete_collection(name=COLLECTION_NAME)
    collection = client.create_collection(
        name=COLLECTION_NAME, embedding_function=embedding_fn,
    )
    log.info(f"Created fresh collection '{COLLECTION_NAME}'")
    return collection


# ---------------------------------------------------------------------------
# Record builders — kept in ingest.py (not a module) because they're the
# contract between the pipeline and ChromaDB's schema requirements.
#
# ChromaDB requires metadata values to be str / int / float / bool. Any
# None value in the nested metadata must be coerced. We do this explicitly
# field-by-field rather than with a generic coercer so the schema is
# self-documenting and grep-able.
# ---------------------------------------------------------------------------

def build_embedding_text(chunk: dict) -> str:
    """
    What actually goes to the embedding model.

    When the heading classifier is confident about the chunk's heading, we
    prepend it so the embedding captures the section context. When it isn't,
    we send the raw text — prefixing with a wrong heading poisons the vector
    more than missing context hurts it.

    Note: we intentionally do NOT prepend doc_type / subject. Those live in
    metadata and are applied as filters at query time. Prepending them to
    the embedding text was the old pipeline's approach and it added noise
    to the vector without improving filter precision (filters are exact).
    """
    text = chunk.get("text", "")
    classification = chunk.get("classification")
    if classification and classification.confident and classification.heading:
        return f"[{classification.heading}]\n{text}"
    return text


def flatten_chunk_metadata(chunk: dict, fm: FilenameMetadata) -> dict:
    """
    Produce a ChromaDB-safe flat metadata dict.

    The keys here MUST match the filter keys main.py's extract_filters()
    emits: subject, doc_type, patrol_name, munitorum_faction. Any rename
    breaks retrieval silently. Diagnostic fields (content_type, word_count,
    etc.) are included for /db-info and debugging but aren't queried.
    """
    inner = chunk.get("metadata", {}) or {}
    classification = chunk.get("classification")

    return {
        # --- Filter-critical: main.py queries these ---
        "doc_type":                 fm.doc_type,
        "subject":                  fm.subject,
        "patrol_name":              fm.patrol_name,
        "munitorum_faction":        inner.get("munitorum_faction", ""),

        # --- Filename-level ---
        "is_legends":               bool(fm.is_legends),

        # --- Chunk identity / provenance ---
        "source":                   str(chunk.get("source_file", "")),
        "page_number":              int(chunk.get("page_number", 0)),
        "region_index":             int(chunk.get("region_index", 0)),
        "chunk_index":              int(chunk.get("chunk_index", 0)),
        "column_label":             str(chunk.get("column_label", "single")),

        # --- Classification (from heading_classifier + carry-forward) ---
        "section_type":             str(chunk.get("section_type", "general")),
        "section_identifier":       str(chunk.get("section_identifier", "")),
        "classification_confident": bool(
            classification.confident if classification else False
        ),

        # --- Extraction diagnostics ---
        "content_type":             str(chunk.get("content_type", "text")),
        "extraction_method":        str(chunk.get("extraction_method", "")),
        "geometric_source":         str(chunk.get("geometric_source", "")),
        "content_confirmed":        bool(chunk.get("content_confirmed", False)),
        "is_pure_artwork":          bool(chunk.get("is_pure_artwork", False)),
        "word_count":               int(chunk.get("word_count", 0)),
        "table_count":              int(chunk.get("table_count", 0)),
        "ocr_confidence":           float(chunk.get("ocr_confidence") or 0.0),

        # --- PDF-level (from the segmenter's nested metadata) ---
        "pdf_type":                 str(inner.get("pdf_type", "")),
        "total_pages":              int(inner.get("total_pages", 0)),
        "char_count":               int(inner.get("char_count", 0)),
        "has_tables":               bool(inner.get("has_tables", False)),
        "has_images":               bool(inner.get("has_images", False)),
        "title":                    str(inner.get("title") or ""),
        "author":                   str(inner.get("author") or ""),
    }


def build_records(chunks: list, fm: FilenameMetadata) -> tuple:
    """
    Turn chunks into (ids, documents, metadatas) triples ready for upsert.
    Also de-duplicates chunk IDs defensively: if two chunks produce the same
    ID, we keep the first and warn. This shouldn't happen given the whitespace
    normalization in make_chunk_id, but a second collision would silently
    overwrite a chunk otherwise.
    """
    ids, documents, metadatas = [], [], []
    seen = set()
    dupes = 0
    for c in chunks:
        cid = make_chunk_id(
            source_file=c.get("source_file", ""),
            page_number=int(c.get("page_number", 0)),
            text=c.get("text", ""),
        )
        if cid in seen:
            dupes += 1
            continue
        seen.add(cid)
        ids.append(cid)
        documents.append(build_embedding_text(c))
        metadatas.append(flatten_chunk_metadata(c, fm))
    if dupes:
        log.warning(f"  [DEDUP-ID] dropped {dupes} chunk(s) with duplicate IDs")
    return ids, documents, metadatas


# ---------------------------------------------------------------------------
# pdfplumber fallback — used when segment_document_into_regions crashes
# ---------------------------------------------------------------------------

def pdfplumber_fallback(filepath: str) -> list:
    """
    Dumbest-possible text extraction so a segmenter crash doesn't lose a PDF
    entirely. Produces chunk dicts compatible with the normal pipeline schema
    (minus column_label — set to "single" — and with a trivial classification).

    Returns an empty list on complete failure; the caller will then skip the PDF.
    """
    try:
        import pdfplumber
    except Exception as e:
        log.error(f"  pdfplumber unavailable for fallback: {e}")
        return []

    try:
        from heading_classifier import classify_chunk
    except Exception:
        classify_chunk = None

    chunks = []
    try:
        with pdfplumber.open(filepath) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                classification = classify_chunk(text) if classify_chunk else None
                chunks.append({
                    "source_file":        filepath,
                    "page_number":        page_num,
                    "region_index":       0,
                    "chunk_index":        0,
                    "column_label":       "single",
                    "text":               text,
                    "content_type":       "text",
                    "extraction_method":  "pdfplumber_fallback",
                    "bbox":               [0, 0, 0, 0],
                    "geometric_source":   "fallback",
                    "content_confirmed":  False,
                    "ocr_confidence":     0.0,
                    "table_count":        0,
                    "word_count":         len(text.split()),
                    "is_pure_artwork":    False,
                    "statlines":          [],
                    "classification":     classification,
                    "section_type":       "general",
                    "section_identifier": "",
                    "metadata":           {},
                })
    except Exception as e:
        log.error(f"  pdfplumber fallback failed: {type(e).__name__}: {e}")
        return []
    return chunks


# ---------------------------------------------------------------------------
# Main ingest
# ---------------------------------------------------------------------------

def ingest_pdfs(folder: str) -> None:
    if not os.path.isdir(folder):
        log.error(f"PDF folder does not exist: {folder}")
        sys.exit(1)

    pdf_files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".pdf"))
    if not pdf_files:
        log.error(f"No PDFs found in {folder}")
        sys.exit(1)

    run_id = get_or_create_run_id()
    log.info(f"Ingest run_id = {run_id}")
    log.info(f"Found {len(pdf_files)} PDFs to ingest from {folder}")

    collection = make_collection()

    total_chunks = 0
    doc_type_counts = Counter()
    subject_counts = Counter()
    failed = []

    for i, filename in enumerate(pdf_files, 1):
        filepath = os.path.join(folder, filename)
        fm = classify_filename(filename)
        log.info(
            f"[{i}/{len(pdf_files)}] {filename} "
            f"| doc_type={fm.doc_type} subject={fm.subject!r} "
            f"patrol={fm.patrol_name!r}"
        )

        # ---- Segment (with fallback) ----
        try:
            assessment = assess_pdf(filepath)
            chunks, statlines = segment_document_into_regions(filepath, assessment)
            if not chunks:
                log.warning("  segment produced 0 chunks — trying pdfplumber fallback")
                chunks = pdfplumber_fallback(filepath)
                statlines = []
        except BaseException as e:
            log.warning(
                f"  segmentation failed ({type(e).__name__}: {e}) "
                f"— trying pdfplumber fallback"
            )
            chunks = pdfplumber_fallback(filepath)
            statlines = []

        if not chunks:
            log.error(f"  no chunks produced — skipping {filename}")
            failed.append(filename)
            continue

        # ---- Carry-forward: page/column-aware heading propagation ----
        # Mutates chunks in place; produces final section_type / section_identifier.
        apply_carry_forward(chunks)

        # ---- Munitorum-only: tag every chunk with its faction ----
        # Intentionally scoped to points_costs so a bug here can't corrupt
        # every other PDF's metadata. The tagger is also strictly additive:
        # it only writes `munitorum_faction` into chunk["metadata"], so
        # other keys are unaffected.
        if fm.doc_type == "points_costs":
            try:
                tag_chunks_with_faction(chunks)
            except Exception as e:
                log.warning(
                    f"  [MUNITORUM] tagging failed ({type(e).__name__}: {e}) "
                    f"— continuing with empty munitorum_faction"
                )

        # ---- Build records ----
        ids, documents, metadatas = build_records(chunks, fm)
        if not ids:
            log.warning(f"  all chunks collapsed by dedup — skipping {filename}")
            failed.append(filename)
            continue

        # ---- Upsert in batches ----
        try:
            for start in range(0, len(ids), BATCH_SIZE):
                end = start + BATCH_SIZE
                collection.upsert(
                    ids=ids[start:end],
                    documents=documents[start:end],
                    metadatas=metadatas[start:end],
                )
        except Exception as e:
            log.error(f"  upsert failed ({type(e).__name__}: {e}) — skipping {filename}")
            failed.append(filename)
            continue

        total_chunks += len(ids)
        doc_type_counts[fm.doc_type] += 1
        subject_counts[fm.subject] += 1
        log.info(f"  -> {len(ids)} chunks upserted | {len(statlines)} statlines")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  INGESTION COMPLETE  (run_id={run_id})")
    print(f"{'='*60}")
    print(f"  PDFs processed : {len(pdf_files) - len(failed)}")
    print(f"  Failed         : {len(failed)}")
    for f in failed:
        print(f"    - {f}")
    print(f"  Total chunks   : {collection.count():,}")
    print(f"  doc_type mix   : {dict(doc_type_counts)}")
    top_subjects = subject_counts.most_common(10)
    print(f"  top subjects   : {top_subjects}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    ingest_pdfs(PDF_FOLDER)

"""
chunk_ids.py
============
Stable content-addressable chunk IDs with per-run version suffix.

Location: C:\\Projects\\wh40k-app\\pdf_agent\\chunk_ids.py

Replaces the positional ID scheme `{stem}_p{page}_r{region}_c{chunk}` which
was not stable across re-ingests because region indices depend on floating-point
geometric detection.

Format: sha1(source + page + text_prefix)[:16] + "-v" + ingest_run_id

The text_prefix (first 100 chars) provides content-addressability, so the same
chunk produces the same ID across runs. The version suffix distinguishes runs,
so a user can track which ingest created a chunk.

TODO(incremental-ingest):
    Once extraction is stable across runs, we can drop the version suffix so
    upsert() actually updates in place. For now the version suffix intentionally
    changes the ID on every run to match the existing nuke-and-rebuild workflow.
    To enable incremental ingestion:
      1. Remove the version suffix from make_chunk_id()
      2. Add a per-PDF checksum cache (see .ingest_cache.json)
      3. Skip PDFs whose checksum matches the cache
      4. On per-PDF ingestion, delete old chunks for that source before upserting
"""

import hashlib
import time
import uuid

# Populated once at ingest startup by get_or_create_run_id()
_CURRENT_RUN_ID = None


def get_or_create_run_id() -> str:
    """
    Return the ingest run ID for this process. Generated once per run.
    Format: YYYYMMDDHHMM-<4char-random> — sortable and unique.
    """
    global _CURRENT_RUN_ID
    if _CURRENT_RUN_ID is None:
        timestamp = time.strftime("%Y%m%d%H%M")
        short = uuid.uuid4().hex[:4]
        _CURRENT_RUN_ID = f"{timestamp}-{short}"
    return _CURRENT_RUN_ID


def set_run_id(run_id: str) -> None:
    """Explicitly set the run ID (useful for tests)."""
    global _CURRENT_RUN_ID
    _CURRENT_RUN_ID = run_id


def make_chunk_id(source_file: str, page_number: int, text: str) -> str:
    """
    Return a chunk ID of the form <16-hex-chars>-v<run-id>.

    Content-addressable: same (source, page, text_prefix) → same hash.
    The run_id suffix distinguishes which ingest run produced this chunk,
    which is important while we're nuking-and-rebuilding the collection on
    each run.

    Signature normalization: WH40K datasheets extracted with layout=True
    often produce chunks that start with 50-200 characters of whitespace
    (from the layout preserving empty box regions). Using the raw first-100
    chars caused every such chunk on a page to hash identically. We strip
    leading whitespace and collapse runs of internal whitespace before
    taking the 100-char signature, so two chunks with different content but
    identical whitespace prefixes produce different IDs.
    """
    raw = text or ""
    # Collapse any run of whitespace (spaces, tabs, newlines) to a single
    # space, then strip. This gives us the first 100 chars of actual content.
    normalized = " ".join(raw.split())
    text_prefix = normalized[:100]
    digest_input = f"{source_file}|{page_number}|{text_prefix}".encode("utf-8")
    digest = hashlib.sha1(digest_input).hexdigest()[:16]
    return f"{digest}-v{get_or_create_run_id()}"

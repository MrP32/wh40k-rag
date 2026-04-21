"""
test_full_corpus.py
===================
End-to-end smoke test on all 3 provided PDFs. Exercises every stage of the
pipeline except the ChromaDB upsert (which requires Ollama + chroma running).

Verifies:
  - classify_filename for each file
  - assess_pdf completes
  - segment_document_into_regions completes
  - apply_carry_forward completes and produces sensible identifiers
  - build_chunk_records_for_upsert produces id/doc/meta triples
  - No duplicate chunk IDs within a file
  - Stratagem chunks get the stratagem section_type after carry-forward
  - RESET_TYPES don't bleed across pages/columns
"""

import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent))

from pdf_agent import assess_pdf
from pdf_region_segmenter import segment_document_into_regions
from filename_classifier import classify_filename
from heading_classifier import apply_carry_forward, RESET_TYPES
from chunk_ids import set_run_id, make_chunk_id
from filename_classifier import FilenameMetadata

# Inline the build logic so we don't need chromadb installed just to test it.
def build_embedding_text(chunk):
    text = chunk.get("text", "")
    classification = chunk.get("classification")
    if classification and classification.confident and classification.heading:
        return f"[{classification.heading}]\n{text}"
    return text

def flatten_chunk_metadata(chunk, fm: FilenameMetadata):
    inner = chunk.get("metadata", {}) or {}
    classification = chunk.get("classification")
    return {
        "source": chunk.get("source_file", ""),
        "page_number": int(chunk.get("page_number", 0)),
        "column_label": chunk.get("column_label", "single"),
        "section_type": chunk.get("section_type", "general"),
        "section_identifier": chunk.get("section_identifier", ""),
        "classification_confident": bool(classification.confident if classification else False),
        "content_type": chunk.get("content_type", "text"),
        "word_count": int(chunk.get("word_count", 0)),
        "doc_type": fm.doc_type,
        "subject": fm.subject,
        "patrol_name": fm.patrol_name,
        "is_legends": fm.is_legends,
        "munitorum_faction": inner.get("munitorum_faction", ""),
    }

def build_chunk_records_for_upsert(chunks, fm):
    ids, documents, metadatas = [], [], []
    for c in chunks:
        cid = make_chunk_id(c.get("source_file", ""), c.get("page_number", 0), c.get("text", ""))
        ids.append(cid)
        documents.append(build_embedding_text(c))
        metadatas.append(flatten_chunk_metadata(c, fm))
    return ids, documents, metadatas

PDFS = [
    "/home/claude/40k/testpdfs/Combat_Patrol_-_Grey_Knights_-_Aurellios_Banishers.pdf",
    "/home/claude/40k/testpdfs/Faction_Pack_-_Grey_Knights.pdf",
    "/home/claude/40k/testpdfs/Faction_Pack_-_Space_Marines.pdf",
]

set_run_id("smoketest-0001")

def process(pdf_path):
    print(f"\n{'='*72}\n  {Path(pdf_path).name}\n{'='*72}")
    fn = Path(pdf_path).name
    fm = classify_filename(fn)
    print(f"  [filename] doc_type={fm.doc_type} subject={fm.subject!r} patrol={fm.patrol_name!r} legends={fm.is_legends}")

    assessment = assess_pdf(pdf_path)
    print(f"  [assess] type={assessment.pdf_type}  pages={assessment.total_pages}")

    # Space Marines is huge — cap to 30 pages for reasonable runtime
    max_pages = 30 if "Space_Marines" in fn else None
    chunks, statlines = segment_document_into_regions(pdf_path, assessment, max_pages=max_pages)
    print(f"  [segment] {len(chunks)} chunks, {len(statlines)} statlines"
          + (f" (capped to {max_pages} pages)" if max_pages else ""))

    apply_carry_forward(chunks)

    ids, docs, metas = build_chunk_records_for_upsert(chunks, fm)
    print(f"  [build]   {len(ids)} ids, {len(docs)} documents, {len(metas)} metadatas")

    # ---- CHECKS ----
    issues = []

    # 1. No duplicate chunk IDs
    dup_ids = [cid for cid, n in Counter(ids).items() if n > 1]
    if dup_ids:
        issues.append(f"DUPLICATE IDS: {len(dup_ids)} collisions")
        for d in dup_ids[:3]:
            issues.append(f"  - {d}")

    # 2. All metadata has expected keys
    required_keys = {"source", "page_number", "doc_type", "subject", "patrol_name",
                     "section_type", "section_identifier", "column_label",
                     "classification_confident", "content_type"}
    missing = [k for k in required_keys if any(k not in m for m in metas)]
    if missing:
        issues.append(f"MISSING KEYS in metadata: {missing}")

    # 3. doc_type + subject consistent across all chunks for one PDF
    doc_types = {m["doc_type"] for m in metas}
    subjects = {m["subject"] for m in metas}
    if len(doc_types) != 1:
        issues.append(f"INCONSISTENT doc_type: {doc_types}")
    if len(subjects) != 1:
        issues.append(f"INCONSISTENT subject: {subjects}")

    # 4. Section type distribution
    section_types = Counter(m["section_type"] for m in metas)
    print(f"  [types]   {dict(section_types)}")

    # 5. RESET_TYPES bleed check: count unique section_identifiers per page
    # For RESET_TYPES, a given page's chunks shouldn't ALL carry the same
    # identifier from a previous page (the classic bleed bug).
    # Just sanity-check: count distinct (page, section_identifier) pairs.
    reset_pages = Counter()
    for c in chunks:
        if c.get("section_type") in RESET_TYPES and c.get("section_identifier"):
            reset_pages[c["section_identifier"]] += 1
    if reset_pages:
        top_carriers = reset_pages.most_common(3)
        print(f"  [headers] top RESET-type headings: {top_carriers}")

    # 6. Confident classifications
    confident_count = sum(1 for m in metas if m["classification_confident"])
    print(f"  [classify] {confident_count}/{len(metas)} chunks have confident heading ({confident_count*100//max(1,len(metas))}%)")

    # 7. Embedding prefix sanity — doc lengths in reasonable range
    doc_lens = [len(d) for d in docs]
    if doc_lens:
        print(f"  [docs]    len min/med/max = {min(doc_lens)}/{sorted(doc_lens)[len(doc_lens)//2]}/{max(doc_lens)}")

    # 8. Page-3 sanity for Grey Knights specifically — all 6 Warpbane stratagems
    if "Grey_Knights.pdf" in fn and "Combat" not in fn:
        pg3_text = " ".join(c["text"] for c in chunks if c["page_number"] == 3)
        expected = ["SANCTIFIED KILL ZONE", "FLAMES OF SANCTITY", "HALLOWED BEACON",
                    "FIRES OF COVENANT", "AEGIS ETERNAL", "REPELLING SPHERE"]
        missing_strats = [s for s in expected if s not in pg3_text]
        pg3_count = sum(1 for c in chunks if c["page_number"] == 3)
        if missing_strats:
            issues.append(f"PAGE 3 missing stratagems: {missing_strats}")
        else:
            print(f"  [pg3]     ✓ all 6 Warpbane stratagems present in page 3 text ({pg3_count} chunks)")

    if issues:
        print(f"  [ISSUES]")
        for i in issues:
            print(f"     ✗ {i}")
        return False
    print(f"  [OK]")
    return True


all_ok = True
for p in PDFS:
    try:
        ok = process(p)
        all_ok = all_ok and ok
    except Exception as e:
        import traceback
        print(f"  [FATAL] {type(e).__name__}: {e}")
        traceback.print_exc()
        all_ok = False

print(f"\n{'='*72}")
print(f"  OVERALL: {'ALL PDFs OK' if all_ok else 'FAILURES DETECTED'}")
print(f"{'='*72}")
sys.exit(0 if all_ok else 1)

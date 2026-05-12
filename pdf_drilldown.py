"""
pdf_drilldown.py
================
Deep inspection of a single PDF's chunks in the warhammer40k collection.

Location: C:\\Projects\\wh40k-app\\pdf_drilldown.py
Run:       python pdf_drilldown.py "<filename-substring>"

Usage examples:
    python pdf_drilldown.py "Munitorum"
    python pdf_drilldown.py "Grey Knights"
    python pdf_drilldown.py "Faction Pack - Tyranids"
    python pdf_drilldown.py "Core Rules.pdf"

If the substring matches multiple PDFs, the tool lists them and exits; pick
a more specific substring and re-run.

What it reports (no writes, no re-ingest):
  - Total chunks
  - Per-page chunk counts
  - doc_type / subject metadata values (should be consistent across one PDF)
  - patrol_name / munitorum_faction distribution
  - section_type distribution
  - Top 20 section_identifier values
  - classification_confident rate
  - A sample chunk per page (first one, truncated) for eyeballing extraction
  - Whitespace-dominated chunks (retrieval noise)
  - Duplicate-text chunks (extraction artifacts)
"""

import os
import sys
import re
from collections import Counter, defaultdict

from dotenv import load_dotenv
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction

load_dotenv()

CHROMA_PATH     = os.getenv("CHROMA_PATH",     r"C:\Projects\wh40k-app\chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "warhammer40k")
OLLAMA_URL      = os.getenv("OLLAMA_URL",      "http://127.0.0.1:11434/api/embeddings")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL",    "nomic-embed-text")

embedding_fn = OllamaEmbeddingFunction(url=OLLAMA_URL, model_name=OLLAMA_MODEL)
client       = chromadb.PersistentClient(path=CHROMA_PATH)
collection   = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)


def hr(title):
    print(f"\n-- {title} " + "-" * max(0, 65 - len(title)))


def _shorten(source):
    return (source or "").replace("\\", "/").split("/")[-1]


def _effective_word_count(text):
    stripped = re.sub(r"[|\-=\s]+", " ", text or "").strip()
    return len(stripped.split()) if stripped else 0


def _resolve_pdf(substring):
    """
    Find PDFs in the collection whose source filename contains the substring.
    Returns list of unique source filenames.
    """
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    sources = sorted({m.get("source", "") for m in all_meta if m.get("source")})
    substring_lower = substring.lower()
    matches = [s for s in sources if substring_lower in _shorten(s).lower()]
    return matches, sources


def drilldown(source):
    """Print everything the collection knows about chunks from this source."""
    data = collection.get(where={"source": {"$eq": source}},
                          include=["documents", "metadatas"])
    docs  = data["documents"]
    metas = data["metadatas"]
    ids   = data["ids"]

    fname = _shorten(source)
    print(f"\n{'=' * 70}")
    print(f"  PDF drilldown: {fname}")
    print(f"  Full path: {source}")
    print(f"  Total chunks: {len(ids):,}")
    print(f"{'=' * 70}")

    if not metas:
        print("\n  (no chunks found for this source)")
        return

    # ---- 1. Filename-derived metadata ----
    hr("1. Filename-derived metadata (expect: consistent across all chunks)")
    doc_types  = Counter(m.get("doc_type", "<missing>") for m in metas)
    subjects   = Counter(m.get("subject", "<missing>") for m in metas)
    patrols    = Counter((m.get("patrol_name") or "<none>") for m in metas)
    for label, counter in [("doc_type", doc_types),
                           ("subject", subjects),
                           ("patrol_name", patrols)]:
        if len(counter) == 1:
            val, count = next(iter(counter.items()))
            print(f"  {label:<15}  {val!r}   ({count} chunks)")
        else:
            print(f"  {label:<15}  INCONSISTENT:")
            for val, count in counter.most_common():
                print(f"                   {val!r}  {count} chunks")

    # ---- 2. Per-page chunk counts ----
    hr("2. Per-page chunk counts")
    page_counts = Counter(int(m.get("page_number", 0) or 0) for m in metas)
    total_pages = int(metas[0].get("total_pages") or 0)
    if total_pages:
        print(f"  PDF has {total_pages} pages. Chunk distribution:")
        missing = [p for p in range(1, total_pages + 1) if p not in page_counts]
        if missing:
            print(f"  Pages with NO chunks: {missing[:20]}" +
                  (f" ... (+{len(missing) - 20} more)" if len(missing) > 20 else ""))
    for page, count in sorted(page_counts.items()):
        bar = "#" * min(count, 40)
        print(f"    p{page:>3}  {count:>3}  {bar}")

    # ---- 3. Munitorum faction distribution (only relevant for points_costs) ----
    if any(m.get("doc_type") == "points_costs" for m in metas):
        hr("3. Munitorum faction distribution (munitorum_faction)")
        muni = Counter((m.get("munitorum_faction") or "<empty>") for m in metas)
        for fac, count in muni.most_common():
            bar = "#" * min(count, 40)
            print(f"    {fac:<25}  {count:>4}  {bar}")
        # Per-page faction tagging: is the tag stable across pages?
        by_page = defaultdict(Counter)
        for m in metas:
            page = int(m.get("page_number", 0) or 0)
            by_page[page][m.get("munitorum_faction") or "<empty>"] += 1
        print(f"\n  Per-page dominant faction (detects mid-page faction flips):")
        for page in sorted(by_page.keys()):
            dominant = by_page[page].most_common(1)[0]
            variety = len(by_page[page])
            if variety == 1:
                print(f"    p{page:>3}  {dominant[0]} ({dominant[1]})")
            else:
                distribution = ", ".join(f"{f}={c}" for f, c in by_page[page].most_common())
                print(f"    p{page:>3}  MIXED: {distribution}")

    # ---- 4. Section classification ----
    hr("4. Section types")
    sec_types = Counter(m.get("section_type", "general") for m in metas)
    for st, count in sec_types.most_common():
        bar = "#" * min(count, 30)
        print(f"    {st:<20}  {count:>4}  {bar}")

    hr("5. Top section_identifier values")
    identifiers = Counter((m.get("section_identifier") or "<empty>") for m in metas)
    for ident, count in identifiers.most_common(20):
        label = ident if ident != "<empty>" else "<empty>"
        print(f"    {count:>4}  {label!r}")
    if len(identifiers) > 20:
        print(f"    ... ({len(identifiers) - 20} more distinct identifiers)")

    # ---- 6. Classification confidence ----
    hr("6. Heading classifier confidence")
    conf = Counter(bool(m.get("classification_confident")) for m in metas)
    total = sum(conf.values()) or 1
    print(f"    confident=True   {conf[True]:>4}  ({conf[True] * 100 // total}%)")
    print(f"    confident=False  {conf[False]:>4}  ({conf[False] * 100 // total}%)")

    # ---- 7. Whitespace-dominated chunks (extraction noise) ----
    hr("7. Low-content chunks (retrieval noise)")
    low_content = []
    for i, doc in enumerate(docs):
        ewc = _effective_word_count(doc)
        if ewc < 20:
            low_content.append((ewc, i, doc, metas[i]))
    low_content.sort(key=lambda x: x[0])
    print(f"  Chunks with <20 effective words: {len(low_content)} / {len(docs)} "
          f"({len(low_content) * 100 // max(1, len(docs))}%)")
    print(f"  Showing the 5 lowest-content examples:")
    for ewc, i, doc, m in low_content[:5]:
        preview = (doc or "")[:120].replace("\n", " ")
        print(f"    page={m.get('page_number')} region={m.get('region_index')} "
              f"ewc={ewc}  preview={preview!r}")

    # ---- 8. Duplicate-text detection ----
    hr("8. Near-duplicate chunks (first 80 chars collision)")
    sig_groups = defaultdict(list)
    for i, doc in enumerate(docs):
        sig = re.sub(r"\s+", " ", (doc or "").strip())[:80]
        if sig:
            sig_groups[sig].append(i)
    dupes = {sig: idxs for sig, idxs in sig_groups.items() if len(idxs) > 1}
    print(f"  Distinct prefixes: {len(sig_groups)}")
    print(f"  Prefixes appearing >1 time: {len(dupes)}")
    if dupes:
        top_dupes = sorted(dupes.items(), key=lambda x: -len(x[1]))[:5]
        print(f"  Top 5 most-duplicated prefixes:")
        for sig, idxs in top_dupes:
            pages = sorted({int(metas[i].get("page_number", 0) or 0) for i in idxs})
            print(f"    x{len(idxs):<2}  pages={pages}  preview={sig[:80]!r}")

    # ---- 9. Sample first chunk from each page ----
    hr("9. First chunk per page (first 150 chars)")
    page_to_first_chunk_idx = {}
    for i, m in enumerate(metas):
        page = int(m.get("page_number", 0) or 0)
        region = int(m.get("region_index", 0) or 0)
        # Prefer region 0 first chunk
        key = (page, region, int(m.get("chunk_index", 0) or 0))
        if page not in page_to_first_chunk_idx:
            page_to_first_chunk_idx[page] = (key, i)
        elif key < page_to_first_chunk_idx[page][0]:
            page_to_first_chunk_idx[page] = (key, i)
    for page in sorted(page_to_first_chunk_idx.keys()):
        _, idx = page_to_first_chunk_idx[page]
        doc = docs[idx] or ""
        meta = metas[idx]
        preview = doc[:150].replace("\n", " ")
        heading = meta.get("section_identifier") or ""
        conf = "T" if meta.get("classification_confident") else "F"
        muni = meta.get("munitorum_faction") or ""
        extras = []
        if heading:
            extras.append(f"heading={heading!r}")
        if muni:
            extras.append(f"muni={muni}")
        extras.append(f"conf={conf}")
        extra_str = " ".join(extras)
        print(f"  p{page:>3}  [{extra_str}]")
        print(f"        {preview!r}")

    print(f"\n{'=' * 70}")
    print(f"  End of drilldown for {fname}")
    print(f"{'=' * 70}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python pdf_drilldown.py <filename-substring>")
        print('Example: python pdf_drilldown.py "Munitorum"')
        print('         python pdf_drilldown.py "Grey Knights"')
        sys.exit(1)

    substring = sys.argv[1]
    matches, all_sources = _resolve_pdf(substring)

    if not matches:
        print(f"\n  No PDFs match substring {substring!r}")
        print(f"  Collection has {len(all_sources)} distinct sources. First 20:")
        for s in all_sources[:20]:
            print(f"    {_shorten(s)}")
        if len(all_sources) > 20:
            print(f"    ... ({len(all_sources) - 20} more)")
        sys.exit(2)

    if len(matches) > 1:
        print(f"\n  Substring {substring!r} matches {len(matches)} PDFs:")
        for s in matches:
            print(f"    {_shorten(s)}")
        print("\n  Please use a more specific substring.")
        sys.exit(2)

    drilldown(matches[0])


if __name__ == "__main__":
    main()

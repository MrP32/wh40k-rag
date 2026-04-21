# WH40K RAG Ingestion Refactor — Handoff v2

Continuation of the original handoff. Items that were TODO in v1 and their
current status, plus issues discovered and fixed while verifying v1's work.

## TL;DR

- TODO #1 (dedup re-verify): **done, with a real fix.** v1's dedup rule was
  geometrically incomplete and let duplicates through on real data. Rewrote
  to use 2D area overlap. Verified on Grey Knights page 3.
- TODO #2 (main.py filter prompt): **done.** Refactored `main.py` in place
  to match the new schema (doc_type, patrol_name, munitorum_faction) and
  added three-tier fallback.
- TODO #3 (.env.example): **done.**
- TODO #4 (deploy.ps1 check): **not done** — didn't have the file.
- TODO #5 (full re-ingest): **not done** — requires live chromadb + Ollama.
- TODO #6 (test_retrieval.py baseline comparison): **not done** — same reason.

## Regressions discovered while running v1 against real PDFs

Three bugs that would have broken the first real re-ingest. All fixed here.

### Bug A — filename_classifier didn't match underscore-form filenames
**Symptom:** every one of the three test PDFs (`Combat_Patrol_-_Grey_Knights_-_Aurellios_Banishers.pdf`, `Faction_Pack_-_Grey_Knights.pdf`, `Faction_Pack_-_Space_Marines.pdf`) fell through to the `other` fallback, losing all doc_type and subject metadata.

**Root cause:** v1's "18/18 tested" cases used space-form filenames. Web-downloaded PDFs arrive with underscores.

**Fix:** normalize `_` → space before applying regex rules. See `filename_classifier.py`. Added 5 underscore-form test cases to `test_pipeline.py`.

### Bug B — whitespace-prefix collision in make_chunk_id
**Symptom:** two different chunks on Combat Patrol page 6 produced the same chunk ID, causing an upsert collision that would silently lose one chunk on each ingest.

**Root cause:** WH40K datasheets extracted with `layout=True` often start with 50–200+ chars of pure whitespace (preserving the layout of empty ability-box regions). v1's hash input was `text[:100].strip()`, so two chunks whose first 100 chars were both pure whitespace — regardless of what followed — hashed identically.

**Fix:** normalize whitespace before slicing. `" ".join(text.split())[:100]` gives the first 100 chars of actual content. See `chunk_ids.py`. Added 2 test cases.

### Bug C — whitespace-only chunks were emitted
**Symptom:** pages with large empty regions produced chunk dicts whose `text` was pure whitespace. Useless entries in the collection, no retrieval value, and contributed to Bug B.

**Fix:** filter `if not chunk_str.strip(): continue` in `PageRegion.to_chunk_dicts()`. One-liner in `pdf_region_segmenter.py`.

## Dedup pass — the real fix (TODO #1)

### Why v1's dedup was insufficient

v1's rule: two regions are duplicates if y overlap > 80% AND x ranges match within 2 units.

Real Grey Knights page 3 geometry shows why this fails. pdfplumber reports **4 overlapping image bboxes on the left column**:
1. `x=[0, 349]   y=[0, 793]` — full-page background
2. `x=[21, 111]  y=[21, 772]` — narrow sidebar ornament
3. `x=[45, 349]  y=[21, 772]` — wide content-area rectangle
4. `x=[29, 102]  y=[641, 713]` — small decoration

Regions 1, 2, and 3 all have near-identical y ranges but different x ranges, so v1's rule kept them all. The segmenter then extracts each via `page.crop(bbox)`; because the three crops all span the same y-slice of body text, they yield near-identical prose. Result on page 3: 14 chunks, 4 duplicate-content signatures including two copies of "FIRES OF COVENANT" on the right column.

### The new rule

Two regions are duplicates if their **2D area overlap is ≥80% of the smaller region's area**. Ordering: earliest y0 first, then widest area first — so the "biggest container" is kept and tighter overlappers are discarded.

See `_deduplicate_regions()` in `pdf_region_segmenter.py` (around line 306). New helpers `_area()` and `_rect_overlap_area()` live just above.

### Verification

`test_dedup_real.py` runs the full pipeline against Grey Knights page 3:
- **Before:** 14 chunks on page 3, 4 duplicate-content signatures, all 6 Warpbane stratagems present.
- **After:** 5 chunks on page 3, 0 duplicate-content signatures, all 6 Warpbane stratagems still present.

First 3 pages total went from 48 chunks → 20 chunks. All 86 unit tests still pass (tests for fully-contained regions, non-overlapping regions, etc. still pass because those cases imply ≥80% 2D overlap too).

## main.py refactor (TODO #2)

Full refactor of `main.py` in place. Changes:

1. **Config from env.** Loads `CHROMA_PATH`, `COLLECTION_NAME`, `OLLAMA_URL`, `OLLAMA_MODEL` via `os.getenv()` with sensible defaults. Matches `.env.example`.

2. **New FILTER_PROMPT.** Completely rewritten to match the new metadata schema:
   - `doc_type` replaces `content_category` (the old field name no longer exists)
   - New `patrol_name` field for combat patrol queries
   - New `munitorum_faction` field for points-cost queries
   - Subject list pruned to codex-faction level only (no sub-chapters like `ultramarines`, which was a v1 mistake — those aren't separate subjects in the ingestion output)
   - Worked example for "Nemesis Dreadknight points" → `munitorum_faction: grey knights`
   - Worked example for "Aurellios Banishers combat patrol" → separate `subject` and `patrol_name`

3. **Three-tier fallback** in `search_context()`:
   - Tier 1: exact filter (best precision)
   - Tier 2: subject-only (if tier 1 returns nothing and the filter had a subject)
   - Tier 3: unfiltered semantic search (final safety net)
   - Handles the "Grey Knights combat patrol returned 0" case from the original handoff.

4. **Hardened empty-result handling.** `_chroma_query()` catches exceptions and returns `([], [])`. The `/db-info` endpoint uses `m.get("source", "unknown")` instead of `m["source"]` so one malformed chunk can't 500 the whole endpoint.

5. **max_tokens bumped from 100 to 200** for the filter extraction call — the new 3-field filter output plus $and wrapper can exceed 100 tokens.

Diff of `main.py` vs. `main.py.original` is included in the final zip.

## Test suites

Three standalone test suites, all passing:

- `test_pipeline.py` — 86 unit tests covering pure-logic modules (filename classifier, text chunker, chunk IDs, heading classifier + carry-forward, munitorum parser, column detection helpers, dedup)
- `test_retrieval_filter.py` — 25 unit tests for the retrieval filter module (parsing Claude responses, subject extraction from nested filters, three-tier fallback with mock ChromaDB)
- `test_full_corpus.py` — end-to-end smoke test running the real pipeline on all 3 provided PDFs. Verifies: filename classification, assessment, segmentation, carry-forward, record building, no duplicate IDs, expected content present. All 3 PDFs pass.
- `test_dedup_real.py` — targeted verification of the dedup fix on Grey Knights page 3.

Plus `debug_*.py` scripts used during investigation, kept for future debugging.

## Still not done

These need the live deployment stack (chromadb + Ollama) or additional files I didn't have:

1. **`deploy.ps1`** — I didn't see the file. If its only job is to run `python ingest.py`, it should still work. If it does anything else with metadata field names, check for `content_category` / `category` / `faction_rules` references.

2. **Full re-ingest.** Everything is ready. Run `.\deploy.ps1 -Action ingest -Message "Refactored pipeline v2: column-aware extraction, heading reset, subject normalization, Munitorum faction tagging, dedup area-overlap fix, whitespace-collision fix, underscore filename support"`.

3. **`test_retrieval.py` comparison.** Run after re-ingest. The four target improvements from v1 should still hold:
   - Grey Knights combat patrol queries should now return results (was 0 under the old `content_category="faction_rules"` filter which no longer exists)
   - "Librarius Conclave" should return Space Marines Librarius chunks, not Grey Knights Spearpoint Paragon chunks (assuming subject filter picks the right faction)
   - "Nemesis Dreadknight points" should return Munitorum chunks tagged `munitorum_faction: grey knights`
   - "Teleport Assault" should return Grey Knights datasheet chunks

4. **Munitorum parser on a real Munitorum PDF.** Untested. Unit tests pass on synthetic data; real Munitorum Field Manual PDF needed for full confidence.

## Priority for next session

1. Verify `deploy.ps1` (quick, mostly just look for stale field names)
2. Full re-ingest
3. `test_retrieval.py` comparison — this is where everything finally gets proven
4. If available, run `test_dedup_real.py` pattern on a few more faction packs to make sure the new dedup rule doesn't over-reach on other layouts

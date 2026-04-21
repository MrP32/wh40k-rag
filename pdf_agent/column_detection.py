"""
column_detection.py
===================
Per-page column layout detection for multi-column PDFs.

Location: C:\\Projects\\wh40k-app\\pdf_agent\\column_detection.py

Why this exists:
  WH40K faction packs use a two-column layout for stratagems, enhancements,
  and rules text. pdfplumber's built-in text extraction reads across both
  columns on each physical line, producing garbled text like:
      "As new ground is seized and consecrated, so are the As the enemy moves close"
  where the first half is from the left column and the second half is from
  the right column on the same y-coordinate.

  This module detects column boundaries by analyzing word-position histograms,
  then returns column crop regions that can be extracted independently.

Algorithm:
  1. Extract all words with bounding boxes from the page
  2. Divide the page into NUM_BANDS horizontal bands
  3. In each band, build an x-coordinate histogram of word centers
  4. Find the widest "zero-density gap" in the middle 30-70% of page width
  5. Classify each band as single-column or two-column
  6. Merge adjacent compatible bands into column regions
  7. Return a ColumnLayout that callers can use to crop regions

Handles three real patterns seen in WH40K PDFs:
  - Pure single-column (title pages, datasheets with one large stat block)
  - Pure two-column (dedicated stratagem/enhancement pages)
  - Mixed (intro paragraph full-width at top, two-column below)
"""

import logging
from dataclasses import dataclass, field
from collections import Counter
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

NUM_BANDS               = 6      # horizontal slices for per-band analysis
BUCKET_WIDTH            = 10     # page-units per histogram bucket
MIN_GAP_WIDTH           = 20     # minimum gap width to count as column separator
SEARCH_MIN_FRACTION     = 0.30   # only look for gaps in the middle 30-70%
SEARCH_MAX_FRACTION     = 0.70
MIN_WORDS_PER_BAND      = 10     # bands with fewer words are skipped (too noisy)
COLUMN_X_STABILITY      = 40     # max drift between band splits to consider them the same column structure


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ColumnRegion:
    """A single rectangular column region to extract independently."""
    x0: float
    y0: float
    x1: float
    y1: float
    column_label: str  # "single" | "left" | "right"

    @property
    def bbox(self) -> tuple:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class ColumnLayout:
    """
    The detected column structure for a single page.

    regions is always non-empty. For single-column pages it contains one region
    spanning the whole page. For two-column pages it contains two side-by-side
    regions. For mixed pages it contains a top single-column region followed
    by two bottom regions (left and right).
    """
    layout_type: str                                # "single" | "two_column" | "mixed"
    regions: list = field(default_factory=list)    # list[ColumnRegion], top-to-bottom, left-to-right
    split_x: Optional[float] = None                 # x-coordinate of column divider (None if single)
    transition_y: Optional[float] = None            # y-coordinate where single→two begins (mixed only)
    page_width: float = 0.0
    page_height: float = 0.0


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def _find_column_gap(words, page_width: float) -> Optional[float]:
    """
    Given a list of word dicts in a horizontal band, find the x-coordinate of
    the widest zero-density gap in the middle of the page. Returns None if no
    gap wide enough to be a column boundary exists.
    """
    if len(words) < MIN_WORDS_PER_BAND:
        return None

    x_counter = Counter()
    for w in words:
        x_center = (w["x0"] + w["x1"]) / 2
        bucket = int(x_center // BUCKET_WIDTH) * BUCKET_WIDTH
        x_counter[bucket] += 1

    search_min = int(page_width * SEARCH_MIN_FRACTION)
    search_max = int(page_width * SEARCH_MAX_FRACTION)

    # Walk the search range and accumulate runs of empty buckets
    gaps = []
    gap_start = None
    for x in range(search_min, search_max, BUCKET_WIDTH):
        bucket = (x // BUCKET_WIDTH) * BUCKET_WIDTH
        if x_counter.get(bucket, 0) == 0:
            if gap_start is None:
                gap_start = x
        else:
            if gap_start is not None:
                gaps.append((gap_start, x, x - gap_start))
                gap_start = None
    if gap_start is not None:
        gaps.append((gap_start, search_max, search_max - gap_start))

    if not gaps:
        return None

    widest = max(gaps, key=lambda g: g[2])
    if widest[2] < MIN_GAP_WIDTH:
        return None

    return (widest[0] + widest[1]) / 2


def _analyze_bands(page) -> list:
    """
    Returns a list of (y0, y1, split_x or None, status) for each band on the page.

    status is one of:
      "two_column"     — a clear column gap was detected; split_x is set
      "single_column"  — enough words but no gap found
      "sparse"         — too few words to decide (page margins, footers, figure captions)
    """
    page_w = float(page.width)
    page_h = float(page.height)
    words = page.extract_words() or []

    band_h = page_h / NUM_BANDS
    results = []
    for i in range(NUM_BANDS):
        y0 = i * band_h
        y1 = (i + 1) * band_h
        band_words = [w for w in words if y0 <= float(w["top"]) <= y1]
        if len(band_words) < MIN_WORDS_PER_BAND:
            results.append((y0, y1, None, "sparse"))
            continue
        split = _find_column_gap(band_words, page_w)
        if split is None:
            results.append((y0, y1, None, "single_column"))
        else:
            results.append((y0, y1, split, "two_column"))
    return results


def _stable_split(splits: list) -> Optional[float]:
    """
    Return a stable column split from a list of per-band split x-coordinates.

    Uses the MEDIAN rather than the mean, and only returns a value if the
    majority of bands agree within COLUMN_X_STABILITY. Single outlier bands
    (e.g. a sparse divider band with a spurious gap) don't derail detection.
    Returns None if no consensus exists.
    """
    if not splits:
        return None
    sorted_splits = sorted(splits)
    median = sorted_splits[len(sorted_splits) // 2]
    # How many bands are within tolerance of the median?
    agreeing = [s for s in splits if abs(s - median) <= COLUMN_X_STABILITY]
    if len(agreeing) < max(2, len(splits) // 2 + 1):
        # No majority agrees
        return None
    return sum(agreeing) / len(agreeing)


def _merge_band_analysis(band_results: list, page_width: float, page_height: float) -> ColumnLayout:
    """
    Walk the per-band analysis top-to-bottom and classify the overall page layout.

    The algorithm:
      1. Ignore sparse bands entirely — they're noise
      2. Count single-column vs two-column bands among substantive ones
      3. If 2-col bands dominate, check for a stable split x-coordinate
         (outlier bands can be ignored if the majority agree)
      4. Walk from top: if there's a leading run of single-col bands and a
         trailing run of two-col bands, treat as mixed
      5. Otherwise use the dominant mode

    This is more forgiving than the original strict-purity approach, which
    failed on real WH40K pages that have one band disagreeing due to
    decorative elements or sparse content.
    """
    substantive = [(y0, y1, split, status)
                   for (y0, y1, split, status) in band_results
                   if status != "sparse"]

    if not substantive:
        return _single_layout(page_width, page_height)

    statuses     = [s[3] for s in substantive]
    two_col_ct   = sum(1 for st in statuses if st == "two_column")
    single_ct    = sum(1 for st in statuses if st == "single_column")
    all_splits   = [s[2] for s in substantive if s[2] is not None]

    # Detect mixed: leading single band(s) then trailing two-col bands
    # Requires at least one leading single band — otherwise a spurious top two-col
    # band (e.g. banner with few words) would cause the transition to be y=0.
    if single_ct > 0 and two_col_ct >= 2:
        first_two_col_idx = next((i for i, st in enumerate(statuses) if st == "two_column"), None)
        if first_two_col_idx is not None and first_two_col_idx >= 1:
            leading  = statuses[:first_two_col_idx]
            trailing = statuses[first_two_col_idx:]
            leading_all_single  = all(st == "single_column" for st in leading)
            trailing_mostly_two = (sum(1 for st in trailing if st == "two_column") >= len(trailing) - 1)

            if leading_all_single and trailing_mostly_two:
                trailing_splits = [s[2] for s in substantive[first_two_col_idx:] if s[2] is not None]
                stable = _stable_split(trailing_splits)
                if stable is not None:
                    transition_y = substantive[first_two_col_idx][0]
                    return ColumnLayout(
                        layout_type="mixed",
                        regions=[
                            ColumnRegion(0.0,    0.0,          page_width, transition_y, "single"),
                            ColumnRegion(0.0,    transition_y, stable,     page_height,  "left"),
                            ColumnRegion(stable, transition_y, page_width, page_height,  "right"),
                        ],
                        split_x=stable,
                        transition_y=transition_y,
                        page_width=page_width,
                        page_height=page_height,
                    )

    # Pure two-column (possibly with one outlier band)
    if two_col_ct >= 2 and two_col_ct > single_ct:
        stable = _stable_split(all_splits)
        if stable is not None:
            return ColumnLayout(
                layout_type="two_column",
                regions=[
                    ColumnRegion(0.0,    0.0, stable,     page_height, "left"),
                    ColumnRegion(stable, 0.0, page_width, page_height, "right"),
                ],
                split_x=stable,
                page_width=page_width,
                page_height=page_height,
            )

    # Single-column dominant or ambiguous
    if single_ct >= two_col_ct:
        return _single_layout(page_width, page_height)

    # Fell through — weird layout. Log and safe-fallback.
    log.debug(f"[COLUMN] ambiguous: single={single_ct}, two_col={two_col_ct}, "
              f"splits={all_splits}, falling back to single")
    return _single_layout(page_width, page_height)


def _single_layout(page_width: float, page_height: float) -> ColumnLayout:
    return ColumnLayout(
        layout_type="single",
        regions=[ColumnRegion(0.0, 0.0, page_width, page_height, "single")],
        page_width=page_width,
        page_height=page_height,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_column_layout(page) -> ColumnLayout:
    """
    Analyze a pdfplumber Page and return its column layout.

    Always returns a valid ColumnLayout with at least one region. Failure modes
    (no words on page, weird layouts) all fall through to single-column, which
    matches the pre-existing extraction behavior and is never worse than the
    status quo.
    """
    try:
        page_w = float(page.width)
        page_h = float(page.height)
    except Exception:
        log.warning("[COLUMN] page has no width/height, returning single-column fallback")
        return ColumnLayout(layout_type="single", regions=[])

    try:
        band_results = _analyze_bands(page)
    except Exception as e:
        log.warning(f"[COLUMN] band analysis failed: {e}, falling back to single-column")
        return ColumnLayout(
            layout_type="single",
            regions=[ColumnRegion(0.0, 0.0, page_w, page_h, "single")],
            page_width=page_w,
            page_height=page_h,
        )

    return _merge_band_analysis(band_results, page_w, page_h)


# ---------------------------------------------------------------------------
# Diagnostic helper — called from logging, not from the main pipeline
# ---------------------------------------------------------------------------

def format_layout_summary(layout: ColumnLayout) -> str:
    """Human-readable one-line summary for logging."""
    if layout.layout_type == "single":
        return "single-column"
    if layout.layout_type == "two_column":
        return f"two-column split@x={layout.split_x:.0f}"
    if layout.layout_type == "mixed":
        return (f"mixed single-top to y={layout.transition_y:.0f}, "
                f"then two-column split@x={layout.split_x:.0f}")
    return f"unknown({layout.layout_type})"

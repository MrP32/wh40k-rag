"""
munitorum_parser.py
===================
Faction-aware extraction for the Munitorum Field Manual PDF.

Location: C:\\Projects\\wh40k-app\\pdf_agent\\munitorum_parser.py

Why this exists:
  The Munitorum Field Manual is one PDF that contains points costs for every
  faction. Its structure is: a faction name in large caps (e.g. "GREY KNIGHTS")
  appears on its own at the top of each faction's section, followed by a
  table of unit names and points values. Without faction-tagging, a query
  like "Nemesis Dreadknight points" can't be filtered to Grey Knights content,
  and the search has to trawl the entire document.

  This module detects faction section headers in the Munitorum and tags every
  chunk with the current faction, enabling metadata filtering.

Detection strategy (v2, after real-PDF failures):
  The v1 detector used re.MULTILINE and matched any isolated-line occurrence
  of a faction name anywhere in the chunk. Against the real Munitorum this
  produced catastrophic false positives — 161 chunks tagged "dark angels"
  when Dark Angels actually only has ~10 units. Root cause: faction names
  also appear on isolated lines as part of unit keyword banners, column-
  spillover artifacts, and footer/header text.

  v2 is stricter:
    1. Detection only fires on the FIRST ~5 lines of a chunk (section
       headers appear at the TOP of sections, not in the middle).
    2. Chunks that look like unit datasheets (contain "FACTION KEYWORDS",
       "KEYWORDS:", or "points") are treated as body content and skipped
       for detection — they inherit whatever faction is already active.
    3. "adeptus astartes" is removed from the detection list — it appears
       on every Space Marines-family page as a keyword, not a section.
    4. Faction names are matched LONGEST-FIRST so "chaos space marines"
       wins over "space marines" when both prefixes match.
    5. Detection requires that EXACTLY ONE faction name appears in the
       header region. Multiple matches → treat as ambiguous, skip.

Usage:
  Called from ingest.py when doc_type == "points_costs". Takes the extracted
  chunks for the Munitorum and adds `munitorum_faction` to their metadata.
  Chunks before the first detected faction header stay tagged with an empty
  string (typically the intro and TOC pages).
"""

import re
import logging
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------

# Only inspect the first N non-empty lines of a chunk for faction headers.
# Real section headers appear at the top; later occurrences are body text.
HEADER_LINES_TO_INSPECT = 5

# Lines with these tokens mean "this is a unit datasheet" — faction names in
# datasheets are KEYWORDS, not section headers. Skip detection for such chunks.
BODY_CONTENT_MARKERS = (
    "FACTION KEYWORDS",
    "KEYWORDS:",
    "pts",
    "models",
)


# ---------------------------------------------------------------------------
# Known faction names — curated from the current 10th edition lineup.
#
# IMPORTANT: ordering matters. Longer / more specific names must come FIRST
# so that "chaos space marines" matches before "space marines" would try to
# match against "CHAOS SPACE MARINES". Python's regex alternation is ordered.
# ---------------------------------------------------------------------------

FACTION_NAMES = [
    # Multi-word / most specific first — Imperium
    "chaos space marines",   # must come before "space marines"
    "genesis chapter",       # any other sub-chapter that sneaks in
    "blood angels",
    "dark angels",
    "space wolves",
    "black templars",
    "grey knights",
    "imperial knights",
    "imperial agents",
    "adepta sororitas",
    "adeptus custodes",
    "adeptus mechanicus",
    "astra militarum",
    "deathwatch",
    "space marines",         # general category after the specific chapters
    # Chaos
    "chaos daemons",
    "chaos knights",
    "death guard",
    "thousand sons",
    "world eaters",
    "emperor's children",
    # Xenos — most specific first
    "craftworld aeldari",
    "genestealer cults",
    "leagues of votann",
    "t'au empire",
    "tau empire",
    "aeldari",
    "drukhari",
    "necrons",
    "orks",
    "tyranids",
    "votann",
]

# Compile once. Each pattern anchors to line boundaries and requires the
# line to contain ONLY the faction name (optionally surrounded by whitespace).
_FACTION_PATTERNS = [
    (name, re.compile(rf"^\s*{re.escape(name.upper())}\s*$"))
    for name in FACTION_NAMES
]


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def _is_body_line(line: str) -> bool:
    """
    Does this single line look like body content (keyword banner, points row,
    unit count)? Used to reject chunks whose TOP LINE is already body,
    without penalizing chunks that have a faction header followed by body.
    """
    return any(marker in line for marker in BODY_CONTENT_MARKERS)


def _candidate_header_lines(text: str) -> list:
    """Return the first HEADER_LINES_TO_INSPECT non-empty lines of the chunk."""
    out = []
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            continue
        out.append(stripped)
        if len(out) >= HEADER_LINES_TO_INSPECT:
            break
    return out


def detect_faction_in_text(text: str) -> Optional[str]:
    """
    Check if the FIRST FEW LINES of the chunk contain exactly one faction
    header. Returns the canonical lowercase name, or None.

    Logic:
      1. Pull the first HEADER_LINES_TO_INSPECT non-empty lines.
      2. If the FIRST line is body content (keyword banner, points row),
         this chunk has no header at the top — return None.
         (This handles the middle-of-section case.)
      3. Check each inspected line against faction patterns.
      4. Require exactly one match; reject if zero or multiple matched.
    """
    if not text:
        return None

    header_lines = _candidate_header_lines(text)
    if not header_lines:
        return None

    # If the very first line is already body content, there's no header here.
    # This is what rejects "FACTION KEYWORDS: GREY KNIGHTS" or "Brother-Captain
    # Stern 1 model 90 pts" at the top of a chunk without blocking legitimate
    # "GREY KNIGHTS" headers that happen to have unit-row lines right after.
    if _is_body_line(header_lines[0]):
        return None

    matches = []
    for line in header_lines:
        for name, pattern in _FACTION_PATTERNS:
            if pattern.match(line):
                matches.append(name)
                break  # only one faction can match a given line

    if len(matches) != 1:
        return None
    return matches[0]


def tag_chunks_with_faction(chunks: list) -> list:
    """
    Walk Munitorum chunks in document order, detect faction headers, and
    annotate each chunk's metadata with `munitorum_faction`.

    Maintains a carry-forward faction: once a header is seen, all subsequent
    chunks are tagged with that faction until a different header appears.
    Body chunks (datasheets, points rows) never update the carry-forward —
    they inherit.

    Input chunks are dicts matching the segmenter's output schema (with at
    least `text` and `metadata` keys). Modifies chunks in place AND returns
    the list for convenience.
    """
    current_faction = ""
    faction_counts = {}
    detected_chunks = 0

    for chunk in chunks:
        text = chunk.get("text", "")
        detected = detect_faction_in_text(text)
        if detected:
            current_faction = detected
            faction_counts[detected] = faction_counts.get(detected, 0) + 1
            detected_chunks += 1

        # Attach to metadata (create if needed)
        if "metadata" not in chunk:
            chunk["metadata"] = {}
        chunk["metadata"]["munitorum_faction"] = current_faction

    log.info(
        f"[MUNITORUM] tagged {len(chunks)} chunks; detected headers in "
        f"{detected_chunks} chunks across {len(faction_counts)} factions: "
        f"{dict(sorted(faction_counts.items()))}"
    )
    return chunks

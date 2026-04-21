"""
munitorum_parser.py
===================
Faction-aware extraction for the Munitorum Field Manual PDF.

Location: C:\\Projects\\wh40k-app\\pdf_agent\\munitorum_parser.py

Why this exists:
  The Munitorum Field Manual is one PDF that contains points costs for every
  faction. Its structure is: a faction name in large caps (e.g. "GREY KNIGHTS")
  appears on its own, followed by a table of unit names and points values for
  that faction. Without faction-tagging, a query like "Nemesis Dreadknight
  points" can't be filtered to Grey Knights content, and the search has to
  trawl the entire document.

  This module detects faction section headers in the Munitorum and tags every
  chunk with the current faction, enabling metadata filtering.

Detection strategy:
  A "faction header" in the Munitorum is a line containing only one of the
  known faction names in all caps, short (< 60 chars), and NOT followed by
  typical body-text words. The detection uses a curated list of faction names
  rather than heuristics — Munitorum faction names are stable and finite.

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
# Known faction names — curated from the current 10th edition lineup.
# Normalized to lowercase for matching; original-case versions are used for
# output labels.
# ---------------------------------------------------------------------------

FACTION_NAMES = [
    # Imperium
    "space marines",
    "adeptus astartes",
    "blood angels",
    "dark angels",
    "space wolves",
    "black templars",
    "deathwatch",
    "grey knights",
    "imperial knights",
    "imperial agents",
    "adepta sororitas",
    "adeptus custodes",
    "adeptus mechanicus",
    "astra militarum",
    # Chaos
    "chaos space marines",
    "chaos daemons",
    "chaos knights",
    "death guard",
    "thousand sons",
    "world eaters",
    "emperor's children",
    # Xenos
    "aeldari",
    "craftworld aeldari",
    "drukhari",
    "genestealer cults",
    "leagues of votann",
    "necrons",
    "orks",
    "tau empire",
    "t'au empire",
    "tyranids",
    # Votann sometimes listed separately
    "votann",
]

# Pre-compile matchers
_FACTION_PATTERNS = {
    name: re.compile(rf"^\s*{re.escape(name.upper())}\s*$", re.MULTILINE)
    for name in FACTION_NAMES
}


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_faction_in_text(text: str) -> Optional[str]:
    """
    Check if the text contains an isolated faction-name line.
    Returns the faction name (lowercased canonical form) or None.

    "Isolated" means the faction name appears on its own line, in all caps,
    with only whitespace around it. This avoids matching incidental mentions
    like "for Space Marines units..." in body text.
    """
    for name, pattern in _FACTION_PATTERNS.items():
        if pattern.search(text):
            return name
    return None


def tag_chunks_with_faction(chunks: list) -> list:
    """
    Walk Munitorum chunks in document order, detect faction headers, and
    annotate each chunk's metadata with `munitorum_faction`.

    The algorithm maintains a carry-forward faction: once a faction header is
    seen, all subsequent chunks are tagged with that faction until a different
    faction header appears.

    Input chunks are dicts matching the segmenter's output schema (with at
    least `text` and `metadata` keys). Modifies chunks in place AND returns
    the list for convenience.
    """
    current_faction = ""
    faction_counts = {}

    for chunk in chunks:
        text = chunk.get("text", "")
        detected = detect_faction_in_text(text)
        if detected:
            current_faction = detected
            faction_counts[detected] = faction_counts.get(detected, 0) + 1

        # Attach to metadata (create if needed)
        if "metadata" not in chunk:
            chunk["metadata"] = {}
        chunk["metadata"]["munitorum_faction"] = current_faction

    log.info(f"[MUNITORUM] tagged {len(chunks)} chunks across "
             f"{len(faction_counts)} factions: {dict(sorted(faction_counts.items()))}")
    return chunks

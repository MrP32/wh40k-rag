"""
heading_classifier.py
=====================
Tightened section-type classification and heading carry-forward.

Location: C:\\Projects\\wh40k-app\\pdf_agent\\heading_classifier.py

Replaces:
  - `_classify_chunk()` in pdf_agent.py (over-eager regex, 80% false positives)
  - `enrich_with_unit_names()` in pdf_agent.py (dead code path)
  - The naive forward-pass in ingest.py (no reset logic)

Design principles:
  1. A heading is only recognized if multiple signals agree. The old code fired
     on any 1-5 allcaps words followed by lowercase, which matched dozens of
     non-heading things per page (stat table headers, keyword rows, weapon
     attribute lines, watermarks).

  2. Each recognized heading type needs a "subtype marker" within a few lines
     to count as a real heading. E.g. "WARPBANE TASK FORCE" alone isn't a
     heading — "WARPBANE TASK FORCE – BATTLE TACTIC STRATAGEM" on the next
     line confirms it.

  3. Carry-forward RESETS on:
       - new column boundary crossed
       - new page
       - section_type transition into a RESET_TYPES category
     The old code never reset, which is why SPEARPOINT PARAGON bled onto 64
     unrelated chunks.

  4. If a chunk has no confident heading, it gets no prefix. This is safer
     than guessing wrong and poisoning the embedding.
"""

import re
import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Section types
# ---------------------------------------------------------------------------

SECTION_TYPES = {
    "unit_datasheet",   # a unit with stats
    "stratagem",        # named stratagem card
    "objective",        # mission objective
    "ability",          # faction/detachment ability
    "enhancement",      # enhancement/relic/warlord trait
    "detachment_rule",  # army-level rule for a detachment
    "rules",            # core rules, glossary, universal special rules
    "narrative",        # lore / fluff text
    "general",          # catchall for unclassified content
}

# Section types where heading context must reset rather than carry forward.
# These are types where each entry is self-contained (a stratagem is one
# stratagem, not the intro to the next). `rules` and `narrative` are allowed
# to carry because those sections can span multiple chunks meaningfully.
RESET_TYPES = {"unit_datasheet", "stratagem", "enhancement", "objective", "detachment_rule"}


# ---------------------------------------------------------------------------
# Subtype markers — lines that confirm what KIND of section a heading belongs to
# ---------------------------------------------------------------------------

# A line like "WARPBANE TASK FORCE – BATTLE TACTIC STRATAGEM" or
# "AURELLIOS' BANISHERS – STRATEGIC PLOY STRATAGEM"
_STRATAGEM_MARKER = re.compile(
    r"^[A-Z][A-Z0-9'\s]+(?:\s+[–-]\s+)(BATTLE TACTIC|STRATEGIC PLOY|BATTLEFIELD|EPIC DEED|WARGEAR|REQUISITION)\s+STRATAGEM",
    re.MULTILINE,
)

# Datasheet markers in combat patrol and faction packs
_DATASHEET_MARKER = re.compile(r"Combat Patrol Datasheet|WARHAMMER\s+LEGENDS", re.IGNORECASE)

# Enhancement section markers
_ENHANCEMENT_SECTION_MARKER = re.compile(
    r"^(DEFAULT ENHANCEMENT|OPTIONAL ENHANCEMENT|ENHANCEMENTS)\s*$",
    re.MULTILINE,
)

# Objective markers
_OBJECTIVE_SECTION_MARKER = re.compile(
    r"^(DEFAULT SECONDARY OBJECTIVE|OPTIONAL SECONDARY OBJECTIVE|SECONDARY OBJECTIVES|PRIMARY OBJECTIVE)\s*$",
    re.MULTILINE,
)

# Detachment rule markers
_DETACHMENT_RULE_MARKER = re.compile(
    r"^DETACHMENT RULE\s*$",
    re.MULTILINE,
)

# Stat row — presence indicates unit datasheet
_STAT_HEADER_ROW = re.compile(
    r"^\s*M\s+T\s+SV\s+W\s+LD\s+OC\s*$",
    re.MULTILINE,
)


# ---------------------------------------------------------------------------
# Candidate heading extraction
# ---------------------------------------------------------------------------

# Allcaps heading on its own line, 2-6 words, between 3 and 60 chars.
# Excludes lines with embedded lowercase (those would be prose with a few
# capitalized words, not headings).
# Critical: [^\S\r\n]* (horizontal whitespace only) rather than \s* which would
# swallow newlines and smoosh multiple headings together.
_CANDIDATE_HEADING_LINE = re.compile(
    r"^[A-Z][A-Z0-9'\-][A-Z0-9'\- \t]*[A-Z0-9']$",
    re.MULTILINE,
)

# Hard denylist — strings that look like headings but aren't. Pruned vs. the
# original 26-item list: we now catch many false positives via subtype markers,
# so this list only needs the truly-universal junk.
_STRUCTURAL_NOISE = {
    "M T SV W LD OC",
    "RANGED WEAPONS",
    "MELEE WEAPONS",
    "ABILITIES",
    "KEYWORDS",
    "FACTION KEYWORDS",
    "WARGEAR OPTIONS",
    "UNIT COMPOSITION",
    "LEADER",
    "TRANSPORT",
    "WARGEAR ABILITIES",
    "WARHAMMER LEGENDS",
    "CORE",
    "FACTION",
    "SUPREME COMMANDER",
    "INVULNERABLE SAVE",
    "DEEP STRIKE",
    "DAMAGED",
    "CP",
    "OR",
    "WHEN",
    "TARGET",
    "EFFECT",
    "RESTRICTIONS",
    "CONTENTS",
    "WHAT'S NEW",
    "UPDATES & ERRATA",
}


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class HeadingResult:
    section_type: str          # one of SECTION_TYPES
    heading: str               # the heading text, or "" if none confidently identified
    confident: bool            # True when heading is attached to a subtype marker
    all_candidates: list       # all heading-like strings found (debugging)


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _find_candidates(text: str) -> list:
    """Return all candidate headings (allcaps lines, 2-6 words, not in denylist)."""
    candidates = []
    for m in _CANDIDATE_HEADING_LINE.finditer(text):
        line = m.group(0).strip()
        words = line.split()
        if not (2 <= len(words) <= 6):
            continue
        if line in _STRUCTURAL_NOISE:
            continue
        # Skip lines that are all-single-chars (stat row variations)
        if all(len(w) <= 2 for w in words):
            continue
        candidates.append((m.start(), line))
    return candidates


def _heading_has_stratagem_marker(text: str, heading_start: int) -> bool:
    """Is there a stratagem subtype marker within ~200 chars after the heading?"""
    window = text[heading_start:heading_start + 300]
    return bool(_STRATAGEM_MARKER.search(window))


def classify_chunk(text: str) -> HeadingResult:
    """
    Classify a chunk's section type and confident heading.

    Returns HeadingResult with:
      - section_type: one of SECTION_TYPES
      - heading: the heading string (empty if no confident heading)
      - confident: True when the heading is backed by a subtype marker
      - all_candidates: all heading-like strings found (for debugging)

    If no confident heading is found but the text has rules/narrative signals,
    returns an appropriate section_type with empty heading. The caller should
    NOT prefix the embedding when confident is False.
    """
    if not text or not text.strip():
        return HeadingResult("general", "", False, [])

    candidates = _find_candidates(text)
    candidate_strs = [c[1] for c in candidates]

    # 1. STRATAGEM — heading followed by a stratagem marker
    for start, line in candidates:
        if _heading_has_stratagem_marker(text, start):
            return HeadingResult("stratagem", line, True, candidate_strs)

    # 2. UNIT DATASHEET — presence of stat row + datasheet marker
    has_stat_row = bool(_STAT_HEADER_ROW.search(text))
    has_datasheet_marker = bool(_DATASHEET_MARKER.search(text))
    if has_stat_row and candidates:
        # The first candidate heading is typically the unit name
        return HeadingResult("unit_datasheet", candidates[0][1], True, candidate_strs)
    if has_datasheet_marker and candidates:
        return HeadingResult("unit_datasheet", candidates[0][1], True, candidate_strs)

    # 3. DETACHMENT RULE — DETACHMENT RULE marker present
    if _DETACHMENT_RULE_MARKER.search(text):
        # The named rule usually follows within a few lines of "DETACHMENT RULE"
        marker_match = _DETACHMENT_RULE_MARKER.search(text)
        marker_end = marker_match.end()
        # Find the next candidate heading after the marker
        for start, line in candidates:
            if start > marker_end:
                return HeadingResult("detachment_rule", line, True, candidate_strs)
        # Marker present but no heading found — still classify as detachment_rule
        return HeadingResult("detachment_rule", "", False, candidate_strs)

    # 4. ENHANCEMENT — enhancement section marker present
    if _ENHANCEMENT_SECTION_MARKER.search(text):
        marker_match = _ENHANCEMENT_SECTION_MARKER.search(text)
        marker_end = marker_match.end()
        for start, line in candidates:
            if start > marker_end:
                return HeadingResult("enhancement", line, True, candidate_strs)
        return HeadingResult("enhancement", "", False, candidate_strs)

    # 5. OBJECTIVE — objective section marker present
    if _OBJECTIVE_SECTION_MARKER.search(text):
        marker_match = _OBJECTIVE_SECTION_MARKER.search(text)
        marker_end = marker_match.end()
        for start, line in candidates:
            if start > marker_end:
                return HeadingResult("objective", line, True, candidate_strs)
        return HeadingResult("objective", "", False, candidate_strs)

    # 6. No confident heading — fall back to content-based typing.
    # Require at least 2 signal hits to avoid classifying every "phase" mention
    # as a rules chunk. Narrative/rules classification here is coarse — it just
    # informs downstream behavior (e.g. whether to carry headings forward).
    lower = text.lower()
    narrative_hits = len(re.findall(
        r"\b(lore|legend|history|ancient|millennia|saga|chapter master|primarch|crusade|heresy)\b",
        lower,
    ))
    rules_hits = len(re.findall(
        r"\b(core rules|universal special rules|glossary|appendix|page \d+|errata)\b",
        lower,
    ))

    if narrative_hits >= 2 and narrative_hits > rules_hits:
        return HeadingResult("narrative", "", False, candidate_strs)
    if rules_hits >= 2:
        return HeadingResult("rules", "", False, candidate_strs)

    return HeadingResult("general", "", False, candidate_strs)


# ---------------------------------------------------------------------------
# Carry-forward state machine
# ---------------------------------------------------------------------------

@dataclass
class CarryState:
    """Tracks what section we're "inside" as we walk chunks in document order."""
    current_heading: str = ""
    current_type: str = "general"
    last_page: int = -1
    last_column: str = ""      # "single" | "left" | "right" (from column_detection)


def apply_carry_forward(chunks: list) -> list:
    """
    Walk chunks in document order and apply heading carry-forward with resets.

    Rules:
      - When a chunk has a confident heading, it becomes the new current heading
      - When crossing a page boundary, reset if last heading was a RESET_TYPE
      - When crossing a column boundary on the same page, reset if last heading
        was a RESET_TYPE
      - Otherwise, carry forward

    Each chunk gets `section_type_carried` and `section_identifier_carried`
    added to its top-level keys (matching the segmenter's existing convention).
    When a chunk has its own confident heading, those fields equal the
    chunk's own values. When it doesn't, they inherit from state.

    Chunks are expected to be dicts with at least:
      - page_number: int
      - region_index: int (or 0)
      - column_label: str ("single", "left", "right") — NEW field from segmenter
      - text: str
      - classification: HeadingResult — NEW field set by earlier call to classify_chunk
    """
    state = CarryState()
    reset_events = 0
    carried_events = 0

    for chunk in chunks:
        page = chunk.get("page_number", 0)
        column = chunk.get("column_label", "")
        classification: HeadingResult = chunk.get("classification")

        # Determine whether to reset state BEFORE attaching to this chunk
        crossed_page = page != state.last_page
        crossed_column = (page == state.last_page and column != state.last_column)

        should_reset = (
            (crossed_page or crossed_column)
            and state.current_type in RESET_TYPES
        )
        if should_reset:
            state.current_heading = ""
            state.current_type = "general"
            reset_events += 1

        # If this chunk has its own confident heading, it takes over
        if classification and classification.confident and classification.heading:
            state.current_heading = classification.heading
            state.current_type = classification.section_type
        elif classification and classification.section_type in ("narrative", "rules"):
            # Type-only info (no heading) — update type but keep heading from prior section
            # unless we'd be carrying a RESET_TYPE heading into non-RESET_TYPE content,
            # in which case clear the heading.
            if state.current_type in RESET_TYPES:
                state.current_heading = ""
            state.current_type = classification.section_type
        else:
            carried_events += 1

        # Attach state to chunk
        chunk["section_type"] = state.current_type
        chunk["section_identifier"] = state.current_heading

        # Update last-seen trackers
        state.last_page = page
        state.last_column = column

    log.info(f"[HEADING] carry-forward applied: {reset_events} resets, "
             f"{carried_events} carries, {len(chunks)} chunks total")
    return chunks

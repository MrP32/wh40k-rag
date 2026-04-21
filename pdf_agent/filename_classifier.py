"""
filename_classifier.py
======================
Single source of truth for extracting doc_type, subject, and ancillary
metadata from a PDF filename.

Location: C:\\Projects\\wh40k-app\\pdf_agent\\filename_classifier.py

Replaces the three overlapping functions in the old ingest.py:
    extract_doc_type()
    extract_content_category()
    extract_subject()

Those three functions re-pattern-matched the same filename for the same
concepts and produced disagreeing values (e.g. filename "Core Rules
Updates.pdf" → doc_type="core_rules" but content_category="core_rules_updates").

This module collapses them into one function returning a single dataclass.

Key fix: for combat patrol filenames of the form
    "Combat Patrol - <FACTION> - <PATROL_NAME>.pdf"
the old code set subject = "<FACTION> - <PATROL_NAME>", which never matched
queries for "<FACTION>". Now subject = "<FACTION>" and patrol_name is a
separate field.
"""

import re
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class FilenameMetadata:
    doc_type: str              # canonical document category
    subject: str               # faction / rulebook name (normalized)
    patrol_name: str = ""      # for combat_patrol only; the named box set
    is_legends: bool = False   # for imperial_armour / legends content


# ---------------------------------------------------------------------------
# Ordered pattern rules — most specific first
# Each rule: (regex, doc_type, subject_strategy)
# subject_strategy is either a literal string or a callable that takes the
# regex match and returns the subject string.
# ---------------------------------------------------------------------------

def _combat_patrol_boxed(m):
    """'Combat Patrol - Grey Knights - Aurellios Banishers.pdf' → ('grey knights', 'aurellios banishers')"""
    return m.group(1).strip().lower(), m.group(2).strip().lower()

def _faction_name(m):
    return m.group(1).strip().lower()


RULES = [
    # Most specific combat patrol variants first
    (re.compile(r"^combat patrol\s*-\s*([^-]+?)\s*-\s*(.+)\.pdf$", re.I),
     "combat_patrol", _combat_patrol_boxed),

    (re.compile(r"^combat patrol\s*-\s*(.+)\.pdf$", re.I),
     "combat_patrol", lambda m: (_faction_name(m), "")),

    (re.compile(r"^combat patrol rules\.pdf$", re.I),
     "combat_patrol_rules", lambda m: ("combat patrol rules", "")),

    # Faction packs
    (re.compile(r"^faction pack\s*-\s*(.+)\.pdf$", re.I),
     "faction_pack", lambda m: (_faction_name(m), "")),

    # Core rules variants
    (re.compile(r"^core rules updates.*\.pdf$", re.I),
     "core_rules_updates", lambda m: ("core rules updates", "")),

    (re.compile(r"^core rules quick.?start.*\.pdf$", re.I),
     "core_rules_quickstart", lambda m: ("core rules quickstart", "")),

    (re.compile(r"^core rules\.pdf$", re.I),
     "core_rules", lambda m: ("core rules", "")),

    # Points / Munitorum
    (re.compile(r"^munitorum.*\.pdf$", re.I),
     "points_costs", lambda m: ("munitorum field manual", "")),

    # Balance / dataslate
    (re.compile(r"^balance dataslate.*\.pdf$", re.I),
     "balance_dataslate", lambda m: ("balance dataslate", "")),

    # Crusade / campaign
    (re.compile(r"^crusade rules.*\.pdf$", re.I),
     "crusade_rules", lambda m: ("crusade rules", "")),

    (re.compile(r"^boarding actions.*\.pdf$", re.I),
     "boarding_actions_rules", lambda m: ("boarding actions", "")),

    # Tournament packs
    (re.compile(r"^chapter approved.*tournament.*\.pdf$", re.I),
     "tournament_rules", lambda m: ("chapter approved tournament companion", "")),

    (re.compile(r"^pariah nexus.*tournament.*\.pdf$", re.I),
     "tournament_rules", lambda m: ("pariah nexus tournament companion", "")),

    # Imperial Armour
    (re.compile(r"^imperial armour\s*-\s*(.+)\.pdf$", re.I),
     "imperial_armour", lambda m: (_faction_name(m), "")),

    # Army roster
    (re.compile(r"^army roster.*\.pdf$", re.I),
     "army_roster", lambda m: ("army roster", "")),
]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def classify_filename(filename: str) -> FilenameMetadata:
    """
    Map a PDF filename to its metadata. Falls back to ('other', filename_stem)
    if nothing matches — which is better than silently returning wrong data.

    Normalization: underscores are treated as spaces before matching, because
    web-downloaded copies of these PDFs often arrive with underscores in place
    of spaces (e.g. 'Faction_Pack_-_Grey_Knights.pdf'). The original filename
    is retained for the is_legends detection and other substring checks.
    """
    # Use only the basename (strip any path)
    original = Path(filename).name
    # Normalize: collapse runs of underscores to single spaces. Applies to all
    # patterns since underscore-variant filenames appear throughout the corpus.
    name = re.sub(r"_+", " ", original)
    # Also collapse any now-doubled spaces produced by mixed separators
    name = re.sub(r"\s+", " ", name).strip()

    for pattern, doc_type, subject_strategy in RULES:
        m = pattern.match(name)
        if not m:
            continue
        if callable(subject_strategy):
            result = subject_strategy(m)
            if isinstance(result, tuple):
                subject, patrol_name = result
            else:
                subject, patrol_name = result, ""
        else:
            subject, patrol_name = subject_strategy, ""

        # Check is_legends against the original (in case the literal token
        # "legends" only appears in a form the normalization affected).
        is_legends = "legends" in original.lower()
        return FilenameMetadata(
            doc_type=doc_type,
            subject=subject,
            patrol_name=patrol_name,
            is_legends=is_legends,
        )

    # No rule matched — log and return sane fallback
    log.warning(f"[FILENAME] no classification rule matched: {original!r} — using 'other'")
    stem = Path(original).stem.lower()
    return FilenameMetadata(doc_type="other", subject=stem)

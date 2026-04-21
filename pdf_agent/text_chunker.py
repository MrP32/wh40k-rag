"""
text_chunker.py
===============
Simple recursive text chunker with deterministic overlap.

Location: C:\\Projects\\wh40k-app\\pdf_agent\\text_chunker.py

Replaces `_chunk_text` in pdf_agent.py, which had an overlap that varied from
~0 to ~100 chars depending on word boundary placement, and could slice
mid-word on text with long unbroken runs.

Algorithm:
  Recursive splitter. Tries to cut on the most natural boundary first:
    1. Paragraph break (\\n\\n)
    2. Line break (\\n)
    3. Sentence boundary (. or ! or ?)
    4. Word boundary (space)
    5. Character boundary (last resort)
  At each level, if no cut would produce a chunk ≤ CHUNK_SIZE, falls through
  to the next separator. This ensures chunks are always reasonable AND
  respect natural structure.

  Overlap is applied consistently by including the last OVERLAP characters
  of each chunk as the start of the next chunk.
"""

from typing import List

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100
SEPARATORS    = ["\n\n", "\n", ". ", " ", ""]


def _recursive_split(text: str, max_size: int, separators: List[str]) -> List[str]:
    """
    Split text into pieces no larger than max_size using the best available
    separator at each level. Does NOT apply overlap — just clean splits.
    """
    if len(text) <= max_size:
        return [text] if text.strip() else []

    # Find the best separator that appears in this text
    sep = ""
    for s in separators:
        if s in text or s == "":
            sep = s
            break

    if sep == "":
        # Last resort: hard split at max_size
        return [text[i:i + max_size] for i in range(0, len(text), max_size)]

    parts = text.split(sep)
    # Re-attach separators so text round-trips cleanly
    if sep != "":
        parts = [p + (sep if i < len(parts) - 1 else "") for i, p in enumerate(parts)]

    # Greedily pack parts into chunks, recursing on any part that's still too big
    chunks = []
    current = ""
    for part in parts:
        if len(part) > max_size:
            # This part alone is too big — flush what we have, then recurse
            if current:
                chunks.append(current)
                current = ""
            next_sep_idx = separators.index(sep) + 1 if sep in separators else len(separators)
            chunks.extend(_recursive_split(part, max_size, separators[next_sep_idx:]))
            continue

        if len(current) + len(part) <= max_size:
            current += part
        else:
            if current:
                chunks.append(current)
            current = part
    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]


def _apply_overlap(chunks: List[str], overlap: int) -> List[str]:
    """Prepend the last `overlap` chars of each chunk to the next chunk."""
    if len(chunks) <= 1 or overlap <= 0:
        return chunks
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        prev_tail = chunks[i - 1][-overlap:]
        # Try to not split prev_tail mid-word — back up to the nearest space
        space_idx = prev_tail.find(" ")
        if 0 < space_idx < overlap // 2:
            prev_tail = prev_tail[space_idx + 1:]
        out.append(prev_tail + chunks[i])
    return out


def chunk_text(text: str, max_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split `text` into chunks of at most ~(max_size + overlap) characters,
    respecting paragraph / line / sentence / word boundaries in that priority.
    """
    if not text or not text.strip():
        return []
    base = _recursive_split(text.strip(), max_size, SEPARATORS)
    return _apply_overlap(base, overlap)

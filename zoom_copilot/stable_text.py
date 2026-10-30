"""Presentation boundaries for append-only model text.

The original snapshots remain stored by the caller.  Readers receive complete
sentences or paragraphs while streaming, and the entire text on completion.
This module never translates, repairs grammar, or changes scientific claims.
"""
from __future__ import annotations

import re


_BOUNDARY = re.compile(r'[.!?]+[\"\'”’）\])]*(?=\s|$)|[。！？]+[\"\'”’）\])]*|\n[ \t]*\n')
_ABBREVIATION = re.compile(
    r"(?:\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|Fig|Figs|Eq|Eqs|Ref|Refs|Sec|No|Nos|cf|etc)|"
    r"\b(?:e\.g|i\.e)|\b(?:[A-Za-z]\.)*[A-Za-z])\.$", re.IGNORECASE)


def completed_text(text: str, *, final: bool = False) -> str:
    """Return the last complete reading unit, without exposing a partial word.

    Paragraph separators count as boundaries even for lists without sentence
    punctuation.  Abbreviations, initials and decimal points do not create a
    sentence boundary.  A final callback always releases remaining text,
    including deliberate fragments, so presentation buffering never drops it.
    """
    value = str(text or "")
    if final:
        return value.strip()
    end = 0
    for match in _BOUNDARY.finditer(value):
        token = match.group()
        if token.startswith("\n"):
            end = match.end()
            continue
        start = match.start()
        if token.startswith("."):
            before = value[:start + 1]
            after = value[match.end():].lstrip()
            if (start and value[start - 1].isdigit() and after[:1].isdigit()
                    or _ABBREVIATION.search(before)):
                continue
        end = match.end()
    return value[:end].strip()

"""Preserve short scientific equations through local neural translation.

No equation is inferred or corrected. Exact formula wording is restored inline;
failure to preserve a placeholder rejects that translation rather than showing a
new, potentially invented equation. English originals remain authoritative.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re


class TranslationProtectionError(ValueError):
    pass


_ATOM = r"(?:[A-Za-zΑ-Ωα-ω](?:_?\d+)?|[-+]?\d+(?:\.\d+)?%?)(?!\w)"
_RHS = _ATOM + r"(?:\s*(?:[+*/×·−-]\s*)?" + _ATOM + r"){0,7}"
_CONTRAST = r"(?:,?\s+(?:not|rather than|instead of)\s+" + _RHS + r")?"
_SPOKEN = re.compile(r"(?<!\w)" + _ATOM + r"\s+(?:equals|is equal to)\s+" + _RHS +
                     _CONTRAST + r"(?!\w)", re.I)
_EQUATION = re.compile(r"(?<!\w)" + _ATOM + r"\s*(?:=|≈|≤|≥|≠)\s*" + _RHS +
                       _CONTRAST + r"(?!\w)", re.I)


@dataclass(frozen=True)
class ProtectedTranslation:
    original: str
    text: str
    slots: tuple[tuple[str, str], ...] = ()

    def restore(self, translated: str) -> str:
        result = str(translated or "")
        for marker, literal in self.slots:
            matches = list(re.finditer(re.escape(marker), result, re.I))
            if len(matches) != 1:
                raise TranslationProtectionError("formula_placeholder_not_preserved")
            result = re.sub(re.escape(marker), lambda _m: literal, result, flags=re.I)
        return normalize_research_translation(self.original, result)


def protect_translation_source(source: str) -> ProtectedTranslation:
    """Protect written equations and explicit spoken equality/contrast phrases.

    For ``s equals D i not M i``, the Chinese retains that exact English phrase.
    This is intentionally preferable to inventing ``s = Di - Mi``. The original
    formula is also already visible immediately above the Chinese subtitle.
    """
    source = str(source or "")
    spans = sorted([(m.start(), m.end()) for pattern in (_SPOKEN, _EQUATION)
                    for m in pattern.finditer(source)], key=lambda item: (item[0], -item[1]))
    selected = []
    for start, end in spans:
        if not selected or start >= selected[-1][1]:
            selected.append((start, end))
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:8].upper()
    prefix = "ZXQFORMULA" + digest
    while prefix.casefold() in source.casefold():
        prefix += "X"
    slots = []
    output = []
    cursor = 0
    for index, (start, end) in enumerate(selected):
        marker = prefix + str(index) + "QXZ"
        output.extend((source[cursor:start], marker))
        slots.append((marker, source[start:end]))
        cursor = end
    output.append(source[cursor:])
    return ProtectedTranslation(source, "".join(output), tuple(slots))


def normalize_research_translation(source: str, translated: str) -> str:
    """Narrow source-conditioned optical terminology, not a content corrector."""
    output = str(translated or "")
    # Polarization also means political/social/electrochemical polarization.
    # An optical interpretation needs explicit source evidence, not app history.
    optical = re.search(r"\b(?:optics?|optical|photons?|light|laser|camera|imaging|"
        r"Stokes|polarimeters?|polarizers?|retarders?|analy[sz]ers?|instrument matrix|"
        r"linear polarization|circular polarization)\b", source, re.I)
    nonoptical = re.search(r"\b(?:politic\w*|social|society|voters?|partisan|electrode|"
        r"electrochem\w*|dielectric|magnetic)\b", source, re.I)
    if optical and not nonoptical and re.search(r"\bpolariz(?:ation|ed|ing|er|ers)\b", source, re.I):
        output = output.replace("极化", "偏振")
    # In linear algebra an inverse matrix is not the scalar reciprocal. Do not
    # touch an intentional reciprocal comparison or other nonmatrix uses.
    if (re.search(r"\b(?:inverse|inversion)\b", source, re.I)
            and re.search(r"\bmatri(?:x|ces)\b", source, re.I)
            and not re.search(r"\breciprocal\b", source, re.I)):
        output = output.replace("矩阵的倒数", "矩阵的逆")
        output = output.replace("矩阵倒数", "矩阵的逆")
    return output

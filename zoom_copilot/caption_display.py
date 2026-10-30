"""Display-only normalization shared by workspace and floating caption readers.

Source IDs and ASR/translation text remain in their ledger. A newline inserted
by a punctuation restorer is not a new translation unit or speaker turn.
"""
from __future__ import annotations


def caption_english(text: str) -> str:
    return " ".join(str(text or "").splitlines()).strip()


def caption_units(record: dict) -> list[tuple[str, str]]:
    """Use explicit source pairs, never guess alignment from sentence counts."""
    units = record.get("units")
    if not units:
        units = [(record.get("en", ""), record.get("zh", ""))]
    return [(caption_english(en), str(zh or "").strip()) for en, zh in units
            if str(en or "").strip() or str(zh or "").strip()]


def confirmed_caption_state(state: str) -> bool:
    """Only confirmation-marked live events can bypass the transcript ledger."""
    return str(state).startswith(("已成句", "已确认", "已保留"))


def caption_follow_target(current: int, maximum: int, top: int, bottom: int,
                          viewport_height: int, *, prefer_start: bool = False) -> int:
    """Keep a visible unit still; advance with reading room when it overflows.

    A newest unit that fits is kept complete, English included. Callers may
    prefer the English beginning of an over-screen unit to avoid replacing it
    with only a Chinese tail; its entire text remains in the scroll area.
    Trailing layout space makes reading anchors possible without
    moving already-readable lines on each additional word-wrap.
    """
    height = max(1, int(viewport_height))
    margin = min(12, height // 12)
    top, bottom = int(top), int(bottom)
    fits = bottom - top <= height - margin
    if prefer_start and not fits:
        return max(0, min(maximum, top))
    if bottom - current <= height - margin:
        if not fits or top >= current - margin:
            return max(0, min(maximum, current))
        return max(0, min(maximum, top))
    reserve = min(96, max(32, height // 3))
    target = max(current, bottom - height + margin + reserve)
    if fits:
        target = min(target, top)
    target = max(0, min(maximum, target))
    # Single-pixel layout rounding is not a new reading position.
    return current if abs(target - current) <= 2 else target

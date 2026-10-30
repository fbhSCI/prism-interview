"""Explicit, local suggestions for narrow conversational coordination.

These are editable suggestions, not facts about the candidate and not speech.
Research questions always continue through the selected answer provider.
"""
from __future__ import annotations

from .answers import BilingualAnswer
from .interview_context import analyze_question


def quick_coordination_reply(question: str, *, packet=None, request_id=""):
    structure = analyze_question(question)
    if structure.interaction_kind != "coordination":
        return None
    # No guessed scientific or personal answer is generated here.
    return BilingualAnswer(
        question_zh="你想先开始，还是希望我先开始？",
        answer_en="Please go ahead. I'll follow.",
        answer_zh="请您先开始，我接着说。",
        sources=[], request_id=request_id, provider="local-template",
        model="本地沟通建议",
        warnings=["这是可修改的轮流发言建议；未调用回答模型或检索论文。"],
        diagnostics={"local_coordination": True, "cloud_answer_calls": 0},
    )

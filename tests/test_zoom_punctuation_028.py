"""Regressions for punctuation failures seen during the two-end Zoom check."""
import re

import pytest

from zoom_copilot.transcript_text import (
    TranscriptRestorer, repair_fragment_boundaries,
)


@pytest.mark.parametrize("raw,bad", [
    ("What are you doing?", "What, are you doing?"),
    ("Could you explain?", "Could, you explain?"),
    ("All right.", "ALL, right."),
    ("I can't do it.", "I. Can't do IT."),
])
def test_short_complete_zoom_sentences_do_not_get_repunctuated(raw, bad):
    result = TranscriptRestorer(punctuator=lambda _: bad).restore(raw)
    assert result.raw == raw
    assert result.clean == raw


def test_model_does_not_create_acronyms_or_break_conditional_question_prefix():
    raw = "if the scattering level increases how would you separate material reflectance from illumination"
    predicted = "IF. The scattering level increases, HOW. Would you separate material reflectance from illumination."
    result = TranscriptRestorer(punctuator=lambda _: predicted).restore(raw)
    assert result.raw == raw
    assert result.clean == (
        "If the scattering level increases, how would you separate material reflectance from illumination?"
    )
    assert len(result.sentences) == 1


def test_all_capital_scientific_source_and_variables_survive_prefix_guards():
    raw = "IF measures the intermediate frequency. SO denotes spin orbit. US is the sample label. i = M s."
    predicted = "If measures the intermediate frequency. So denotes spin orbit. Us is the sample label. I = M S."
    result = TranscriptRestorer(punctuator=lambda _: predicted).restore(raw)
    assert result.raw == raw
    assert "IF measures" in result.clean
    assert "SO denotes" in result.clean
    assert "US is" in result.clean
    assert "i = M s." in result.clean
    assert repair_fragment_boundaries("IF. The next frequency is 95% of SO.") == (
        "IF. The next frequency is 95% of SO."
    )


def test_repair_keeps_lexical_uncertainty_negation_and_numbers():
    raw = "I. Can't do it. If. The level is not 0.95, How. Would you measure S3?"
    clean = repair_fragment_boundaries(raw)
    assert clean == "I can't do it. If the level is not 0.95, How would you measure S3?"
    words = lambda text: re.findall(r"\w+(?:['’]\w+)*", text.casefold())
    assert words(clean) == words(raw)


def test_source_word_uncertainty_is_not_replaced_by_expected_research_question():
    raw = "I can't do it. reconstructs a target from intensity measurements."
    predicted = "I. Can't do IT. Reconstructs a target from intensity measurements."
    result = TranscriptRestorer(punctuator=lambda _: predicted).restore(raw)
    assert result.raw == raw
    assert result.clean == "I can't do it. Reconstructs a target from intensity measurements."
    assert "Could you" not in result.clean


@pytest.mark.parametrize("raw,predicted,expected", [
    ("what are you doing", "What, are you doing.", "What are you doing?"),
    ("could you explain", "Could, you explain.", "Could you explain?"),
    ("I cannot measure S3 with 95% confidence", "I. Cannot measure S3 with 95% confidence.",
     "I cannot measure S3 with 95% confidence."),
])
def test_unpunctuated_short_questions_and_pronoun_auxiliary_prefixes(raw, predicted, expected):
    result = TranscriptRestorer(punctuator=lambda _: predicted).restore(raw)
    assert result.clean == expected
    assert result.raw == raw

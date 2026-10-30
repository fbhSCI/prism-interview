"""Source-preserving compact asks; no new audio/model accuracy claim.

These cases use the two public-video transcript forms already observed in
recorded fixtures and the current local display. Full setup remains mandatory
model input; focus is not a generated summary or a word-count truncation.
"""
import pytest

from zoom_copilot.question_packet import QuestionContext
from zoom_copilot.turns import confirmed_question_boundary, explicit_question_focus


def test_recorded_tum_choice_keeps_both_alternatives_in_one_focus():
    raw = ('Patron Griff asks, first thank you for the fantastic talk. '
           'Which do you think is the direction more promising, focusing on improving '
           'the existing reconstruction pipelines. or fixing them with neural rendering.')
    packet = QuestionContext().build(raw)
    assert len(packet.subquestions) == 1
    assert 'existing reconstruction pipelines.' in packet.answer_question_en
    assert 'or fixing them with neural rendering.' in packet.answer_question_en
    assert 'fantastic talk' not in packet.answer_question_en
    assert packet.raw_question_en == raw
    assert 'fantastic talk' in packet.background_en


@pytest.mark.parametrize('last', [
    'Or you know what other kind of information do you use.',
    'You take out segmenting, or you know what other kind of information do you use.',
])
def test_recorded_albedo_followup_is_an_actual_ask_with_full_setup_preserved(last):
    raw = ("I've got a question about refining the albedo. Yeah. "
           'So, you start out by getting an initial albedo by low-pass filtering '
           'and then when you go from that initial albedo to refine, '
           'do you use any other information? '
           'Do you use the already extracted distracted, diffused illumination? ' + last)
    packet = QuestionContext().build(raw)
    assert len(packet.subquestions) == 3
    assert 'do you use any other information?' in packet.answer_question_en
    assert 'diffused illumination?' in packet.answer_question_en
    assert 'or you know what other kind of information do you use.' in packet.answer_question_en.casefold()
    assert 'low-pass filtering' in packet.background_en
    assert 'low-pass filtering' not in packet.answer_question_en
    assert packet.raw_question_en == raw
    if last.startswith('You'):
        assert 'You take out segmenting,' in packet.background_en
        assert 'segmenting' not in packet.answer_question_en
    for fragment in packet.subquestions:
        assert fragment in packet.question_en
    prompt = packet.to_prompt_data()
    assert prompt['answer_question_en'] == packet.answer_question_en
    assert prompt['subquestions_to_cover'] == list(packet.subquestions)
    # The new focus grammar is not a shortcut for sealing a question early.
    assert confirmed_question_boundary(raw) is None


def test_late_alternative_conditions_and_independent_subquestions_are_never_cut():
    raw = ('How would you estimate the Stokes vector? '
           'Or would you first measure additional linear channels under unknown illumination? '
           'Assume that the target is moving and the irradiance is not calibrated. '
           'Which assumptions fail, and how would you validate the result at 0.95 confidence?')
    packet = QuestionContext().build(raw)
    assert 'additional linear channels under unknown illumination?' in packet.answer_question_en
    assert 'Which assumptions fail' in packet.answer_question_en
    assert '0.95 confidence?' in packet.answer_question_en
    assert 'not calibrated' in packet.background_en
    assert packet.raw_question_en == raw and packet.question_en == raw
    assert confirmed_question_boundary(raw) is None


def test_quoted_self_question_and_relative_clause_do_not_become_followup_focus():
    raw = ('How would you validate the reconstruction? '
           'The notes say "use a sphere, or how do you estimate reflectance?" '
           'The system uses one reference, or what the model estimates is uncertain.')
    focus, background = explicit_question_focus(raw)
    assert focus == 'How would you validate the reconstruction?'
    assert 'how do you estimate reflectance?' in background
    assert 'what the model estimates is uncertain' in background


def test_or_comment_after_a_question_is_background_rather_than_an_extra_ask():
    raw = ('How would you validate the reconstruction? '
           'Or maybe the comparison is already in your backup slides. '
           'Or fixing this is impossible without more data.')
    focus, background = explicit_question_focus(raw)
    assert focus == 'How would you validate the reconstruction?'
    assert 'backup slides' in background
    assert 'fixing this is impossible' in background


def test_fallback_keeps_full_unrecognized_request_without_a_length_cap():
    source = ('An unfamiliar elliptical formulation ' * 90).strip() + '.'
    packet = QuestionContext().build(source)
    assert packet.question_en == source
    assert packet.answer_question_en == source
    assert packet.raw_question_en == source

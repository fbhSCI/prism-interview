"""Constructed spoken-request regressions, not claimed real-ASR measurements.

Actual ASR/timer fixtures remain covered by test_continuous_questions_035 and
test_question_routing_036. These new sentences target confirmed text routing.
"""
from dataclasses import replace
import time

import pytest

from test_controller_v2 import controller
from zoom_copilot.question_packet import QuestionContext
from zoom_copilot.turns import has_complete_direct_question, confirmed_question_boundary


def qa(c):
    c.pref.caption_speaker_mode = 'single_remote'
    c.pref.auto_answer = True
    c._listen_pref = replace(c.pref)
    c.change_interview_phase('qa')
    c.listening = True


def speaking(c, text):
    c.assembler.voice()
    c.on_transcript('final', text)
    c.assembler.voice()
    c.auto_answer()


@pytest.mark.parametrize('question', [
    'I was wondering whether your method can recover circular polarization.',
    'I wonder how you would calibrate the sensing operator.',
    "I'm curious about your uncertainty estimates under scattering.",
    'I am curious whether the inverse problem remains identifiable.',
    'I would be interested in your comparison with conventional polarimetric imaging.',
    'I wanted to ask how your reconstruction handles unknown illumination.',
    "I'd like to ask whether you can recover the full Stokes vector.",
    'Any thoughts on handling multiple scattering?',
    'Any comments on the amount of calibration data required?',
    'What about scattering?',
    'How about calibration?',
    'Does it generalize?',
])
def test_spoken_request_seals_at_confirmed_response_without_global_pause(controller, question):
    c = controller; qa(c)
    speaking(c, question)
    assert not c.busy, 'Question punctuation alone must not dispatch during continuous speech.'
    speaking(c, 'Sure. We use independently calibrated reference measurements.')
    assert c.busy and not c._requests
    assert c._question_raw_by_request[c._active] == question
    assert c._question_packets[c._active].answer_question_en == question
    speaking(c, 'This also lets us evaluate reconstruction uncertainty.')
    assert not c._cancel.is_set() and not c._requests
    c.test_run_worker()
    assert len(c.service.calls) == 1


@pytest.mark.parametrize('question', [
    'I wonder whether your method can recover circular polarization.',
    'Any thoughts on the calibration procedure.',
    'Any thoughts on the calibration procedure',
])
def test_period_ended_spoken_request_can_dispatch_after_real_pause(controller, question):
    c = controller; qa(c)
    speaking(c, question)
    c.assembler.last_voice = c.assembler.last_text = time.monotonic() - 2
    c.auto_answer()
    assert c.busy and c._question_raw_by_request[c._active] == question


def test_direct_indirect_and_short_followup_all_reach_model_checklist(controller):
    c = controller; qa(c)
    setup = 'Assume that the scene is dynamic and the illumination is unknown.'
    questions = ['How would you calibrate M?',
                 'I am curious about your uncertainty estimates.',
                 'Any thoughts on validating recovered material reflectance?']
    for fragment in [setup, *questions]:
        speaking(c, fragment)
        assert not c.busy
    speaking(c, 'Sure.')
    assert c.busy
    packet = c._question_packets[c._active]
    assert packet.subquestions == tuple(questions)
    assert packet.answer_question_en == ' '.join(questions)
    assert setup in packet.background_en
    assert packet.raw_question_en == ' '.join([setup, *questions])


def test_compound_spoken_request_keeps_second_indirect_subquestion():
    raw = ('How would you calibrate the sensing operator and I was wondering whether '
           'your validation also covers circular polarization?')
    packet = QuestionContext().build(raw)
    assert len(packet.subquestions) == 2
    assert 'How would you calibrate the sensing operator' == packet.subquestions[0]
    assert 'I was wondering whether your validation also covers circular polarization?' in packet.subquestions[1]
    assert packet.raw_question_en == raw


@pytest.mark.parametrize('question', [
    'I was wondering how your method handles dynamic scenes.',
    'Any thoughts on low photon counts?',
    'What about multiple scattering?',
])
def test_explicit_spoken_followup_after_previous_reply_becomes_next_question(controller, question):
    c = controller; qa(c)
    speaking(c, 'How would you calibrate the sensing operator? Sure. We use reference targets.')
    first = c._active
    speaking(c, question)
    assert c._active == first and not c._cancel.is_set()
    speaking(c, 'Sure.')
    assert len(c._requests) == 1
    assert c._question_raw_by_request[c._requests[0][0]] == question


@pytest.mark.parametrize('fragment', [
    'I was wondering whether', 'I am curious about', 'Any thoughts on',
    'What about', 'How do you',
])
def test_unfinished_spoken_request_does_not_dispatch_on_backchannel(controller, fragment):
    c = controller; qa(c)
    speaking(c, fragment)
    speaking(c, 'Sure.')
    c.assembler.last_voice = c.assembler.last_text = time.monotonic() - 2
    c.auto_answer()
    assert not c.busy and not c._requests


def test_rejected_middle_in_indirect_request_remains_visible_and_blocks(controller):
    c = controller; qa(c)
    speaking(c, 'I was wondering whether your method can recover')
    c.on_transcript('rejected', '')
    speaking(c, 'circular polarization. Sure.')
    c.assembler.last_voice = c.assembler.last_text = time.monotonic() - 2
    c.auto_answer()
    assert c._question_uncertain and not c.busy and not c._requests
    assert 'circular polarization' in c._last_question_display


def test_quoted_self_question_inside_answer_is_not_a_new_request(controller):
    c = controller; qa(c)
    speaking(c, 'How would you calibrate the sensing operator? Sure.')
    first = c._active
    speaking(c, 'In our experiments we ask ourselves: what about multiple scattering? '
                'We then measure it using a reference target.')
    speaking(c, 'Sure.')
    assert c._active == first and not c._cancel.is_set() and not c._requests


@pytest.mark.parametrize('reflection', [
    'I wonder whether the albedo model itself explains this effect.',
    'I am curious about the cause of this reconstruction artifact.',
])
def test_undirected_self_reflection_during_reply_does_not_restart_question(controller, reflection):
    c = controller; qa(c)
    speaking(c, 'How would you calibrate the sensing operator? Sure.')
    first = c._active
    speaking(c, reflection)
    speaking(c, 'Sure.')
    c.assembler.last_voice = c.assembler.last_text = time.monotonic() - 2
    c.auto_answer()
    assert c._active == first and not c._cancel.is_set() and not c._requests


@pytest.mark.parametrize('text', [
    'Any further questions about reconstruction?', 'I wonder.',
    'I am curious.', 'These results are interesting, right?',
])
def test_invitation_and_statement_are_not_complete_technical_requests(text):
    assert not has_complete_direct_question(text)


@pytest.mark.parametrize('question', [
    'I was wondering whether your method can recover circular polarization.',
    "I'm curious about your uncertainty estimates under scattering.",
    'Any thoughts on handling multiple scattering?',
    'What about scattering?', 'How about calibration?', 'Does it generalize?',
])
def test_confirmed_request_grammar_and_packet_preserve_source(question):
    assert has_complete_direct_question(question)
    assert confirmed_question_boundary(question + ' Sure.') == len(question) + 1
    packet = QuestionContext().build(question)
    assert packet.answer_question_en == question
    assert packet.subquestions == (question,)
    assert packet.raw_question_en == question


@pytest.mark.parametrize('fragment', [
    'I was wondering whether', 'I am curious about', 'Any thoughts on',
    'What about', 'How do you',
])
def test_guessed_question_mark_on_request_prefix_cannot_trigger(fragment):
    assert not has_complete_direct_question(fragment + '?')
    assert confirmed_question_boundary(fragment + '? Sure.') is None


@pytest.mark.parametrize('fragment', ['Do you think?', 'Do you think it?', 'Could you tell me?'])
def test_short_unfinished_opening_is_not_sealed_by_a_continuous_backchannel(controller, fragment):
    c = controller; qa(c)
    speaking(c, fragment)
    speaking(c, 'Sure.')
    assert not c.busy and not c._requests and not c._semantic_response

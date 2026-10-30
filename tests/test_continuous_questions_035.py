"""Continuous speech boundaries: confirmed real ASR plus adversarial turn cases.

No model, cloud, devices, or credentials are used by these controller tests.
The fixture records real Whisper output; it is not an ASR accuracy benchmark.
"""
from dataclasses import replace
import json
from pathlib import Path
from public_fixtures import read_optional_recording
from types import SimpleNamespace
import time

import pytest

from test_controller_v2 import controller
from zoom_copilot.question_packet import QuestionContext
from zoom_copilot.turns import confirmed_question_boundary


def qa(c):
    c.pref.caption_speaker_mode = 'single_remote'
    c._listen_pref = replace(c.pref)
    c.change_interview_phase('qa')
    c.listening = True


def speaking(c, text, context=None):
    c.assembler.voice()
    c.on_transcript('final', text, context)
    c.assembler.voice()  # No artificial globally silent endpoint.
    c.auto_answer()


def test_real_confirmed_asr_dispatches_during_continuous_reply_without_pause(controller):
    c = controller; qa(c)
    fixture = json.loads(read_optional_recording((Path(__file__).parent / 'fixtures' /
                          'tum_continuous_confirmed_035.json')))
    c.speech = SimpleNamespace(pending=3, close=lambda: None)
    dispatched_at = None
    for event in fixture['events']:
        if event['kind'] == 'rejected':
            c.on_transcript('rejected', event['text'], event['context'])
        else:
            speaking(c, event['text'], event['context'])
        if c.busy and dispatched_at is None:
            dispatched_at = event['video_seconds']
    c.speech = None
    assert dispatched_at is not None and 347 < dispatched_at < 354
    assert c.busy and not c._cancel.is_set() and not c._requests
    assert c.last_answer_timing['answer_trigger'] == 'semantic_boundary'
    assert 'answer_endpoint_ms' not in c.last_answer_timing
    packet = c._question_packets[c._active]
    assert packet.raw_question_en.startswith('Alex, one question')
    assert 'do you think that is a good alternative' in packet.raw_question_en
    assert packet.raw_question_en.endswith('have a network hallucinate the video.')
    assert 'rambling answer' not in packet.raw_question_en
    assert 'Sure.' not in packet.raw_question_en and 'back propagated' not in packet.raw_question_en
    assert packet.answer_question_en.casefold().startswith('do you think')
    assert 'motion integration' in packet.background_en
    assert all('Sure.' not in turn.raw_en for turn in packet.turns)
    c.test_run_worker()
    assert len(c.service.calls) == 1
    # Prompt budgeting may omit duplicated background metadata, but never the
    # complete authoritative current question passed to the answer service.
    assert 'motion integration' in c.service.calls[0]['question']
    assert c.service.calls[0]['interview_packet']['answer_question_en'].casefold().startswith('do you think')


def test_question_mark_waits_for_confirmed_complement_and_response(controller):
    c = controller; qa(c)
    speaking(c, 'Do you think we should ignore motion modeling and have a network hallucinate?')
    assert not c.busy
    speaking(c, 'The video?')
    assert not c.busy
    speaking(c, "Sure. It's a great question.")
    assert c.busy
    packet = c._question_packets[c._active]
    assert 'The video?' in packet.raw_question_en
    assert 'Sure' not in packet.raw_question_en
    assert 'The video?' in packet.answer_question_en


@pytest.mark.parametrize('setup', [
    'This is an end-to-end system, right?',
    'The model learns from data. Yep.',
    'We can integrate the motion field, okay?',
])
def test_confirmation_tags_or_background_alone_do_not_trigger(controller, setup):
    c = controller; qa(c)
    speaking(c, setup)
    speaking(c, 'Sure.')
    assert not c.busy and not c._requests


def test_compound_asks_and_late_qualifier_are_kept_until_reply(controller):
    c = controller; qa(c)
    for fragment in ('How would you calibrate the instrument?',
                     'And how would you validate uncertainty?',
                     'Especially under strong scattering.', 'Sure.'):
        speaking(c, fragment)
    packet = c._question_packets[c._active]
    assert 'calibrate' in packet.raw_question_en
    assert 'validate uncertainty' in packet.raw_question_en
    assert 'strong scattering' in packet.raw_question_en
    assert len(packet.subquestions) == 2


def test_frozen_questions_are_not_cancelled_or_removed_from_queue(controller):
    c = controller; qa(c)
    asks = [f'How would you validate condition {number}?' for number in (1, 2, 3)]
    for ask in asks:
        speaking(c, ask + ' Sure. We used controlled reference targets.')
    assert c.busy and not c._cancel.is_set()
    assert len(c._requests) == 2
    speaking(c, 'The calibration then proceeds with held-out observations.')
    assert len(c._requests) == 2 and not c._cancel.is_set()
    for _ in asks:
        c.test_run_worker(); c.test_run_callbacks()
    assert [entry['raw_question'] for entry in c.service.calls] == asks
    assert len(c._answers) == 3


def test_named_question_introduction_retires_old_background_without_topic_guess(controller):
    c = controller; qa(c)
    speaking(c, 'Would you favor reconstruction or neural rendering? Sure.')
    first = c._active
    speaking(c, 'That concludes my response. Robin, one question about your latest work.')
    speaking(c, 'The method combines learned priors with classical motion integration.')
    speaking(c, 'Do you think an entirely learned video model is a good alternative? Sure.')
    assert c._active == first and not c._cancel.is_set()
    assert len(c._requests) == 1
    packet = c._question_packets[c._requests[0][0]]
    assert packet.raw_question_en.startswith('Robin, one question')
    assert 'reconstruction or neural rendering' not in packet.raw_question_en
    assert 'concludes my response' not in packet.raw_question_en
    assert 'motion integration' in packet.raw_question_en


def test_uncertain_middle_is_not_silently_accepted_by_response_boundary(controller):
    c = controller; qa(c)
    speaking(c, 'How would you estimate reflectance?')
    c.on_transcript('rejected', '')
    speaking(c, 'Sure. We use reference materials.')
    assert not c.busy and not c._requests
    assert c._question_uncertain


@pytest.mark.parametrize('previous_text', ['', 'That concludes the previous answer.'])
def test_previous_rejection_does_not_block_explicit_next_question(controller, previous_text):
    c = controller; qa(c)
    if previous_text:
        speaking(c, previous_text)
    c.on_transcript('rejected', '')
    assert c._question_uncertain
    speaking(c, 'Robin, one question about your latest work.')
    assert not c._question_uncertain
    speaking(c, 'Do you think an entirely learned video model is a good alternative? Sure.')
    assert c.busy and not c._cancel.is_set()
    raw = c._question_raw_by_request[c._active]
    assert raw.startswith('Robin, one question')
    assert 'previous answer' not in raw


def test_rejection_inside_new_explicit_question_remains_blocked(controller):
    c = controller; qa(c)
    speaking(c, 'Robin, one question about your latest work.')
    speaking(c, 'Do you think we should ignore motion modeling?')
    c.on_transcript('rejected', '')
    speaking(c, 'The video? Sure.')
    assert c._question_uncertain and not c.busy


def test_partial_response_never_confirms_question_boundary(controller):
    c = controller; qa(c)
    speaking(c, 'How would you validate the inverse problem?')
    c.on_transcript('partial', 'Sure.')
    c.assembler.voice()
    c.auto_answer()
    assert not c.busy


def test_addressed_ask_after_comma_keeps_exact_source_and_background():
    text = ('We could train the network end-to-end, do you think it is a good alternative '
            'to ignore motion modeling and generate the video.')
    packet = QuestionContext().build(text)
    assert packet.question_en == text
    assert packet.raw_question_en == text
    assert packet.answer_question_en.startswith('do you think')
    assert packet.background_en == 'We could train the network end-to-end,'
    assert confirmed_question_boundary(text + ' Sure.') == len(text) + 1


def test_new_question_inside_same_asr_crop_is_not_lost(controller):
    c = controller; qa(c)
    speaking(c, 'How would you estimate uncertainty? Sure. We use calibration targets. '
               'Robin, another question about validation. How do you measure generalization? Sure.')
    assert not c._cancel.is_set() and len(c._requests) == 1
    assert 'estimate uncertainty' in c._question_raw_by_request[c._active]
    next_raw = c._question_raw_by_request[c._requests[0][0]]
    assert next_raw.startswith('Robin, another question')
    assert 'estimate uncertainty' not in next_raw


@pytest.mark.parametrize('correction', [
    ['No, I mean under unknown illumination.'],
    ['No.', 'I mean under unknown illumination.'],
    ['Sorry, I meant under unknown illumination.'],
    ['Let me rephrase that using unknown illumination.'],
])
def test_self_correction_is_retained_before_question_closure(controller, correction):
    c = controller; qa(c)
    speaking(c, 'Do you estimate reflectance under controlled illumination?')
    for text in correction:
        speaking(c, text)
        assert not c.busy and not c._semantic_response
    speaking(c, 'Sure.')
    assert c.busy
    assert 'unknown illumination' in c._question_raw_by_request[c._active]


def test_same_crop_question_self_correction_does_not_trigger(controller):
    c = controller; qa(c)
    speaking(c, 'Do you assume known illumination? No, I mean unknown illumination.')
    assert not c.busy and not c._semantic_response


@pytest.mark.parametrize('qualifier', [
    'We use controlled lighting, so please compare that condition.',
    'I used calibrated targets in this example, so assume that calibration.',
    'I would start by assuming known illumination for this question.',
])
def test_remote_interviewer_first_person_background_is_kept(controller, qualifier):
    c = controller; qa(c)
    speaking(c, 'How would you estimate the material reflectance?')
    speaking(c, qualifier)
    assert not c.busy and not c._semantic_response
    speaking(c, 'Sure.')
    assert qualifier in c._question_raw_by_request[c._active]


def test_real_two_question_asr_preserves_both_without_reply_or_topic_pollution(controller):
    c = controller; qa(c)
    fixture = json.loads(read_optional_recording((Path(__file__).parent / 'fixtures' /
                          'tum_two_questions_confirmed_035.json')))
    for event in fixture['events']:
        if event['kind'] == 'rejected':
            c.on_transcript('rejected', event['text'], event['context'])
        else:
            speaking(c, event['text'], event['context'])
    assert c.busy and not c._cancel.is_set() and len(c._requests) == 1
    first = c._question_raw_by_request[c._active]
    second = c._question_raw_by_request[c._requests[0][0]]
    assert 'existing reconstruction pipelines' in first and 'neural rendering' in first
    assert 'great question' not in first and 'late to the neural network game' not in first
    assert second.startswith('Alex, one question')
    assert 'existing reconstruction pipelines' not in second
    assert 'ignore the motion modeling' in second and 'hallucinate the video' in second
    assert 'Sure.' not in second and 'back propagated' not in second
    c.test_run_worker(); c._next_answer(); c.test_run_worker()
    assert len(c.service.calls) == 2 and len(c._answers) == 2


@pytest.mark.parametrize('complete_immediately', [False, True])
def test_actual_timer_and_final_event_replay_does_not_duplicate_question(controller, complete_immediately):
    c = controller; qa(c)
    fixture = json.loads(read_optional_recording((Path(__file__).parent / 'fixtures' /
                          'tum_two_questions_timed_035.json')))
    c.speech = SimpleNamespace(pending=0, close=lambda: None)
    for event in fixture['events']:
        if event['kind'] == 'timer':
            now = time.monotonic()
            c.assembler.last_voice = now - event['voice_silence_ms'] / 1000
            c.assembler.last_text = now - event['postcommit_ms'] / 1000
            c.speech.pending = event['asr_pending']
            c._partial_id = 'recorded-unconfirmed-tail' if event['partial'] else None
            c.auto_answer()
        else:
            c.assembler.voice()
            c.on_transcript(event['kind'], event['text'], event['context'])
        if complete_immediately and c.busy:
            c.test_run_worker(); c._next_answer()
    c.speech = None
    if not complete_immediately:
        assert c.busy and not c._cancel.is_set() and len(c._requests) == 1
        c.test_run_worker(); c._next_answer(); c.test_run_worker()
    assert len(c.service.calls) == 2
    assert 'reconstruction pipelines' in c.service.calls[0]['raw_question']
    assert c.service.calls[1]['raw_question'].startswith('Alex, one question')


@pytest.mark.parametrize('completed', [False, True])
def test_silence_request_then_same_confirmed_response_boundary_is_not_sent_twice(controller, completed):
    c = controller; qa(c)
    speaking(c, 'How would you calibrate the instrument?')
    c.assembler.last_voice = c.assembler.last_text = time.monotonic() - 2
    c.auto_answer()
    if completed:
        c.test_run_worker()
    speaking(c, "Oh, that's that's a great question. I")
    assert not c._requests
    if not completed:
        assert not c._cancel.is_set()
        c.test_run_worker()
    assert len(c.service.calls) == 1


def test_identical_question_in_explicit_new_turn_is_still_answered(controller):
    c = controller; qa(c)
    ask = 'How would you calibrate the instrument?'
    speaking(c, ask + ' Sure.')
    speaking(c, ask + ' Sure.')
    assert len(c._requests) == 1 and not c._cancel.is_set()
    c.test_run_worker(); c._next_answer(); c.test_run_worker()
    assert [call['raw_question'] for call in c.service.calls] == [ask, ask]


def test_substantive_qualifier_after_silence_still_requests_revision(controller):
    c = controller; qa(c)
    speaking(c, 'How would you calibrate the instrument at 5 degrees?')
    c.assembler.last_voice = c.assembler.last_text = time.monotonic() - 2
    c.auto_answer()
    speaking(c, 'I mean minus 5 degrees, not 5 degrees.')
    speaking(c, 'Sure.')
    assert c._cancel.is_set() and len(c._requests) == 1
    assert 'minus 5 degrees, not 5 degrees' in c._question_raw_by_request[c._requests[0][0]]

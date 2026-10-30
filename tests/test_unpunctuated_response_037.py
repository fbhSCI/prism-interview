"""Real recorded controller events plus constructed boundary safety cases.

The public audio fixture preserves all recorded final/rejected/timer events;
replaying it is not a fresh audio, ASR, translation or model-accuracy test.
"""
import json
from pathlib import Path
from public_fixtures import read_optional_recording
from types import SimpleNamespace
import time

import pytest

from test_controller_v2 import controller
from test_question_recall_037 import qa, speaking
from zoom_copilot.turns import confirmed_question_boundary, has_complete_direct_question


@pytest.mark.parametrize('complete_immediately', [False, True])
def test_real_indirect_question_seals_before_continuous_greeting_and_answer(controller, complete_immediately):
    c = controller; qa(c)
    fixture = json.loads(read_optional_recording((Path(__file__).parent / 'fixtures' /
                          'svsiggraph_q1_timed_037.json')))
    c.speech = SimpleNamespace(pending=0, close=lambda: None)
    first_dispatch = None
    packet = None
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
        if c.busy and first_dispatch is None:
            first_dispatch = event
            packet = c._question_packets[c._active]
        if complete_immediately and c.busy:
            c.test_run_worker(); c._next_answer()
    c.speech = None
    assert first_dispatch is not None
    assert first_dispatch['input_still_playing']
    assert 30 < first_dispatch['replay_seconds'] < 40
    if not complete_immediately:
        assert c.busy and not c._cancel.is_set() and not c._requests
        c.test_run_worker()
    assert len(c.service.calls) == 1
    raw = c.service.calls[0]['raw_question']
    assert 'I was just wondering if' in raw
    assert 'skin and eyes' in raw and 'anisotropic' in raw
    assert 'flash photography' in raw and raw.endswith('our human bodies')
    assert 'hi Ngan' not in raw and 'specialized capture setups' not in raw
    assert packet is not None
    assert 'I was just wondering if' in packet.answer_question_en
    assert 'So I wonder if' in packet.answer_question_en
    assert all('hi Ngan' not in turn.raw_en for turn in packet.turns)
    assert c._semantic_response


@pytest.mark.parametrize('opening', [
    'I was just wondering if', 'I was actually wondering whether',
    'I was kind of wondering how', 'I was just kind of wondering whether',
    'I just wonder whether', 'I was just wondering like if',
])
def test_filled_indirect_request_with_unpunctuated_social_response(controller, opening):
    c = controller; qa(c)
    question = opening + ' your method can reconstruct moving objects'
    speaking(c, question + ' yes')
    assert not c.busy, 'A bare yes cannot seal an unpunctuated request.'
    speaking(c, "yeah hello Alex it's nice to see you")
    assert c.busy and c._semantic_response
    assert c._question_raw_by_request[c._active] == question
    assert has_complete_direct_question(question, require_terminal=False)


def test_unpunctuated_direct_request_can_use_strong_greeting_boundary(controller):
    c = controller; qa(c)
    question = 'How would you estimate the uncertainty in moving objects'
    speaking(c, question + " yes hi Alex it is good to see you")
    assert c.busy and c._question_raw_by_request[c._active] == question


@pytest.mark.parametrize('fragment', [
    'I was just wondering whether', 'I was actually wondering if your',
    'I was kind of wondering how you would', 'Could you tell me',
    'Do you think', 'Do you think it',
])
def test_strong_greeting_does_not_make_an_incomplete_request_complete(controller, fragment):
    c = controller; qa(c)
    speaking(c, fragment + " yes hi Alex it's good to see you")
    assert not c.busy and not c._semantic_response and not c._requests


@pytest.mark.parametrize('text', [
    "This method estimates uncertainty yes hi Alex it's good to see you",
    'How should the interface transcribe "yes hi Alex it is good to see you"?',
    "I was just wondering whether your method can recover polarization yes I mean circular polarization",
    "I was just wondering whether your method can recover polarization yes hi Alex",
])
def test_social_boundary_requires_real_request_and_complete_unquoted_greeting(text):
    assert confirmed_question_boundary(text) is None


def test_partial_greeting_cannot_confirm_a_response(controller):
    c = controller; qa(c)
    speaking(c, 'I was just wondering whether your method can recover polarization')
    c.on_transcript('partial', "yes hi Alex it's good to see you")
    c.assembler.voice(); c.auto_answer()
    assert not c.busy and not c._semantic_response


def test_rejected_middle_blocks_even_a_complete_unpunctuated_greeting(controller):
    c = controller; qa(c)
    speaking(c, 'I was just wondering whether your method can recover')
    c.on_transcript('rejected', '')
    speaking(c, "circular polarization yes hi Alex it's good to see you")
    assert c._question_uncertain and not c.busy and not c._semantic_response


def test_reply_reflection_after_social_boundary_does_not_restart_question(controller):
    c = controller; qa(c)
    speaking(c, "I was just wondering whether your method handles low light yes hi Alex it's good to see you")
    first = c._active
    speaking(c, 'For this direction it is like how can we extract supervision from the data we do have.')
    speaking(c, 'I was actually wondering if the noise explains this phenomenon.')
    speaking(c, 'Sure.')
    assert c._active == first and not c._cancel.is_set() and not c._requests

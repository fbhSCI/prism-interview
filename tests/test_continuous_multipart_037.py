"""Unedited full real-event replay and constructed reply/ask ambiguity cases."""
import json
from pathlib import Path
from public_fixtures import read_optional_recording
from types import SimpleNamespace
import time

import pytest

from test_controller_v2 import controller
from test_question_recall_037 import qa, speaking
from zoom_copilot.turns import confirmed_question_boundary


@pytest.mark.parametrize('complete_immediately', [False, True])
def test_full_360s_replay_has_two_complete_questions_and_no_answer_self_question(controller, complete_immediately):
    c = controller; qa(c)
    fixture = json.loads(read_optional_recording((Path(__file__).parent / 'fixtures' /
                          'svsiggraph_full_timed_037.json')))
    c.speech = SimpleNamespace(pending=0, close=lambda: None)
    dispatched = {}
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
        for identity, raw in c._question_raw_by_request.items():
            if identity not in dispatched:
                dispatched[identity] = {'event': event, 'raw': raw}
        if complete_immediately and c.busy:
            c.test_run_worker(); c._next_answer()
    c.speech = None
    assert len(dispatched) == 2, list(dispatched.values())
    first, second = list(dispatched.values())
    assert 30 < first['event']['replay_seconds'] < 40
    assert 310 < second['event']['replay_seconds'] < 326
    assert first['event']['input_still_playing'] and second['event']['input_still_playing']
    assert first['raw'].endswith('our human bodies')
    assert "I've got a question about" in second['raw']
    assert 'initial albedo by low' in second['raw'] and '-pass filtering' in second['raw']
    assert 'other information?' in second['raw'] and 'Diffuse illumination' in second['raw']
    assert 'segmenting' in second['raw'] and second['raw'].endswith('information do you use')
    assert 'honestly' not in second['raw'] and 'what we have' not in second['raw']
    if not complete_immediately:
        assert not c._cancel.is_set()
        while c.busy:
            c.test_run_worker(); c._next_answer()
    assert len(c.service.calls) == 2


@pytest.mark.parametrize('statement', [
    'And so what you actually care about most is the quality of the output.',
    'Then when you go from that initial state to the next state the operator changes.',
    'How you calibrate the instrument affects the result.',
    'Where you place the source is important.',
    'What the algorithm produces is your initial estimate.',
    'What your measurements tell us is that the target moved.',
])
def test_declarative_wh_clause_in_an_answer_does_not_become_a_new_question(controller, statement):
    c = controller; qa(c)
    speaking(c, 'How would you calibrate the instrument? Sure.')
    first = c._active
    speaking(c, statement)
    speaking(c, 'Yeah, for sure.')
    assert c._active == first and c._semantic_response
    assert not c._requests and not c._cancel.is_set()


@pytest.mark.parametrize('question', [
    'What do you actually measure?', 'How would you calibrate a new device?',
    'Which measurements would you choose?', 'What information do you use?',
])
def test_inverted_addressed_question_still_reopens_after_reply(controller, question):
    c = controller; qa(c)
    speaking(c, 'How would you calibrate the instrument? Sure.')
    speaking(c, question)
    speaking(c, 'Sure.')
    assert len(c._requests) == 1 and not c._cancel.is_set()
    assert c._question_raw_by_request[c._requests[0][0]] == question


@pytest.mark.parametrize('intro', [
    "Yes, I've got a question about", 'Yeah, I have got a question about',
    'I have a quick question about', 'I got a question about',
    "I've got a related question about",
])
def test_explicit_new_question_intro_keeps_setup_before_the_actual_ask(controller, intro):
    c = controller; qa(c)
    speaking(c, 'How would you calibrate the instrument? Sure.')
    first = c._active
    speaking(c, intro)
    speaking(c, 'the initial estimate. Yeah.')
    speaking(c, 'You start by filtering the measured observations.')
    assert c._active == first and not c._requests
    speaking(c, 'Which other measurements do you use in the next step? Sure.')
    assert len(c._requests) == 1 and not c._cancel.is_set()
    raw = c._question_raw_by_request[c._requests[0][0]]
    assert raw.startswith(intro) and 'filtering the measured observations' in raw
    assert 'calibrate the instrument' not in raw


@pytest.mark.parametrize('reply', [
    'this honestly so at this point what we have is a measured initial estimate.',
    'that actually in our implementation we already use independent observations.',
    'this basically for our experiments I normally measure the initial response.',
])
def test_tentative_stance_marker_requires_later_explanatory_clause(controller, reply):
    c = controller; qa(c)
    question = 'What additional information do you use'
    marker, explanation = reply.split(' ', 2)[:2], reply.split(' ', 2)[2]
    beginning = ' '.join(marker)
    speaking(c, question + ' ' + beginning)
    assert not c.busy and not c._semantic_response
    speaking(c, explanation)
    assert c.busy and c._question_raw_by_request[c._active] == question


@pytest.mark.parametrize('text', [
    'What additional information do you use honestly I normally measure the initial response.',
    'What additional information do you use this honestly if we already have measurements?',
    'What additional information do you use this honestly I mean the measured response? We have data.',
    'What kind of observations make this actually feasible when we measure a moving scene?',
    'What kind of change makes this actually affect how we measure the signal',
    'What additional information do you use "this honestly what we have is data"?',
    'How do you use this actually for calibration?',
])
def test_unconfirmed_or_conditional_stance_is_not_a_reply_boundary(text):
    assert confirmed_question_boundary(text) is None


def test_rejected_middle_blocks_backtracked_explanatory_response(controller):
    c = controller; qa(c)
    speaking(c, 'What additional information do you use')
    c.on_transcript('rejected', '')
    speaking(c, 'this honestly so at this point what we have is a measured estimate.')
    assert c._question_uncertain and not c.busy and not c._semantic_response


@pytest.mark.parametrize('pieces', [
    ["Yes, I've got a", 'question about'],
    ['I have', 'got a related', 'question about'],
    ['Yeah, I', 'have a quick', 'question about'],
])
def test_question_introduction_across_confirmed_crops_retains_the_full_new_setup(controller, pieces):
    c = controller; qa(c)
    speaking(c, 'How would you calibrate the instrument? Sure.')
    for piece in pieces:
        speaking(c, piece)
    speaking(c, 'your initialization. You start from measured observations.')
    speaking(c, 'What additional information do you use? Sure.')
    assert len(c._requests) == 1
    raw = c._question_raw_by_request[c._requests[0][0]]
    assert raw.startswith(' '.join(pieces))
    assert 'initialization' in raw and 'measured observations' in raw
    assert 'calibrate the instrument' not in raw


def test_incomplete_intro_during_answer_is_not_reused_after_unrelated_speech(controller):
    c = controller; qa(c)
    speaking(c, 'How would you calibrate the instrument? Sure.')
    speaking(c, 'I have')
    speaking(c, 'two measured reference targets in these experiments.')
    speaking(c, 'another question about this could be interesting to study.')
    assert not c._requests
    assert not c._question_original.startswith('I have')


def test_rejected_crop_does_not_bridge_a_pending_question_introduction(controller):
    c = controller; qa(c)
    speaking(c, 'How would you calibrate the instrument? Sure.')
    first = c._active
    speaking(c, "Yes, I've got a")
    assert c._question_intro_tail
    c.on_transcript('rejected', '')
    speaking(c, 'question about the initial estimate. Yeah.')
    assert not c._question_intro_tail and c._active == first and not c._requests


def test_rejection_after_confirmed_new_introduction_still_blocks_question(controller):
    c = controller; qa(c)
    speaking(c, 'How would you calibrate the instrument? Sure.')
    speaking(c, "Yes, I've got a question about")
    c.on_transcript('rejected', '')
    speaking(c, 'the initial estimate. What additional information do you use? Sure.')
    assert c._question_uncertain and not c._requests

"""Completed automatic-answer boundaries; no cloud, audio or model execution.

These controller regressions distinguish complete independent asks from late
conditions. Existing recorded text/timer fixtures are not new ASR benchmarks.
"""
import pytest

from test_controller_v2 import controller
from test_question_recall_037 import qa, speaking
from test_question_routing_036 import pause
from test_semantic_latency_039 import test_all_recorded_requests_preserve_questions_and_timing as replay_test


def answered(c):
    speaking(c, 'How would you calibrate the spectral-polarimetric instrument?')
    pause(c)
    assert c.busy
    c.test_run_worker()
    assert not c.busy and len(c._answers) == 1 and not c._semantic_response


@pytest.mark.parametrize('next_ask', [
    'How would you validate reconstruction uncertainty?',
    'I was wondering whether you can estimate circular polarization.',
])
def test_completed_auto_starts_fresh_with_history_and_ledger(controller, next_ask):
    c = controller; qa(c); answered(c)
    old = c._answers[0]['question']; old_answer = c.window._answer_en
    generation = c.assembler.generation
    speaking(c, next_ask)
    assert c.assembler.generation == generation + 1
    assert c._question_original == next_ask and c.assembler.text == next_ask
    assert c.window._answer_en == old_answer
    speaking(c, 'Sure.')
    assert c._question_raw_by_request[c._active] == next_ask
    packet = c._question_packets[c._active]
    assert old in ' '.join(t.raw_en for t in packet.recent_context)
    c.test_run_worker()
    call = c.service.calls[-1]
    assert call['raw_question'] == next_ask
    assert call['history'] == [{'question': old, 'answer_en': 'Suggested English answer.'}]


@pytest.mark.parametrize('continuation', [
    'Especially under unknown illumination.',
    'I mean minus five degrees, not five degrees.',
    'Sorry, I meant an uncalibrated camera.',
    'No, I mean under unknown illumination.',
    'How do we calibrate the sensor?',
    'What should I do with this matrix?',
    'In our method we ask ourselves: how would you calibrate it?',
    'The question is: how would you calibrate it?',
    'How would you',
    'Could you tell me?',
    'And how would you calibrate it?',
    'Also how would you validate uncertainty?',
    'Well, and how would you validate uncertainty?',
    'How would you calibrate this instrument?',
    'How would you validate the same sensor?',
    'Which additional measurements would you acquire?',
    'Could you compare other methods?',
    'For example, how would you calibrate the sensor?',
])
def test_qualifier_self_ask_or_dependent_opening_keeps_prior(controller, continuation):
    c = controller; qa(c); answered(c)
    old = c._question_original; generation = c.assembler.generation
    speaking(c, continuation)
    assert c.assembler.generation == generation and old in c._question_original


@pytest.mark.parametrize('outcome', [
    'cancelled', 'failed', 'new_generation', 'changed_source', 'manual_completed',
])
def test_nonmatching_or_unsuccessful_auto_cannot_start_boundary(controller, outcome):
    c = controller; qa(c)
    if outcome == 'cancelled':
        speaking(c, 'How would you calibrate the instrument?'); pause(c)
        c.cancel_answer(); c.test_run_worker()
    elif outcome == 'failed':
        c.service.failure = 'offline failure'
        speaking(c, 'How would you calibrate the instrument?'); pause(c)
        c.test_run_worker(); c.service.failure = ''
    else:
        answered(c)
        if outcome == 'new_generation': c._begin_semantic_question()
        if outcome == 'changed_source': speaking(c, 'Especially under unknown illumination.')
        if outcome == 'manual_completed':
            c.request_answer('How would you calibrate the instrument manually?'); c.test_run_worker()
    generation = c.assembler.generation
    speaking(c, 'How would you validate uncertainty?')
    assert c.assembler.generation == generation


def test_partial_does_not_reset_completed_question(controller):
    c = controller; qa(c); answered(c)
    old = c._question_original; generation = c.assembler.generation
    c.on_transcript('partial', 'How would you validate uncertainty?')
    assert c._question_original == old and c.assembler.generation == generation


def test_cross_confirmed_crop_question_remains_conservative(controller):
    c = controller; qa(c); answered(c)
    old = c._question_original; generation = c.assembler.generation
    speaking(c, 'How would you')
    speaking(c, 'validate reconstruction uncertainty?')
    assert c.assembler.generation == generation
    assert old in c._question_original and 'validate reconstruction uncertainty' in c._question_original


@pytest.mark.parametrize('name,expected', [
    ('svsiggraph_full_timed_038.json', 2), ('svsiggraph_full_timed_037.json', 2),
    ('svsiggraph_q1_timed_037.json', 1), ('tum_q2_timed_037_final4.json', 1),
    ('tum_two_questions_timed_035.json', 2),
])
@pytest.mark.parametrize('complete_immediately', [False, True])
def test_recorded_timer_regressions(controller, monkeypatch, name, expected, complete_immediately):
    replay_test(controller, monkeypatch, name, expected, complete_immediately)

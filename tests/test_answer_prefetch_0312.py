"""Source-exact hidden Qwen preparation; no devices, credentials or cloud calls."""
from dataclasses import asdict
import json
import time

import pytest

from test_controller_v2 import controller
from test_question_recall_037 import qa, speaking


QUESTION = 'What additional measurements do you use?'


def prepare(c, source=QUESTION):
    c.pref.provider = 'qwen'
    c.pref.answer_model = 'qwen-test-model'
    c.pref.qwen_paid_usage_authorized = True
    c.window.set_settings(asdict(c.pref))
    qa(c)
    speaking(c, source)
    assert not c.busy and not c.service.calls
    candidate = c._answer_prefetch_candidate
    assert candidate
    c._answer_prefetch_candidate = (candidate[0], time.monotonic()-1)
    c.auto_answer()
    assert c._answer_prefetch and not c._answer_prefetch['promoted']
    return c._answer_prefetch


def test_completed_prefetch_is_hidden_until_exact_confirmed_endpoint(controller):
    c = controller
    slot = prepare(c)
    c.test_run_worker()
    assert slot['finished'] and len(c.service.calls) == 1
    assert not c.busy and not c._answers and not c.window._answer_en
    assert c.assembler.text == QUESTION
    speaking(c, 'Sure.')
    c.pump()
    assert len(c._answers) == 1 and not c.busy
    assert len(c.service.calls) == 1
    assert c._answers[0]['result'].request_id == slot['id']
    assert c.last_answer_timing['answer_prefetch_reused'] is True
    assert c._question_original == QUESTION


def test_running_prefetch_promotes_existing_pair_validation_and_finishes_once(controller):
    c = controller
    slot = prepare(c)
    pair = dict(id='pair-1', en='A complete answer.', zh='完整回答。',
                en_complete=True, zh_complete=True, citations=[])
    c.post('answer_prefetch_event', (slot['id'], 'answer_delta',
        (slot['id'], 'answer_pair', json.dumps(pair, ensure_ascii=False))))
    c.pump()
    assert not c.window._answer_en and not c._stream_pairs
    speaking(c, 'Sure.')
    c.pump()
    assert c.busy and slot['promoted']
    assert c._stream_pairs == [pair]
    assert c._cancel is slot['cancel']
    c.test_run_worker()
    assert not c.busy and len(c._answers) == len(c.service.calls) == 1


def test_unpunctuated_question_prefix_is_reused_when_reply_confirms_it(controller):
    c = controller
    raw = 'What additional measurements do you use'
    slot = prepare(c, raw+' this honestly')
    c.test_run_worker()
    speaking(c, 'like the, so at this stage in the pipeline, what we have is a measured estimate.')
    c.pump()
    assert not slot['cancel'].is_set()
    assert not c.busy and len(c._answers) == len(c.service.calls) == 1
    assert c._answers[0]['result'].request_id == slot['id']
    assert c._question_original == raw
    assert c.last_answer_timing['answer_prefetch_reused'] is True


def test_same_questioner_late_qualifier_cancels_hidden_work_and_keeps_full_source(controller):
    c = controller
    opening = QUESTION+' This honestly so at this stage'
    slot = prepare(c, opening)
    qualification = 'I need quantitative results under variable illumination.'
    speaking(c, qualification)
    assert slot['cancel'].is_set() and not c._semantic_response
    assert qualification in c.assembler.text and not c.window._answer_en
    c.test_run_worker()
    c.assembler.last_voice = c.assembler.last_text = time.monotonic()-2
    c.auto_answer()
    c.test_run_worker()
    assert qualification in c.service.calls[-1]['raw_question']
    assert c._answers[-1]['result'].request_id != slot['id']
    assert not c.last_answer_timing.get('answer_prefetch_reused')


def test_manual_request_cancels_hidden_work_without_consuming_its_result(controller):
    c = controller
    slot = prepare(c)
    c.request_answer('Why do you calibrate the sensor?')
    assert slot['cancel'].is_set() and c.busy and c._active != slot['id']
    c.test_run_worker()
    assert not c._answers and c.busy
    c.test_run_worker()
    assert len(c._answers) == 1
    assert c._answers[0]['question'] == 'Why do you calibrate the sensor?'


@pytest.mark.parametrize('action', ['model', 'key', 'consent', 'docs', 'cancel', 'rejected', 'new', 'phase', 'stop'])
def test_lifecycle_changes_cancel_and_never_publish_hidden_answers(controller, action):
    c = controller
    slot = prepare(c)
    if action == 'model':
        c.save_settings({'answer_model': 'different-test-model'})
    elif action == 'key':
        c.save_settings({'api_key': 'test-isolated-replacement'})
    elif action == 'consent':
        c.save_settings({'cloud_consent': False})
    elif action == 'docs':
        c.refresh_docs()
    elif action == 'cancel':
        c.cancel_answer()
    elif action == 'rejected':
        c.on_transcript('rejected', '')
    elif action == 'new':
        c.new_question()
    elif action == 'phase':
        c.change_interview_phase('presentation')
    elif action == 'stop':
        c.stop_listening(drain=False)
    assert slot['cancel'].is_set()
    c.test_run_worker()
    assert not c._answers and not c.window._answer_en
    assert c._answer_prefetch is None


def test_external_library_change_cannot_reuse_buffered_answer(controller):
    c = controller
    slot = prepare(c)
    c.test_run_worker()
    from zoom_copilot.config import app_dir
    (app_dir()/'knowledge.json').write_text('{}', encoding='utf8')
    speaking(c, 'Sure.')
    assert slot['cancel'].is_set()
    c.test_run_worker()
    assert len(c.service.calls) == 2
    assert c._answers[-1]['result'].request_id != slot['id']


def test_retrieval_coverage_is_owned_by_each_hidden_worker(controller):
    c = controller
    sentinel = {'unrelated_active_request': True}
    c.last_retrieval_coverage = sentinel
    prepare(c)
    c.test_run_worker()
    assert c.last_retrieval_coverage is sentinel
    assert c.service.calls[0]['interview_packet']['retrieval_coverage']['selected_passage_count'] == 0


def test_unchanged_timer_polls_do_not_repeat_hidden_request(controller):
    c = controller
    slot = prepare(c)
    c.test_run_worker()
    for _ in range(8):
        c.assembler.voice()
        c.auto_answer()
    assert c._answer_prefetch is slot and len(c.service.calls) == 1
    assert not c.test_work and not c._answers


def test_codex_and_local_models_do_not_start_extra_speculative_calls(controller):
    c = controller
    qa(c)
    speaking(c, QUESTION)
    for _ in range(4):
        c.auto_answer()
    assert not c._answer_prefetch and not c.test_work and not c.service.calls


def test_no_more_than_two_hidden_attempts_per_question(controller):
    c = controller
    prepare(c)
    c.test_run_worker()
    speaking(c, 'How do you calibrate the sensor?')
    key, _ = c._answer_prefetch_candidate
    c._answer_prefetch_candidate = (key, time.monotonic()-1)
    c.auto_answer()
    c.test_run_worker()
    speaking(c, 'How do you validate the reconstruction?')
    for _ in range(5):
        c.assembler.voice()
        c.auto_answer()
    assert c._answer_prefetch_attempts == 2
    assert not c._answer_prefetch and not c.test_work
    assert len(c.service.calls) == 2 and not c._answers
    speaking(c, 'Sure.')
    c.test_run_worker()
    assert len(c._answers) == 1 and len(c.service.calls) == 3
    assert 'calibrate' in c._answers[0]['question']
    assert 'validate' in c._answers[0]['question']


def test_hidden_failure_is_invisible_and_real_endpoint_can_retry(controller):
    c = controller
    prepare(c)
    c.service.failure = 'Synthetic failure before confirmation.'
    c.test_run_worker()
    assert not c._answer_prefetch and not c._answers and not c.busy
    c.service.failure = ''
    speaking(c, 'Sure.')
    c.test_run_worker()
    assert len(c._answers) == 1 and len(c.service.calls) == 2
    assert not c.last_answer_timing.get('answer_prefetch_reused')


def test_simple_coordination_keeps_existing_zero_cloud_reply(controller):
    c = controller
    c.pref.provider = 'qwen'
    c.window.set_settings(asdict(c.pref))
    qa(c)
    speaking(c, 'Do you want to start or do you want me to start?')
    candidate = c._answer_prefetch_candidate
    assert candidate
    c._answer_prefetch_candidate = (candidate[0], time.monotonic()-1)
    c.auto_answer()
    assert not c._answer_prefetch and not c.test_work and not c.service.calls

"""All-request replay timing and adversarial confirmed-reply boundaries.

Recorded final/timer fixtures replay the controller only. They are not a new
audio, translation, cloud or target-laptop benchmark. No event is filtered.
"""
import importlib.util
import json
import os
from pathlib import Path
from public_fixtures import read_optional_recording
from types import SimpleNamespace
import time

import pytest

from test_controller_v2 import controller
from test_question_recall_037 import qa, speaking
from zoom_copilot import app as app_module
from zoom_copilot.turns import confirmed_question_boundary


@pytest.mark.parametrize('name,expected', [
    ('svsiggraph_full_timed_038.json', 2),
    ('svsiggraph_full_timed_037.json', 2),
    ('svsiggraph_q1_timed_037.json', 1),
    ('tum_q2_timed_037_final4.json', 1),
    ('tum_two_questions_timed_035.json', 2),
])
@pytest.mark.parametrize('complete_immediately', [False, True])
def test_all_recorded_requests_preserve_questions_and_timing(controller, monkeypatch, name, expected, complete_immediately):
    c = controller; qa(c)
    baseline = os.getenv('PRISM_BOUNDARY_BASELINE')
    if baseline:
        spec = importlib.util.spec_from_file_location('boundary_baseline', baseline)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        monkeypatch.setattr(app_module, 'confirmed_question_boundary', module.confirmed_question_boundary)
    fixture = json.loads(read_optional_recording((Path(__file__).parent/'fixtures'/name)))
    c.speech = SimpleNamespace(pending=0, close=lambda: None, stop=lambda **kwargs: None)
    event_now = {}
    attempts = []
    request = c.request_answer

    def record_request(question, *args, **kwargs):
        before = set(c._question_raw_by_request)
        request(question, *args, **kwargs)
        for identity in set(c._question_raw_by_request) - before:
            attempts.append({'request_id':identity, 'replay_seconds':event_now.get('replay_seconds'),
                'at':event_now['at'], 'video_seconds':event_now.get('video_seconds'),
                'input_still_playing':event_now.get('input_still_playing'),
                'kind':event_now['kind'], 'endpoint':kwargs.get('endpoint','silence'),
                'raw':c._question_raw_by_request[identity],
                'subquestions':list(c._question_packets[identity].subquestions)})
    c.request_answer = record_request
    for event in fixture['events']:
        event_now = event
        if event['kind'] == 'timer':
            now = time.monotonic()
            c.assembler.last_voice = now-event['voice_silence_ms']/1000
            c.assembler.last_text = now-event['postcommit_ms']/1000
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
        while c.busy:
            c.test_run_worker(); c._next_answer()
    report_dir=os.getenv('PRISM_TIMING_REPORT_DIR')
    if report_dir:
        dest=Path(report_dir); dest.mkdir(parents=True,exist_ok=True)
        (dest/(name.removesuffix('.json')+f'-immediate-{complete_immediately}.json')).write_text(
            json.dumps({'fixture':name,'baseline':bool(baseline),'complete_immediately':complete_immediately,
                'source_event_count':len(fixture['events']), 'all_requests':attempts,
                'service_call_count':len(c.service.calls)},ensure_ascii=False,indent=2),encoding='utf-8')
    assert len(attempts) == expected, attempts
    assert len(c.service.calls) == expected
    assert all(a['input_still_playing'] is not False for a in attempts)
    if name.startswith('svsiggraph_full'):
        second=attempts[1]
        assert "I've got a question about" in second['raw']
        assert 'initial albedo by low' in second['raw'] and '-pass filtering' in second['raw']
        assert 'other information?' in second['raw'] and 'Diffuse illumination' in second['raw']
        assert 'segmenting' in second['raw'] and second['raw'].endswith('information do you use')
        assert all(x not in second['raw'] for x in ['honestly','what we have','at this stage'])
        # A restart-only optimization was rejected: it lost same-questioner
        # constraints. Preserve the measured baseline until real audio proves
        # a safe capture/recognition improvement with unchanged turn semantics.
        assert 319 < second['replay_seconds'] < 320
    if name == 'tum_q2_timed_037_final4.json':
        raw=attempts[0]['raw']
        assert raw.startswith('Alex, one question')
        assert 'neural networks' in raw and 'classical modeling' in raw and 'still images' in raw
        assert 'ignore the motion modeling' in raw and 'hallucinate the video' in raw
        assert 'Sure.' not in raw and 'back propagated' not in raw
        assert attempts[0]['video_seconds'] == pytest.approx(349.922,abs=.001)


@pytest.mark.parametrize('reply', [
    'this honestly like the, so at this stage in the um in',
    'that frankly uh well at that point in the',
    'this basically um so in this stage of the',
    'that actually well at this point',
])
@pytest.mark.parametrize('punctuation', ['', '?'])
def test_restart_frame_alone_keeps_missing_punctuation_question_open(controller, reply, punctuation):
    c=controller; qa(c)
    question='What additional measurements do you use'+punctuation
    speaking(c, question+' '+reply)
    assert not c.busy and not c._semantic_response and not c._requests
    speaking(c, 'the calibration pipeline, what we have is a measured estimate.')
    assert c.busy and c._semantic_response
    assert c._question_raw_by_request[c._active]==question


@pytest.mark.parametrize('continuation', [
    'this honestly',
    'that frankly so at this',
    'this actually for calibration',
    'this honestly so at this stage if the input is uncalibrated',
    'this frankly well at that point unless calibration is available',
    'that actually so at this stage I mean the uncalibrated measurements',
    'this basically well at this point can you also explain the noise model',
    'this honestly so at this stage what other conditions do you assume',
    'this actually so at this stage please compare different methods',
    'that frankly so at this point not the calibrated observations',
    'this honestly so at this stage of your process',
    'this actually the whole system so at this point in the pipeline',
])
def test_stance_or_same_speaker_condition_correction_followup_remains_open(controller, continuation):
    c=controller; qa(c)
    question='What additional information do you use'
    speaking(c, question+' '+continuation)
    assert not c.busy and not c._semantic_response and not c._requests
    assert c.assembler.text.endswith(continuation)


def test_partial_restart_does_not_trigger(controller):
    c=controller; qa(c)
    speaking(c,'What additional information do you use this honestly')
    c.on_transcript('partial','like the, so at this stage in the pipeline')
    c.assembler.voice(); c.auto_answer()
    assert not c.busy and not c._semantic_response


def test_rejected_words_cannot_be_bridged_by_restart(controller):
    c=controller; qa(c)
    speaking(c,'What additional information do you use this honestly')
    c.on_transcript('rejected','')
    speaking(c,'like the, so at this stage in the pipeline')
    assert c._question_uncertain and not c.busy and not c._semantic_response


@pytest.mark.parametrize('text', [
    'How would you interpret "this honestly so at this stage in the pipeline"?',
    'What additional information do you use if this honestly so at this stage matters?',
    'How do you use this actually so at this stage in the',
    'What additional information do you',
])
def test_quotes_conditionals_and_incomplete_prefixes_do_not_create_boundary(text):
    assert confirmed_question_boundary(text) is None


@pytest.mark.parametrize('text', [
    'Which information do you use? This honestly may be too broad, so let me clarify: I mean under spatially varying lighting.',
    'Which assumptions do you use? This actually matters because our sensor is uncalibrated.',
    'What labels do you use, this honestly noisy ground-truth signal or clean synthetic labels?',
])
def test_stance_inside_interviewer_rephrasing_does_not_dispatch(controller, text):
    c=controller; qa(c)
    speaking(c,text)
    assert not c.busy and not c._semantic_response and not c._requests
    assert c.assembler.text==text


def test_negation_in_complete_question_is_preserved_after_confirmed_explanation(controller):
    c=controller; qa(c)
    question='Why do you not use the measured illumination in this reconstruction'
    speaking(c,question+' that frankly uh well at this point in the pipeline')
    assert not c.busy
    speaking(c,'what we have is a measured estimate.')
    assert c.busy and c._question_raw_by_request[c._active]==question


@pytest.mark.parametrize('qualification', [
    'I need quantitative results under variable illumination.',
    'I want quantitative results for a moving scene.',
    'my question concerns the uncalibrated sensor.',
    'under very noisy conditions.',
    'unless the calibration is unavailable.',
    'not the calibration data, but the independent test measurements.',
    'can you also describe how you handle polarization?',
])
@pytest.mark.parametrize('split_before_qualification', [False, True])
def test_interviewer_restart_keeps_constraints_even_when_they_arrive_in_next_crop(controller, qualification, split_before_qualification):
    c=controller; qa(c)
    opening='What additional measurements do you use? This honestly so at this stage'
    if split_before_qualification:
        speaking(c,opening)
        assert not c.busy and not c._semantic_response and not c._requests
        # A qualifier may begin a new ASR crop. It must not be discarded as an
        # answer merely because the preceding crop ended at a discourse frame.
        speaking(c,'and '+qualification)
    else:
        speaking(c,opening+' '+qualification)
    assert not c.busy and not c._semantic_response and not c._requests
    assert qualification in c.assembler.text


def test_unpunctuated_question_restart_does_not_drop_noisy_condition(controller):
    c=controller; qa(c)
    text='What additional measurements do you use this honestly well at this point under very noisy conditions'
    speaking(c,text)
    assert not c.busy and not c._semantic_response and not c._requests
    assert c.assembler.text==text

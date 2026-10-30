"""Actual 420 s TUM recognition slice: split floor invitation is not a question.

The fixture replays confirmed text and recorded timer states, not ASR/audio.
No identity or speaker role is inferred from the default single-remote source.
"""
import json
from pathlib import Path
from public_fixtures import read_optional_recording
from types import SimpleNamespace
import time

import pytest

from test_controller_v2 import controller
from test_question_recall_037 import qa, speaking
from zoom_copilot.turns import invitation_prefix_offset, is_question_invitation


@pytest.mark.parametrize('prefix', ['Do we', 'Do you', 'Does anybody', 'Did anyone'])
def test_incomplete_floor_invitation_can_span_before_have(prefix):
    assert invitation_prefix_offset('Thank you for the presentation. '+prefix) is not None
    assert not is_question_invitation(prefix)
    assert is_question_invitation(prefix+' have questions?')


@pytest.mark.parametrize('tail', [
    'need spectral calibration before reconstruction?',
    'have evidence for the nonlinear response?',
    'have a calibrated Stokes matrix for circular polarization?',
    'model wavelength-dependent attenuation explicitly?',
])
def test_ambiguous_prefix_does_not_remove_substantive_question(controller, tail):
    c=controller;qa(c)
    speaking(c,'Do we')
    speaking(c,tail)
    expected='Do we '+tail
    assert c._question_original==expected
    assert c.assembler.text==expected
    assert not c._awaiting_actual_question


def test_actual_host_praise_split_invitation_waits_for_first_real_question(controller):
    c=controller;qa(c)
    fixture=json.loads(read_optional_recording((Path(__file__).parent/'fixtures/tum_invitation_split_0312.json')))
    c.speech=SimpleNamespace(pending=0,stop=lambda **kwargs:None,close=lambda:None)
    attempts=[];current={}
    request=c.request_answer
    def record(question,*args,**kwargs):
        before=set(c._question_raw_by_request)
        request(question,*args,**kwargs)
        for identity in set(c._question_raw_by_request)-before:
            attempts.append({'raw':c._question_raw_by_request[identity],
                             'local_seconds':current['local_media_seconds']})
    c.request_answer=record
    for event in fixture['events']:
        current=event
        if event['kind']=='timer':
            now=time.monotonic()
            c.assembler.last_voice=now-event['voice_silence_ms']/1000
            c.assembler.last_text=now-event['postcommit_ms']/1000
            c.speech.pending=event['asr_pending']
            c._partial_id='recorded-tail' if event['partial'] else None
            c.auto_answer()
        else:
            c.assembler.voice()
            c.on_transcript(event['kind'],event['text'],event['context'])
        if c.busy:
            c.test_run_worker();c._next_answer()
    c.speech=None
    assert len(attempts)==len(c.service.calls)==1
    # Request was logged 15 ms after its source-final callback in the actual
    # Qt run; confirmed-source replay executes that callback immediately.
    assert attempts[0]['local_seconds']==pytest.approx(120.797,abs=.03)
    assert attempts[0]['raw']==fixture['expected_actual_question_raw']
    assert 'Do we have questions' not in attempts[0]['raw']


def test_actual_split_invitation_preserves_following_multicrop_scientific_question(controller):
    c=controller;qa(c)
    speaking(c,"Thank you, Alex, for the wonderful presentation. Really amazing. Do we")
    speaking(c,'have questions?')
    assert c._awaiting_actual_question and not c.assembler.text and not c.busy
    speaking(c,'How does your sensing operator')
    speaking(c,'account for wavelength-dependent attenuation?')
    assert c._question_original=='How does your sensing operator account for wavelength-dependent attenuation?'
    assert not c.busy

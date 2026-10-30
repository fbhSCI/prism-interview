"""Faster cadence is constrained to the measured decoder/hardware/source."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from zoom_copilot.app import CopilotController
from zoom_copilot.config import Preferences


def options(**changes):
    prefs = Preferences()
    prefs.caption_update_mode = 'responsive'
    for name,value in changes.items():
        setattr(prefs,name,value)
    c=SimpleNamespace(_listen_pref=prefs,speech=SimpleNamespace(actual_device='cuda'))
    return c, CopilotController._capture_confirmation_options(c)


def test_measured_gpu_single_remote_profile_preserves_regular_context():
    c,value=options()
    assert value=={'commit_interval_sec':1.6,'initial_maximum_sec':3.0}
    assert c._listen_pref.caption_window_seconds==3.6
    assert c._listen_pref.compute_type=='int8_float16'


@pytest.mark.parametrize('changes',[
    {'caption_update_mode':'original'}, {'caption_update_mode':'unknown'},
    {'caption_speaker_mode':'acoustic'}, {'whisper_size':'large-v3'},
    {'compute_type':'float16'}, {'compute_type':'int8'},
    {'caption_window_seconds':3.0}, {'caption_window_seconds':3.2},
    {'caption_window_seconds':6.0}, {'caption_window_seconds':12.0},
    {'caption_window_seconds':None}, {'caption_window_seconds':'bad'},
])
def test_other_preferences_keep_original_capture_cadence(changes):
    assert options(**changes)[1]=={}


def test_actual_cpu_fallback_does_not_use_dense_gpu_cadence():
    c,_=options(asr_device='cuda')
    c.speech.actual_device='cpu'
    assert CopilotController._capture_confirmation_options(c)=={}


def test_requested_gpu_after_start_does_not_override_active_capture_snapshot():
    c,_=options(caption_update_mode='original')
    c.pref=replace(c._listen_pref)
    c.pref.caption_update_mode='responsive'
    assert CopilotController._capture_confirmation_options(c)=={}

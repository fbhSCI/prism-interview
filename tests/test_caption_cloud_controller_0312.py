"""Consent revocation reaches queued/in-flight confirmed caption translation."""
from dataclasses import asdict, replace
from types import SimpleNamespace
import threading

from test_controller_v2 import controller


def test_caption_revocation_cancels_worker_and_discards_posted_result(controller):
    c=controller
    c.pref.translation_engine='qwen'
    c.pref.translation_cloud_consent=True
    c.pref.qwen_paid_usage_authorized=True
    c._listen_pref=replace(c.pref)
    cancelled=threading.Event()
    c.translation=SimpleNamespace(stop=cancelled.set)
    c.window.append_transcript('source','Original English.','旧中文。')
    c._caption_sources['source']='Original English.'
    c.save_settings({'translation_cloud_consent':False})
    assert cancelled.is_set() and c.translation is None
    c.post('translation',(c._session,('source','Original English.'),'撤销后不得发布。'))
    c.pump()
    assert '撤销后不得发布' not in c.window._entries['source'].chinese.text()


def test_caption_paid_permission_revocation_stops_worker(controller):
    c=controller
    c.pref.translation_engine='qwen'
    c.pref.translation_cloud_consent=True
    c.pref.qwen_paid_usage_authorized=True
    c.pref.free_quota_confirmed=False
    c._listen_pref=replace(c.pref)
    cancelled=threading.Event()
    c.translation=SimpleNamespace(stop=cancelled.set)
    c.save_settings({'qwen_paid_usage_authorized':False})
    assert cancelled.is_set() and c.translation is None


def test_qwen_start_without_caption_consent_recovers_idle_buttons(controller):
    c=controller
    c.pref.speaker_id='test-output'
    c.pref.translation_engine='qwen'
    c.pref.translation_cloud_consent=False
    c.window.set_settings(asdict(c.pref))
    c.window.device_select.addItem('test-output','test-output')
    c.window.device_select.setCurrentIndex(c.window.device_select.count()-1)
    c.start_listening()
    assert not c.preparing and not c.listening
    assert c.speech is None and c.audio is None and c.translation is None
    assert c.window.start_button.isEnabled()

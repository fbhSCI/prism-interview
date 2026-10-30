"""Preparing local models is a distinct action from recording or cloud consent."""
from zoom_copilot import model_setup
from test_controller_v2 import controller


def test_prepare_button_uses_selected_model_path_without_authorizing_cloud(controller, monkeypatch, tmp_path):
    c=controller
    c.pref.cloud_consent=False
    c.pref.translation_cloud_consent=False
    path=str(tmp_path/'new models')
    c.window.set_settings({'model_cache_dir':path,'whisper_size':'small.en',
                           'cloud_consent':False,'translation_cloud_consent':False})
    calls=[]
    monkeypatch.setattr(model_setup,'show_model_setup',lambda window,pref:calls.append(
        (window,pref.model_cache_dir,pref.whisper_size)))
    before=(c.listening,c.preparing,c.busy,list(c.service.calls))
    c.window.prepare_speech_requested.emit()
    assert calls==[(c.window,path,'small.en')]
    assert not c.pref.cloud_consent and not c.pref.translation_cloud_consent
    assert (c.listening,c.preparing,c.busy,list(c.service.calls))==before


def test_model_preparation_does_not_touch_active_capture(controller,monkeypatch):
    c=controller;c.listening=True
    calls=[]
    monkeypatch.setattr(model_setup,'show_model_setup',lambda *a,**kw:calls.append(True))
    c.window.prepare_speech_requested.emit()
    assert not calls and c.listening

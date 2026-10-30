"""Caption scheduling preference persistence/UX only; no ASR/cloud inference."""
import json
import os
from dataclasses import asdict

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import pytest
from PySide6.QtWidgets import QApplication, QFormLayout
from zoom_copilot.config import Preferences, app_dir
from zoom_copilot.ui import CopilotWindow


def test_default_caption_update_mode_is_original():
    assert Preferences().caption_update_mode=='original'


@pytest.mark.parametrize('value',[None,'','auto','FAST','responsive ',True,0,{},[]])
def test_invalid_explicit_update_mode_fails_to_original(value):
    assert Preferences(caption_update_mode=value).caption_update_mode=='original'


@pytest.mark.parametrize('mode',['responsive','original'])
def test_update_mode_roundtrips_preferences_without_credentials(tmp_path,monkeypatch,mode):
    monkeypatch.setenv('PRISM_INTERVIEW_HOME',str(tmp_path))
    prefs=Preferences(caption_update_mode=mode,api_key='not-persisted',
                      session_api_keys={'qwen':'not-persisted-either'})
    prefs.save()
    saved=json.loads((app_dir()/'preferences.json').read_text(encoding='utf-8'))
    assert saved['caption_update_mode']==mode
    assert 'api_key' not in saved and 'session_api_keys' not in saved
    assert Preferences.load().caption_update_mode==mode


def test_existing_settings_without_field_keep_original_and_other_choices(tmp_path,monkeypatch):
    monkeypatch.setenv('PRISM_INTERVIEW_HOME',str(tmp_path))
    legacy={'whisper_size':'large-v3-turbo','caption_window_seconds':6.0,
            'caption_speaker_mode':'acoustic','asr_device':'cpu','translation_engine':'qwen',
            'translation_cloud_consent':False}
    (app_dir()/'preferences.json').write_text(json.dumps(legacy),encoding='utf-8')
    loaded=Preferences.load()
    assert loaded.caption_update_mode=='original'
    assert all(getattr(loaded,key)==value for key,value in legacy.items())


def test_persisted_invalid_explicit_field_migrates_to_original(tmp_path,monkeypatch):
    monkeypatch.setenv('PRISM_INTERVIEW_HOME',str(tmp_path))
    (app_dir()/'preferences.json').write_text(json.dumps({'caption_update_mode':'unknown'}),encoding='utf-8')
    assert Preferences.load().caption_update_mode=='original'


@pytest.fixture
def window(tmp_path,monkeypatch):
    monkeypatch.setenv('PRISM_INTERVIEW_HOME',str(tmp_path))
    qt=QApplication.instance() or QApplication([])
    value=CopilotWindow()
    value.set_settings(asdict(Preferences()))
    yield value
    value.close();value.deleteLater();qt.processEvents()


@pytest.mark.parametrize('mode',['responsive','original'])
def test_ui_mode_roundtrip_and_save_emits_choice(window,mode):
    window.set_settings({'caption_update_mode':mode})
    assert window.caption_update_mode_select.currentData()==mode
    assert window.get_settings()['caption_update_mode']==mode
    received=[];window.settings_saved.connect(received.append)
    window._save_settings()
    assert received[-1]['caption_update_mode']==mode
    assert received[-1]['caption_window_seconds']==3.6
    assert received[-1]['whisper_size']=='large-v3-turbo'


def test_ui_defaults_and_invalid_setting_fallback(window):
    assert window.caption_update_mode_select.currentData()=='original'
    window.set_settings({})
    assert window.caption_update_mode_select.currentData()=='original'
    assert window.get_settings()['caption_update_mode']=='original'
    window.set_settings({'caption_update_mode':'broken'})
    assert window.caption_update_mode_select.currentData()=='original'
    assert window.get_settings()['caption_update_mode']=='original'


def test_new_control_immediately_follows_whisper_window_and_explains_scope(window):
    parent=window.caption_window_select.parentWidget()
    form=next(parent.layout().itemAt(i).layout() for i in range(parent.layout().count())
              if isinstance(parent.layout().itemAt(i).layout(),QFormLayout))
    assert form.getWidgetPosition(window.caption_update_mode_select)[0]==form.getWidgetPosition(window.caption_window_select)[0]+1
    assert window.caption_update_mode_select.itemText(0)=='更快确认（GPU·单人）'
    assert window.caption_update_mode_select.itemText(1)=='原版节奏'
    tip=window.caption_update_mode_select.toolTip()
    assert all(value in tip for value in ('保留 large-v3-turbo','beam','CUDA','单一远端','3.0','3.6','1.6','2.4',
                                         '停顿后重置','CPU','多人','50%','千问','费用','不保证识别准确性','重新开始监听'))


@pytest.mark.parametrize('device,speakers,window_seconds',[
    ('cpu','single_remote',3.6),('cuda','acoustic',3.6),
    ('cuda','single_remote',3),('cuda','single_remote',6),('cuda','single_remote',12)])
def test_ui_mode_selection_keeps_cpu_multi_and_custom_windows(window,device,speakers,window_seconds):
    window.set_settings({'caption_update_mode':'responsive','asr_device':device,
                         'caption_speaker_mode':speakers,'caption_window_seconds':window_seconds})
    settings=window.get_settings()
    assert settings['caption_update_mode']=='responsive'
    assert settings['asr_device']==device and settings['caption_speaker_mode']==speakers
    assert settings['caption_window_seconds']==window_seconds

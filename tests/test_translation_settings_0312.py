"""Continuous caption cloud consent is separate from answer cloud consent."""
import json
import os

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication
from zoom_copilot.config import Preferences
from zoom_copilot.ui import CopilotWindow


def test_new_caption_default_is_checked_but_legacy_settings_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv('PRISM_INTERVIEW_HOME', str(tmp_path))
    defaults = Preferences()
    assert defaults.translation_engine == 'qwen'
    assert defaults.translation_cloud_consent is True
    assert defaults.translation_qwen_model == 'qwen3.8-flash'
    assert defaults.translation_timeout == 20.0
    (tmp_path/'preferences.json').write_text(json.dumps({'provider':'qwen','cloud_consent':True}), encoding='utf-8')
    legacy = Preferences.load()
    assert legacy.cloud_consent is True
    assert legacy.translation_cloud_consent is False
    assert legacy.translation_engine == 'argos'


def test_fresh_checkbox_can_be_disabled_and_stays_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv('PRISM_INTERVIEW_HOME', str(tmp_path))
    prefs = Preferences.load()
    assert prefs.translation_cloud_consent is True
    prefs.translation_cloud_consent = False
    prefs.save()
    assert Preferences.load().translation_cloud_consent is False


def test_continuous_translation_preferences_roundtrip_without_api_keys(tmp_path, monkeypatch):
    monkeypatch.setenv('PRISM_INTERVIEW_HOME', str(tmp_path))
    preferences = Preferences(translation_engine='qwen', translation_cloud_consent=True,
        translation_qwen_model='fixture-model', translation_qwen_base_url='https://caption.example/v1/',
        translation_timeout=12.5, cloud_consent=False, api_key='test-session-only')
    preferences.save()
    payload = json.loads((tmp_path/'preferences.json').read_text(encoding='utf-8'))
    assert 'api_key' not in payload and 'session_api_keys' not in payload
    loaded = Preferences.load()
    assert loaded.translation_engine == 'qwen' and loaded.translation_cloud_consent is True
    assert loaded.translation_qwen_model == 'fixture-model'
    assert loaded.translation_qwen_base_url == 'https://caption.example/v1'
    assert loaded.translation_timeout == 12.5
    assert loaded.cloud_consent is False


@pytest.mark.parametrize('value',[None,'true','false',1,{},[]])
def test_nonboolean_continuous_cloud_consent_fails_closed(value):
    assert Preferences(translation_cloud_consent=value).translation_cloud_consent is False


@pytest.mark.parametrize(('value','expected'),[(None,20.0),('bad',20.0),(float('nan'),20.0),
    (float('inf'),20.0),(-1,3.0),(100,60.0)])
def test_translation_timeout_is_finite_and_bounded(value,expected):
    assert Preferences(translation_timeout=value).translation_timeout == expected


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    instance = CopilotWindow()
    yield instance
    instance.close()
    instance.deleteLater()
    app.processEvents()


def test_ui_roundtrip_cloud_caption_configuration_is_independent(window):
    emitted=[]
    window.settings_saved.connect(emitted.append)
    window.set_settings({'provider':'deepseek', 'cloud_consent':False,
        'api_base_url':'https://answer.example/v1', 'api_key':'test-only-answer',
        'translation_engine':'qwen', 'translation_cloud_consent':True,
        'translation_qwen_model':'fixture-caption-model',
        'translation_qwen_base_url':'https://caption.example/v1', 'translation_timeout':11.5})
    settings = window.get_settings()
    assert emitted == []
    assert settings['provider'] == 'deepseek'
    assert settings['api_base_url'] == 'https://answer.example/v1'
    assert settings['api_key'] == 'test-only-answer'
    assert settings['cloud_consent'] is False
    assert settings['translation_cloud_consent'] is True
    assert settings['translation_qwen_model'] == 'fixture-caption-model'
    assert settings['translation_qwen_base_url'] == 'https://caption.example/v1'
    assert settings['translation_timeout'] == 11.5
    assert window.translation_cloud_checkbox.isEnabled()


def test_selecting_qwen_engine_or_answer_provider_never_grants_continuous_consent(window):
    window.set_settings({'provider':'qwen','cloud_consent':True,'translation_cloud_consent':False})
    window.translation_engine_select.setCurrentIndex(window.translation_engine_select.findData('qwen'))
    assert window.translation_cloud_checkbox.isEnabled()
    assert window.get_settings()['translation_cloud_consent'] is False
    window.provider_select.setCurrentIndex(window.provider_select.findData('deepseek'))
    assert window.get_settings()['translation_cloud_consent'] is False
    assert window.translation_engine_select.currentData() == 'qwen'


def test_local_engine_disables_cloud_controls_without_erasing_saved_choice(window):
    window.set_settings({'translation_engine':'qwen','translation_cloud_consent':True,
                         'translation_qwen_model':'fixture-model'})
    window.translation_engine_select.setCurrentIndex(window.translation_engine_select.findData('argos'))
    assert window.translation_cloud_checkbox.isEnabled()  # Revocation remains accessible.
    assert not window.translation_qwen_model_input.isEnabled()
    assert window.get_settings()['translation_qwen_model'] == 'fixture-model'
    assert window.get_settings()['translation_cloud_consent'] is True
    assert window.get_settings()['translation_engine'] == 'argos'


def test_ui_reports_actual_active_engine_separately_from_next_capture_settings(window):
    window.set_translation_engine_status('argos')
    active = window.translation_active_label.text()
    window.set_settings({'translation_engine':'qwen','translation_cloud_consent':True})
    assert window.translation_active_label.text() == active
    window.set_translation_engine_status('qwen','qwen3.8-flash')
    assert '千问' in window.translation_active_label.text()
    assert 'qwen3.8-flash' in window.translation_active_label.text()
    assert '重新开始监听' in window.translation_active_label.toolTip()


def test_revoking_continuous_caption_consent_is_immediate_but_grant_does_not_submit(window):
    window.set_settings({'translation_engine':'qwen','translation_cloud_consent':False})
    emitted=[]
    window.settings_saved.connect(emitted.append)
    window.translation_cloud_checkbox.setChecked(True)
    assert emitted == []
    window.translation_cloud_checkbox.setChecked(False)
    assert len(emitted) == 1
    assert emitted[0]['translation_cloud_consent'] is False
    assert emitted[0]['translation_engine'] == 'qwen'


def test_revocation_remains_accessible_after_selecting_local_for_next_capture(window):
    window.set_settings({'translation_engine':'qwen','translation_cloud_consent':True})
    window.set_translation_engine_status('qwen','qwen3.8-flash')
    window.translation_engine_select.setCurrentIndex(window.translation_engine_select.findData('argos'))
    emitted=[]
    window.settings_saved.connect(emitted.append)
    assert window.translation_cloud_checkbox.isEnabled()
    window.translation_cloud_checkbox.setChecked(False)
    assert len(emitted) == 1 and emitted[0]['translation_cloud_consent'] is False

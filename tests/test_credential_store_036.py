import json
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest

from zoom_copilot import credential_store as vault, providers
from zoom_copilot.config import Preferences


def test_target_is_scoped_and_normalizes_https_default_port():
    target = vault.credential_target('qwen', 'https://EXAMPLE.com:443/v1/')
    assert target == vault.credential_target('qwen', 'https://example.com/v1')
    assert target != vault.credential_target('deepseek', 'https://example.com/v1')
    assert target != vault.credential_target('qwen', 'https://other.example.com/v1')
    assert target != vault.credential_target('qwen', 'https://example.com/v2')
    assert 'example.com' not in target


@pytest.mark.parametrize('endpoint', ['http://example.com/v1', 'https://u:p@example.com/v1',
                                    'https://example.com/v1?k=v', 'https://example.com:9443/v1'])
def test_invalid_endpoint_cannot_access_vault(endpoint):
    with pytest.raises(ValueError):
        vault.credential_target('qwen', endpoint)


def test_unsupported_provider_cannot_access_vault():
    with pytest.raises(ValueError):
        vault.credential_target('codex', 'https://example.com/v1')


def prefs(**kwargs):
    return Preferences(provider='qwen', cloud_consent=True, free_quota_confirmed=True,
                       api_base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', **kwargs)


def test_vault_is_fallback_after_session_and_environment(monkeypatch):
    monkeypatch.delenv('DASHSCOPE_API_KEY', raising=False)
    calls = []
    monkeypatch.setattr(providers, 'load_key', lambda p, e: calls.append((p, e)) or 'test-saved')
    assert providers.provider_connection(prefs())[2] == 'test-saved'
    assert calls == [('qwen', 'https://dashscope.aliyuncs.com/compatible-mode/v1')]
    calls.clear()
    assert providers.provider_connection(prefs(session_api_keys={'qwen': 'test-session'}))[2] == 'test-session'
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'test-env')
    assert providers.provider_connection(prefs())[2] == 'test-env'
    assert not calls


def test_no_vault_read_before_consent_quota_and_endpoint_validation(monkeypatch):
    calls = []
    monkeypatch.setattr(providers, 'load_key', lambda *args: calls.append(args) or 'test-saved')
    for field, value in [('cloud_consent', False), ('free_quota_confirmed', False),
                         ('api_base_url', 'https://untrusted.invalid/v1')]:
        value_prefs = prefs()
        setattr(value_prefs, field, value)
        with pytest.raises(ValueError):
            providers.provider_connection(value_prefs)
    assert not calls


def test_preferences_still_exclude_all_session_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv('PRISM_INTERVIEW_HOME', str(tmp_path))
    prefs(api_key='test-secret', session_api_keys={'qwen': 'test-secret'}).save()
    body = (tmp_path/'preferences.json').read_text(encoding='utf8')
    assert 'test-secret' not in body
    assert 'api_key' not in json.loads(body) and 'session_api_keys' not in json.loads(body)


def test_picker_saves_and_deletes_only_selected_target(monkeypatch):
    from PySide6.QtWidgets import QApplication
    from zoom_copilot.model_picker import ModelPickerDialog
    app = QApplication.instance() or QApplication([])
    calls = []
    monkeypatch.setattr(vault, 'store_key', lambda *args: calls.append(('store', *args)))
    monkeypatch.setattr(vault, 'delete_key', lambda *args: calls.append(('delete', *args)))
    dialog = ModelPickerDialog(prefs())
    dialog.key_input.setText('test-secret')
    dialog.remember_key_button.click()
    assert calls == [('store', 'qwen', 'https://dashscope.aliyuncs.com/compatible-mode/v1', 'test-secret')]
    assert dialog.key_input.text() == 'test-secret'
    assert '自动填入' in dialog.key_status_label.text()
    assert dialog.selected_preferences().api_key == ''
    assert not dialog.selected_preferences().session_api_keys
    assert 'test-secret' not in dialog.status_label.text()
    dialog.forget_key_button.click()
    assert calls[-1] == ('delete', 'qwen', 'https://dashscope.aliyuncs.com/compatible-mode/v1')
    dialog.close()
    app.processEvents()

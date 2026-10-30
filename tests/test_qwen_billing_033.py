"""Paid Qwen consent must not relax any other provider or privacy boundary."""
from dataclasses import replace
import json
import pytest
from zoom_copilot.config import Preferences
from zoom_copilot.providers import provider_connection


def test_paid_authorization_is_qwen_only_and_requires_cloud_consent():
    prefs = Preferences(provider='qwen', cloud_consent=True,
                        qwen_paid_usage_authorized=True, api_key='test-qwen',
                        free_quota_confirmed=False)
    assert provider_connection(prefs)[0] == 'qwen'
    with pytest.raises(ValueError, match='授权云端'):
        provider_connection(replace(prefs, cloud_consent=False))
    with pytest.raises(ValueError, match='免费额度'):
        provider_connection(replace(prefs, provider='deepseek'))
    with pytest.raises(ValueError, match='免费额度'):
        provider_connection(replace(prefs, provider='compatible', api_base_url='https://example.com/v1'))
    with pytest.raises(ValueError, match='免费额度'):
        provider_connection(replace(prefs, qwen_paid_usage_authorized=False))


def test_paid_consent_persists_but_neither_provider_key_does(tmp_path, monkeypatch):
    monkeypatch.setenv('PRISM_INTERVIEW_HOME', str(tmp_path))
    prefs = Preferences(provider='qwen', cloud_consent=True,
                        qwen_paid_usage_authorized=True, api_key='test-secret-one',
                        session_api_keys={'qwen':'test-secret-two'})
    prefs.save()
    data = json.loads((tmp_path/'preferences.json').read_text(encoding='utf-8'))
    assert data['qwen_paid_usage_authorized'] is True
    assert 'api_key' not in data and 'session_api_keys' not in data
    assert 'test-secret' not in json.dumps(data)
    assert Preferences.load().qwen_paid_usage_authorized is True


def test_paid_authorization_defaults_off():
    assert Preferences().qwen_paid_usage_authorized is False


def test_official_qianwen_platform_host_supported_without_lookalike_hosts():
    prefs = Preferences(provider='qwen', cloud_consent=True,
                        qwen_paid_usage_authorized=True, api_key='test-qwen',
                        api_base_url='https://dashscope.aliyuncs.com/compatible-mode/v1')
    assert provider_connection(prefs)[1] == prefs.api_base_url
    for host in ('dashscope.aliyuncs.com.example.org', 'evil.aliyuncs.com'):
        with pytest.raises(ValueError,match='官方域名'):
            provider_connection(replace(prefs,api_base_url='https://'+host+'/compatible-mode/v1'))

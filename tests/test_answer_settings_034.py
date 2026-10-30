"""Full bilingual answers must survive configuration and UI round trips."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from dataclasses import replace
import json
import pytest
from PySide6.QtWidgets import QApplication

from zoom_copilot.config import Preferences
from zoom_copilot.providers import OpenAICompatibleClient
from zoom_copilot.ui import CopilotWindow


def test_new_full_answer_settings_and_legacy_explicit_choice_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv('PRISM_INTERVIEW_HOME', str(tmp_path))
    prefs = Preferences()
    assert not prefs.brief_answer and prefs.max_answer_words == 220
    Preferences(brief_answer=True, max_answer_words=60).save()
    old = Preferences.load()
    assert old.brief_answer and old.max_answer_words == 60
    replace(old, brief_answer=False, max_answer_words=220).save()
    loaded = Preferences.load()
    assert not loaded.brief_answer and loaded.max_answer_words == 220


@pytest.mark.parametrize('value,expected', [(None,220), ('bad',220), (float('nan'),220),
                                           (-10,60), (999,360), (180,180)])
def test_corrupt_or_unbounded_word_limits_are_bounded(value, expected):
    assert Preferences(max_answer_words=value).max_answer_words == expected


def test_user_can_change_full_answer_length_without_silent_save_on_load():
    app = QApplication.instance() or QApplication([])
    window = CopilotWindow()
    saved = []
    window.settings_saved.connect(saved.append)
    try:
        window.set_settings({'max_answer_words':220, 'brief_answer':False})
        assert saved == []
        window.answer_words_input.setValue(280)
        window._save_settings_clicked()
        assert saved[-1]['max_answer_words'] == 280
        assert saved[-1]['brief_answer'] is False
        window.set_settings({'max_answer_words':80, 'brief_answer':True})
        assert window.get_settings()['max_answer_words'] == 80
        assert window.get_settings()['brief_answer'] is True
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_full_bilingual_output_budget_is_distinct_from_short_mode(monkeypatch):
    from zoom_copilot import providers
    requests = []
    monkeypatch.setattr(providers, 'stream_http', lambda request, *a, **kw:
                        requests.append(json.loads(request.data)) or '{}')
    prefs = Preferences(provider='qwen', answer_model='qwen3.7-plus',
                        cloud_consent=True, qwen_paid_usage_authorized=True,
                        api_key='synthetic-session-secret')
    OpenAICompatibleClient(prefs).complete('system', 'question')
    OpenAICompatibleClient(replace(prefs, brief_answer=True, max_answer_words=80)).complete('system', 'question')
    assert requests[0]['max_tokens'] == 2000
    assert requests[1]['max_tokens'] == 1000
    assert requests[0]['enable_thinking'] is False
    assert requests[0]['response_format'] == {'type':'json_object'}

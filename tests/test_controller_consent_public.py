"""Settings updates cannot reinterpret malformed values as upload/payment consent."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from zoom_copilot import app as module
from test_controller_v2 import controller


def test_controller_normalizes_invalid_consent_before_active_queued_and_caption_checks(controller):
    c = controller
    c.request_answer("Please describe your experience.")
    c.request_answer("What would you improve next?")
    c.pref.translation_cloud_consent = True
    c.pref.qwen_paid_usage_authorized = True
    c.pref.free_quota_confirmed = True
    c.pref.auto_save_sessions = True
    stopped = []
    c.translation = SimpleNamespace(stop=lambda: stopped.append(True))
    c._listen_pref = replace(c.pref, translation_engine="qwen")
    invalid = {"cloud_consent": "false", "share_evidence": 1,
               "translation_cloud_consent": "yes", "qwen_paid_usage_authorized": {"enabled": True},
               "free_quota_confirmed": [True], "auto_save_sessions": "false"}
    c.save_settings(invalid)
    assert all(getattr(c.pref, name) is False for name in invalid)
    assert c._cancel.is_set()
    assert c._requests[0][2].cloud_consent is False
    assert c._requests[0][2].share_evidence is False
    assert stopped == [True] and c.translation is None
    saved = json.loads((module.app_dir() / "preferences.json").read_text(encoding="utf-8"))
    assert all(saved[name] is False for name in invalid)
    assert not c.service.calls


def test_controller_preserves_explicit_boolean_consent_without_calling_provider(controller):
    c = controller
    values = {"cloud_consent": True, "share_evidence": True,
              "translation_cloud_consent": True, "qwen_paid_usage_authorized": True,
              "free_quota_confirmed": False, "auto_save_sessions": True}
    c.save_settings(values)
    assert all(getattr(c.pref, name) is value for name, value in values.items())
    assert not c.service.calls and not c.listening
    c.pref.auto_save_sessions = False  # Teardown must not persist an empty fixture.


@pytest.mark.parametrize("content", ["[]", "null", '"invalid"', "true", "42"])
def test_runtime_non_object_is_ignored_without_loading_paths_or_network(tmp_path, monkeypatch, content):
    base = tmp_path / "application"
    base.mkdir()
    (base / "runtime.json").write_text(content, encoding="utf-8")
    monkeypatch.setattr(module.sys, "frozen", True, raising=False)
    monkeypatch.setattr(module.sys, "executable", str(base / "PrismInterview.exe"))
    before = {name: module.os.environ.get(name) for name in
              ("PRISM_INTERVIEW_HOME", "PRISM_INTERVIEW_MODELS", "PRISM_INTERVIEW_CUDA_DIR")}
    module.load_runtime()
    assert {name: module.os.environ.get(name) for name in before} == before

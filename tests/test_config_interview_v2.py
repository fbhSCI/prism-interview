"""Safe interview-phase migration and persistence, without changing auto preference."""
import json

from zoom_copilot.config import Preferences


def test_existing_preferences_without_phase_use_requested_automatic_qa_default(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISM_INTERVIEW_HOME", str(tmp_path))
    (tmp_path / "preferences.json").write_text(json.dumps({"auto_answer": True}), encoding="utf-8")
    prefs = Preferences.load()
    assert prefs.interview_phase == "qa"
    assert prefs.auto_answer is True


def test_qa_phase_roundtrips_without_persisting_session_key(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISM_INTERVIEW_HOME", str(tmp_path))
    prefs = Preferences(interview_phase="qa", auto_answer=True, api_key="only-a-test-session-key")
    prefs.save()
    data = json.loads((tmp_path / "preferences.json").read_text(encoding="utf-8"))
    assert data["interview_phase"] == "qa"
    assert "api_key" not in data
    loaded = Preferences.load()
    assert loaded.interview_phase == "qa" and loaded.auto_answer is True
    assert loaded.api_key == ""


def test_invalid_phase_falls_back_to_requested_qa_default():
    assert Preferences(interview_phase="unknown").interview_phase == "qa"
    assert Preferences().interview_phase == "qa"


def test_answer_speed_defaults_to_authorized_fast_and_preserves_saved_standard(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISM_INTERVIEW_HOME", str(tmp_path))
    assert Preferences().answer_speed == "fast"
    assert Preferences(answer_speed="unsupported").answer_speed == "standard"
    Preferences(answer_speed="fast").save()
    assert Preferences.load().answer_speed == "fast"
    Preferences(answer_speed="standard").save()
    assert Preferences.load().answer_speed == "standard"

"""Local conversation persistence, export, opt-in and read-only browsing."""
import json

import pytest
from PySide6.QtWidgets import QDialog, QListWidget, QPlainTextEdit

from zoom_copilot.answers import BilingualAnswer
from zoom_copilot.session_store import SessionStore, build_snapshot, new_session_id
from test_controller_v2 import controller


def sample():
    result = BilingualAnswer("你如何处理意见分歧？", "I first clarify the shared goal.",
        "我会先明确共同目标。", ["CV.pdf · page 2"], provider="local", model="demo",
        evidence=[{"id": "E1", "source": r"C:\private\CV.pdf", "location": "page 2",
                   "text": "Led a team discussion.", "api_key": "MUST_NOT_SAVE"}],
        pairs=[{"id": "p1", "en": "I first clarify the shared goal.", "zh": "我会先明确共同目标。",
                "prompt": "MUST_NOT_SAVE"}])
    return build_snapshot(new_session_id(), "2026-01-01T10:00:00+00:00",
        [{"id": "t1", "en": "How do you handle disagreement?", "zh": "你如何处理意见分歧？",
          "raw_en": "Um, how do you handle disagreement?", "speaker_id": 2,
          "speaker_label": "Speaker 2", "api_key": "MUST_NOT_SAVE", "audio": "MUST_NOT_SAVE"}],
        [{"question": "How do you handle disagreement?", "result": result,
          "api_key": "MUST_NOT_SAVE"}], updated_at="2026-01-01T10:01:00+00:00")


def test_roundtrip_keeps_bilingual_speaker_answer_evidence_without_state(tmp_path):
    store = SessionStore(tmp_path / "sessions")
    snapshot = sample()
    snapshot["settings"] = {"api_key": "MUST_NOT_SAVE"}
    destination = store.save(snapshot)
    restored = store.load(snapshot["id"])
    text = destination.read_text(encoding="utf-8")
    assert "MUST_NOT_SAVE" not in text and '"audio"' not in text and '"settings"' not in text
    assert restored["transcripts"][0]["raw_en"].startswith("Um,")
    assert restored["transcripts"][0]["speaker_id"] == 2
    assert restored["answers"][0]["pairs"][0]["zh"] == "我会先明确共同目标。"
    assert restored["answers"][0]["evidence"][0]["source"] == "CV.pdf"
    assert restored["answers"][0]["provider"] == "local"
    assert not list(store.root.glob("*.tmp"))


@pytest.mark.parametrize("extension", [".json", ".md", ".txt"])
def test_export_keeps_complete_bilingual_text_and_evidence(tmp_path, extension):
    snapshot = sample()
    target = tmp_path / ("conversation" + extension)
    SessionStore(tmp_path / "sessions").export(snapshot, target)
    text = target.read_text(encoding="utf-8")
    assert "How do you handle disagreement?" in text
    assert "我会先明确共同目标。" in text and "Led a team discussion." in text
    if extension == ".json":
        assert json.loads(text)["schema_version"] == 1
    else:
        assert "Speaker 2" in text and "Um, how" in text


def test_atomic_failure_retains_previous_record_and_cleans_temporary(tmp_path, monkeypatch):
    store = SessionStore(tmp_path)
    snapshot = sample()
    path = store.save(snapshot)
    original = path.read_bytes()
    monkeypatch.setattr("zoom_copilot.session_store.os.replace", lambda *args: (_ for _ in ()).throw(OSError("disk unavailable")))
    snapshot["transcripts"][0]["en"] = "A changed turn."
    with pytest.raises(OSError):
        store.save(snapshot)
    assert path.read_bytes() == original
    assert not list(tmp_path.glob("*.tmp"))


@pytest.mark.parametrize("identity", ["../escape", "", "a" * 33, "A" * 32, r"..\escape"])
def test_history_paths_reject_traversal(tmp_path, identity):
    with pytest.raises(ValueError):
        SessionStore(tmp_path).load(identity)


def test_listing_ignores_corrupt_foreign_and_mismatched_files(tmp_path):
    store = SessionStore(tmp_path)
    valid = sample()
    store.save(valid)
    (tmp_path / (new_session_id() + ".json")).write_text("broken", encoding="utf-8")
    foreign = sample()
    foreign["schema_version"] = 99
    (tmp_path / (foreign["id"] + ".json")).write_text(json.dumps(foreign), encoding="utf-8")
    (tmp_path / (new_session_id() + ".json")).write_text(json.dumps(valid), encoding="utf-8")
    assert [row["id"] for row in store.list_sessions()] == [valid["id"]]
    store.delete(valid["id"])
    assert store.list_sessions() == []


def test_schema_and_export_extension_are_explicit(tmp_path):
    snapshot = sample()
    store = SessionStore(tmp_path)
    with pytest.raises(ValueError):
        store.export(snapshot, tmp_path / "session.html")
    snapshot["schema_version"] = True
    with pytest.raises(ValueError):
        store.save(snapshot)


def test_manual_save_does_not_enable_auto_save_or_call_model(controller):
    c = controller
    assert c.pref.auto_save_sessions is False
    assert not c.session_store.root.exists()
    c.on_transcript("final", "Please explain your approach.")
    c._auto_save_conversation()
    assert not c.session_store.root.exists()
    path = c.save_conversation()
    assert path and path.is_file()
    assert not c.pref.auto_save_sessions and not c.service.calls
    assert not c.listening and not c.busy
    assert c.session_store.load(c._saved_session_id)["transcripts"][0]["en"]


def test_auto_save_is_opt_in_updates_same_session_and_stops_when_disabled(controller):
    c = controller
    c.on_transcript("final", "Please describe the team project.")
    c.set_auto_save_sessions(True)
    c._auto_save_conversation()
    rows = c.session_store.list_sessions()
    assert len(rows) == 1
    c._transcripts[0]["zh"] = "请介绍这个团队项目。"
    c._session_dirty = True
    c._auto_save_conversation()
    assert len(c.session_store.list_sessions()) == 1
    assert c.session_store.load(rows[0]["id"])["transcripts"][0]["zh"] == "请介绍这个团队项目。"
    c.set_auto_save_sessions(False)
    c._transcripts[0]["zh"] = "Should remain in memory only."
    c._session_dirty = True
    c._auto_save_conversation()
    assert c.session_store.load(rows[0]["id"])["transcripts"][0]["zh"] == "请介绍这个团队项目。"
    assert not c.service.calls


def test_history_browser_is_complete_read_only_and_has_no_capture_or_model_side_effect(controller, monkeypatch):
    c = controller
    c.session_store.save(sample())
    before = (c.listening, c.busy, list(c.service.calls), list(c.test_work))
    inspected = []

    def inspect(dialog):
        archive = dialog.findChild(QListWidget)
        preview = dialog.findChild(QPlainTextEdit)
        assert archive.count() == 1
        assert preview.isReadOnly()
        assert "How do you handle disagreement?" in preview.toPlainText()
        assert "我会先明确共同目标。" in preview.toPlainText()
        inspected.append(True)
        return 0

    monkeypatch.setattr(QDialog, "exec", inspect)
    c.show_session_history()
    assert inspected == [True]
    assert (c.listening, c.busy, list(c.service.calls), list(c.test_work)) == before


def test_completed_answer_is_saved_with_raw_question_but_unfinished_stream_is_not(controller):
    c = controller
    c.request_answer("How do you work with others?")
    c.post("answer_delta", (c._active, "answer_en", "A provisional answer."))
    c.pump()
    assert not c._conversation_snapshot()["answers"]
    c.test_run_worker()
    snapshot = c._conversation_snapshot()
    assert len(snapshot["answers"]) == 1
    assert snapshot["answers"][0]["answer_en"] == "Suggested English answer."
    assert snapshot["answers"][0]["question_raw_en"] == "How do you work with others?"


def test_shutdown_saves_final_visible_text_only_when_auto_save_was_selected(controller):
    c = controller
    c.set_auto_save_sessions(True)
    c.on_transcript("final", "Thank you for the interview.")
    c.close()
    assert not c.session_save_timer.isActive()
    assert c.session_store.load(c._saved_session_id)["transcripts"][0]["en"]


def test_invalid_truthy_setting_cannot_turn_on_conversation_persistence(controller):
    c = controller
    c.save_settings({"auto_save_sessions": "false"})
    c.on_transcript("final", "Please describe your background.")
    c._auto_save_conversation()
    assert c.pref.auto_save_sessions is False
    assert not c.session_store.root.exists()

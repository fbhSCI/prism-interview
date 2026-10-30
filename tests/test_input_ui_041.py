"""Source selection, independent speaker cards and remote-only question routing."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from dataclasses import replace
from types import SimpleNamespace
import threading
import time

from PySide6.QtWidgets import QApplication
from zoom_copilot.ui import CopilotWindow
from zoom_copilot import app as controller_module
from zoom_copilot.config import Preferences
from zoom_copilot.model_picker import ModelPickerDialog
from test_controller_v2 import controller


def both(c):
    c.window.set_settings({"input_source": "both", "speaker_id": "speaker", "microphone_id": "mic",
                           "caption_speaker_mode": "single_remote", "interview_phase": "qa", "auto_answer": True})
    c.window.set_devices([("speaker", "Conference output")], "speaker", source="loopback")
    c.window.set_devices([("mic", "My headset microphone")], "mic", source="microphone")
    c.save_settings(c.window.get_settings())
    c._listen_pref = replace(c.pref)


def context(c, source, start=0, end=48000):
    return {"input_source": source, "phase_epoch": c._phase_epoch,
            "stream_id": source + "-clock", "start_sample": start, "end_sample": end,
            "word_timed": True}


def test_source_selector_remembers_both_devices_and_never_selects_missing_headset():
    qt = QApplication.instance() or QApplication([])
    w = CopilotWindow()
    try:
        w.set_settings({"input_source": "loopback", "speaker_id": "out", "microphone_id": "mic"})
        w.set_devices([("out", "Zoom output")], "out", source="loopback")
        w.set_devices([("mic", "Headset")], "mic", source="microphone")
        w.input_source_select.setCurrentIndex(w.input_source_select.findData("both"))
        values = w.get_settings()
        assert values["input_source"] == "both"
        assert values["speaker_id"] == "out" and values["microphone_id"] == "mic"
        assert w.microphone_select.isHidden() is False
        w.input_source_select.setCurrentIndex(w.input_source_select.findData("microphone"))
        assert w.get_settings()["speaker_id"] == "out" and w.device_select.currentData() == "mic"
        w.set_devices([("other", "Laptop microphone")], "mic", source="microphone")
        assert not w.device_select.currentData() and not w.start_button.isEnabled()
        assert "不可用" in w.device_select.currentText()
        w.input_source_select.setCurrentIndex(w.input_source_select.findData("loopback"))
        assert w.device_select.currentData() == "out"
    finally:
        w.close(); w.deleteLater(); qt.processEvents()


def test_new_microphone_requires_explicit_device_choice():
    qt = QApplication.instance() or QApplication([])
    w = CopilotWindow()
    try:
        w.set_settings({"input_source": "microphone", "microphone_id": ""})
        w.set_devices([("mic", "Headset microphone")], source="microphone")
        assert not w.device_select.currentData()
        started = []
        w.start_requested.connect(lambda: started.append(True))
        w._start()
        assert not started
        w.device_select.setCurrentIndex(w.device_select.findData("mic"))
        w._start()
        assert started == [True]
    finally:
        w.close(); w.deleteLater(); qt.processEvents()


def test_interleaved_sources_keep_separate_caption_cards_and_remote_question(controller):
    c = controller; both(c)
    c.on_transcript("final", "Please explain the main", context(c, "loopback"))
    remote_id = c._transcripts[0]["id"]
    before = (c.assembler.text, c.assembler.version, c._question_original, c._question_uncertain)
    c.on_transcript("final", "I can explain my own role.", context(c, "microphone"))
    assert (c.assembler.text, c.assembler.version, c._question_original, c._question_uncertain) == before
    c.on_transcript("final", "trade-off in your design.", context(c, "loopback", 48000, 96000))
    assert len(c._transcripts) == 2
    remote = next(r for r in c._transcripts if r["id"] == remote_id)
    mine = next(r for r in c._transcripts if r["speaker_id"] == "local-microphone")
    assert "main trade-off" in remote["en"] and "my own" not in remote["en"]
    assert mine["speaker_label"] == "我 · 麦克风"
    assert mine["en"] == "I can explain my own role."
    assert "my own" not in c._question_original


def test_mic_drafts_gaps_and_rejections_do_not_invalidate_remote_question(controller):
    c = controller; both(c)
    c.on_transcript("final", "How did you validate the result?", context(c, "loopback"))
    before = (c.assembler.text, c.assembler.version, c._question_original)
    c.on_transcript("partial", "Why would I", context(c, "microphone"))
    assert c._live_data["speaker_label"] == "我 · 麦克风"
    bad = dict(context(c, "microphone"), recognition_gap=True)
    c.on_transcript("final", "I would compare the outcome.", bad)
    c.on_transcript("rejected", "", context(c, "microphone"))
    assert not c._question_uncertain
    assert (c.assembler.text, c.assembler.version, c._question_original) == before
    assert c.window.current_question().find("Why would I") < 0


def test_auto_answer_ignores_mic_pending_and_draft(controller, monkeypatch):
    c = controller; both(c)
    c.listening = True
    calls = []
    monkeypatch.setattr(c, "request_answer", lambda q, automatic=False, **kw: calls.append((q, automatic)))
    c.on_transcript("final", "How did you validate the result?", context(c, "loopback"))
    c.on_transcript("partial", "I would start by", context(c, "microphone"))
    c.assembler.last_voice = c.assembler.last_text = time.monotonic() - 5
    c.speech = SimpleNamespace(pending=10, pending_for_source=lambda source: 0,
                              stop=lambda **kw: None, thread=None)
    c.auto_answer()
    assert calls and all("start by" not in q for q, _ in calls)
    assert calls[-1][1] is True


def test_microphone_only_never_autoanswers_but_manual_request_still_works(controller):
    c = controller
    c.window.set_settings({"input_source": "microphone", "microphone_id": "mic"})
    c.window.set_devices([("mic", "Headset")], "mic", source="microphone")
    c.save_settings(c.window.get_settings())
    c._listen_pref = replace(c.pref); c.listening = True
    c.on_transcript("final", "How do I handle the result?", context(c, "microphone"))
    c.auto_answer()
    assert not c.busy and not c.service.calls and not c.assembler.text
    c.request_answer("How would you organize a team?")
    c.test_run_worker()
    assert c.service.calls[0]["question"] == "How would you organize a team?"


def test_switching_source_stops_old_capture_without_autorestart(controller):
    c = controller
    stops = []
    c.listening = True
    c.audio = SimpleNamespace(stop=lambda: stops.append("audio"), thread=None)
    c.speech = SimpleNamespace(stop=lambda **kw: stops.append("speech"), thread=None)
    c.window.set_listening(True)
    old_session = c._session
    c.window.input_source_select.setCurrentIndex(c.window.input_source_select.findData("microphone"))
    assert c.stopping and not c.listening and c._session == old_session
    assert c.pref.input_source == "microphone" and stops == ["audio"]
    assert not c.window.start_button.isEnabled()


def test_stale_device_discovery_never_replaces_selected_source(controller):
    c = controller
    c._device_refresh_generation = 4
    c.window.set_settings({"input_source": "microphone", "microphone_id": "mic"})
    c.window.set_devices([("mic", "My microphone")], "mic", source="microphone")
    c.pref.input_source = "microphone"
    c.post("devices", {"generation": 3, "input_source": "loopback", "devices": [("old", "Old output")]})
    c.pump()
    assert c.window.device_select.currentData() == "mic"


def test_paid_qwen_can_be_explicitly_authorized_in_model_dialog_without_calling_model():
    qt = QApplication.instance() or QApplication([])
    d = ModelPickerDialog(Preferences())
    try:
        assert d.qwen_paid_checkbox.isChecked() is False
        d.qwen_paid_checkbox.setChecked(True)
        chosen = d.selected_preferences()
        assert chosen.provider == "qwen" and chosen.qwen_paid_usage_authorized is True
        assert chosen.cloud_consent is False
        d.provider_select.setCurrentIndex(d.provider_select.findData("deepseek"))
        assert d.selected_preferences().qwen_paid_usage_authorized is False
    finally:
        d.close(); d.deleteLater(); qt.processEvents()


def test_both_mode_reports_visible_output_when_saved_output_was_blank():
    qt = QApplication.instance() or QApplication([])
    w = CopilotWindow()
    try:
        w.set_settings({"input_source": "both", "speaker_id": "", "microphone_id": "mic"})
        w.set_devices([("out", "Meeting output")], source="loopback")
        w.set_devices([("mic", "Headset")], "mic", source="microphone")
        assert w.get_settings()["speaker_id"] == "out"
        assert w.get_settings()["microphone_id"] == "mic"
    finally:
        w.close(); w.deleteLater(); qt.processEvents()


def test_empty_remote_commit_retires_its_draft_without_replacing_visible_mic_card(controller):
    c = controller; both(c)
    c.on_transcript("final", "How did you validate the result?", context(c, "loopback"))
    c.on_transcript("partial", "Please explain the validation", dict(context(c, "loopback", 48000, 96000), text_end_sample=80000))
    assert c._remote_partial_id()
    c.on_transcript("partial", "I would explain my part", dict(context(c, "microphone", 0, 96000), text_end_sample=60000))
    live = dict(c._live_data)
    question = c._question_original
    c.on_transcript("final", "", dict(context(c, "loopback", 48000, 96000), empty_commit=True))
    assert c._remote_partial_id() is None
    assert c._live_data == live and c._live_data["speaker_label"] == "我 · 麦克风"
    assert c._question_original == question


def test_capture_failure_blocks_late_auto_dispatch_but_keeps_manual_review(controller):
    c = controller; both(c); c.listening = True
    c.on_transcript("final", "How did you validate", context(c, "loopback"))
    failed = threading.Event(); failed.set()
    c.audio = SimpleNamespace(error_event=failed, stop=lambda: None, thread=None)
    c.speech = SimpleNamespace(pending=0, pending_for_source=lambda s: 0,
                              stop=lambda **kw: None, thread=None)
    c.pump()
    assert c._capture_error_handled and not c.listening and c.stopping
    c.on_transcript("final", "the result? Sure.", context(c, "loopback", 48000, 96000))
    c.auto_answer()
    c.request_answer("How did you validate the result?", automatic=True)
    assert not c.busy and not c._requests and not c.service.calls
    c.request_answer("How did you validate the result?")
    assert c.busy and c._active not in c._automatic_ids


def test_model_preparation_does_not_allow_simultaneous_capture_start(controller):
    c = controller
    c.window._model_setup_dialog = SimpleNamespace(active=True)
    c.start_listening()
    assert not c.listening and not c.preparing
    assert "语音模型仍在准备" in c.window.status_label.text()

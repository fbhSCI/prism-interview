"""Controller user scenarios with real Qt widgets and deterministic worker boundaries.

No audio devices, model downloads, cloud inference, or API credentials are used.
"""
from __future__ import annotations

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import replace
from types import SimpleNamespace

import pytest
from PySide6.QtWidgets import QApplication

from zoom_copilot import app as controller_module
from zoom_copilot.answers import BilingualAnswer
from zoom_copilot.codex_client import AnswerCancelled
from zoom_copilot.config import Preferences
from zoom_copilot.ui import CopilotWindow


class LocalIndex:
    document_names = []
    document_records = {}
    passages = []

    def __init__(self, *args):
        self.queries = []

    def search(self, question, top_k=5):
        self.queries.append((question, top_k))
        return []


class StubAnswerService:
    """Records controller choices without pretending to validate model quality."""
    def __init__(self, preferences):
        self.preferences = preferences
        self.calls = []
        self.failure = ""
        self.closed = False

    def answer(self, question, passages, on_delta=None, cancel_event=None,
               history=None, request_id="", preferences=None, raw_question=None, interview_packet=None):
        prefs = preferences or self.preferences
        self.calls.append({"question": question, "prefs": replace(prefs),
                           "history": list(history or []), "request_id": request_id, "raw_question":raw_question,
                           "interview_packet":interview_packet})
        if cancel_event and cancel_event.is_set():
            raise AnswerCancelled("已取消回答。")
        if prefs.provider != "ollama" and not prefs.cloud_consent:
            raise ValueError("尚未授权云端。")
        if self.failure:
            raise RuntimeError(self.failure)
        return BilingualAnswer("问题中文", "Suggested English answer.", "中文建议。", [],
                               request_id=request_id, provider=prefs.provider,
                               model=prefs.answer_model or prefs.ollama_model)

    def close(self):
        self.closed = True


@pytest.fixture
def controller(tmp_path, monkeypatch):
    qt = QApplication.instance() or QApplication([])
    monkeypatch.setenv("PRISM_INTERVIEW_HOME", str(tmp_path / "data"))
    monkeypatch.setattr(controller_module, "KnowledgeBase", LocalIndex)
    monkeypatch.setattr(controller_module, "AnswerService", StubAnswerService)
    work, callbacks = [], []

    def defer(self, operation, name="PrismWorker"):
        work.append((name, operation))

    monkeypatch.setattr(controller_module.CopilotController, "background", defer)
    monkeypatch.setattr(controller_module.QTimer, "singleShot",
                        lambda delay, callback: callbacks.append(callback))
    window = CopilotWindow()
    prefs = Preferences(provider="codex", translation_engine="argos", cloud_consent=True, share_evidence=True,
                        speaker_id="speaker", model_cache_dir=str(tmp_path / "models"),
                        caption_speaker_mode="acoustic", interview_phase="presentation")
    ctrl = controller_module.CopilotController(window, prefs)
    ctrl.timer.stop()
    ctrl.auto_timer.stop()
    window.set_devices([("speaker", "Test speaker")], "speaker")
    callbacks.clear()  # Discard startup device discovery; no real hardware access.

    def run_worker():
        _, operation = work.pop(0)
        operation()
        ctrl.pump()

    def run_callbacks():
        for _ in range(100):
            if not callbacks:
                return
            callbacks.pop(0)()
        raise AssertionError("Unbounded Qt callback loop")

    ctrl.test_work = work
    ctrl.test_callbacks = callbacks
    ctrl.test_run_worker = run_worker
    ctrl.test_run_callbacks = run_callbacks
    try:
        yield ctrl
    finally:
        ctrl.close()
        callbacks.clear()
        work.clear()
        window.close()
        window.deleteLater()
        qt.processEvents()


def test_queue_preserves_question_model_and_completed_history(controller):
    ctrl = controller
    ctrl.window.set_settings({"provider": "codex", "answer_model": "model-one"})
    ctrl.request_answer("How do you calibrate?")
    first_identity = ctrl._active
    ctrl.window.set_settings({"provider": "ollama", "ollama_model": "local-two"})
    ctrl.request_answer("Why?")
    assert ctrl.busy and len(ctrl._requests) == 1
    ctrl.test_run_worker()
    assert ctrl.service.calls[0]["prefs"].answer_model == "model-one"
    assert ctrl.service.calls[0]["prefs"].provider == "codex"
    ctrl.test_run_callbacks()
    assert ctrl.busy and ctrl._active != first_identity
    ctrl.test_run_worker()
    second = ctrl.service.calls[1]
    assert second["question"] == "Why?"
    assert second["prefs"].provider == "ollama"
    assert second["prefs"].ollama_model == "local-two"
    assert second["history"] == [{"question": "How do you calibrate?",
                                   "answer_en": "Suggested English answer."}]
    assert len(ctrl._answers) == 2 and not ctrl.busy


def test_cancelled_stream_is_not_left_as_completed_or_generating(controller):
    ctrl = controller
    ctrl.request_answer("How would you estimate uncertainty?")
    identity = ctrl._active
    ctrl.post("answer_delta", (identity, "answer_en", "Incomplete suggestion."))
    ctrl.pump()
    assert ctrl.window.answer_english.text() == "Incomplete suggestion."
    ctrl.cancel_answer()
    ctrl.test_run_worker()
    assert not ctrl.busy and not ctrl._answers and not ctrl._history
    assert ctrl.window._answer_en == ""
    assert ctrl.window.answer_state.text() != "正在生成…"
    assert not ctrl.window.copy_button.isEnabled()


def test_model_check_cannot_replace_scheduled_answer_preferences(controller):
    ctrl = controller
    ctrl.window.set_settings({"answer_model": "intended-model"})
    ctrl.request_answer("Question for the selected model?")
    # Simulate a previously scheduled settings/model checker running late.
    ctrl.service.preferences = Preferences(provider="ollama", ollama_model="unrelated-local-model")
    ctrl.test_run_worker()
    call = ctrl.service.calls[0]
    assert call["prefs"].provider == "codex"
    assert call["prefs"].answer_model == "intended-model"
    assert ctrl._answers[0]["result"].model == "intended-model"


def test_late_events_cannot_overwrite_new_question(controller):
    ctrl = controller
    ctrl.request_answer("First question?")
    first = ctrl._active
    ctrl.request_answer("Second question?")
    ctrl.cancel_answer()
    ctrl.test_run_worker()
    ctrl.test_run_callbacks()
    second = ctrl._active
    ctrl.post("answer_delta", (second, "answer_en", "Current answer."))
    ctrl.post("answer_delta", (first, "answer_en", "STALE"))
    ctrl.post("answer", (first, "First question?", BilingualAnswer("旧问题", "STALE", "旧答案", [])))
    ctrl.post("answer_error", (first, "stale error", False))
    ctrl.pump()
    assert ctrl._active == second and ctrl.busy
    assert ctrl.window.answer_english.text() == "Current answer."
    assert "STALE" not in ctrl.window.answer_english.text()


@pytest.mark.parametrize("revoked", ["cloud_consent", "share_evidence"])
def test_revoking_privacy_cancels_active_and_restricts_queued_requests(controller, revoked):
    ctrl = controller
    ctrl.request_answer("Current cloud question?")
    ctrl.request_answer("Queued cloud question?")
    ctrl.window.set_settings({revoked: False})
    ctrl.save_settings(ctrl.window.get_settings())
    assert ctrl._cancel.is_set()
    assert getattr(ctrl._requests[0][2], revoked) is False
    ctrl.test_run_worker()
    ctrl.test_run_callbacks()
    assert getattr(ctrl.service.preferences, revoked) is False
    ctrl.test_run_worker()
    second = ctrl.service.calls[1]
    assert getattr(second["prefs"], revoked) is False
    if revoked == "cloud_consent":
        assert not ctrl._answers


def test_privacy_change_does_not_cancel_local_answer(controller):
    ctrl = controller
    ctrl.window.set_settings({"provider": "ollama"})
    ctrl.request_answer("A local question?")
    ctrl.window.set_settings({"cloud_consent": False, "share_evidence": False})
    ctrl.save_settings(ctrl.window.get_settings())
    assert not ctrl._cancel.is_set()
    ctrl.test_run_worker()
    assert len(ctrl._answers) == 1


def test_failed_generation_clears_provisional_text_and_continues_queue(controller):
    ctrl = controller
    ctrl.service.failure = "connection lost"
    ctrl.request_answer("Current question?")
    ctrl.request_answer("Next question?")
    ctrl.post("answer_delta", (ctrl._active, "answer_en", "Unverified partial"))
    ctrl.pump()
    ctrl.test_run_worker()
    assert not ctrl._answers and ctrl.window._answer_en == ""
    ctrl.service.failure = ""
    ctrl.test_run_callbacks()
    ctrl.test_run_worker()
    assert len(ctrl._answers) == 1
    assert ctrl._answers[0]["question"] == "Next question?"


def test_close_blocks_queued_work_and_late_callbacks(controller):
    ctrl = controller
    ctrl.request_answer("First question?")
    ctrl.request_answer("Queued question?")
    old_identity = ctrl._active
    ctrl.close()
    assert not ctrl._requests and ctrl._cancel.is_set()
    ctrl.test_run_worker()
    ctrl.post("answer_delta", (old_identity, "answer_en", "LATE"))
    ctrl.pump()
    ctrl.request_answer("After close?")
    ctrl.test_run_callbacks()
    assert len(ctrl.service.calls) == 1
    assert not ctrl._answers and "LATE" not in ctrl.window._answer_en


def test_auto_answer_keeps_long_question_and_does_not_repeat_it(controller):
    ctrl = controller
    ctrl.window.set_settings({"auto_answer": True, "interview_phase": "qa"})
    ctrl.save_settings(ctrl.window.get_settings())
    ctrl.listening = True
    ctrl.on_transcript("final", "How would you combine spectral measurements")
    ctrl.on_transcript("final", "with polarization information for inverse rendering?")
    ctrl.assembler.last_text = ctrl.assembler.last_voice = 0
    ctrl.auto_answer()
    assert ctrl.busy
    ctrl.auto_answer()
    assert len(ctrl.test_work) == 1 and not ctrl._requests
    ctrl.test_run_worker()
    assert ctrl.service.calls[0]["question"] == (
        "How would you combine spectral measurements with polarization information for inverse rendering?")


def test_stopped_session_transcripts_and_translations_are_ignored(controller):
    ctrl = controller
    ctrl._session = 7
    ctrl.post("transcript", (6, "final", "Old transcript?"))
    ctrl.post("translation", (6, "old-utterance", "旧译文"))
    ctrl.post("transcript", (7, "final", "Current question?"))
    ctrl.pump()
    assert len(ctrl._transcripts) == 1
    assert ctrl._transcripts[0]["en"] == "Current question?"
    assert ctrl.window.current_question() == "Current question?"


def test_queue_capacity_keeps_controller_responsive(controller):
    ctrl = controller
    ctrl.request_answer("Active question?")
    for index in range(14):
        ctrl.request_answer(f"Queued question {index}?")
    assert ctrl.busy and len(ctrl._requests) == 12
    assert len(ctrl.test_work) == 1
    assert "队列已满" in ctrl.window.status_label.text()


def test_live_chinese_preview_is_revisable_and_final_cannot_be_overwritten(controller):
    ctrl = controller
    ctrl.pref.live_translation=True
    previews, finals = [], []
    ctrl.translation = SimpleNamespace(submit_preview=lambda *args: previews.append(args),
        submit=lambda *args: finals.append(args), stop=lambda: None)
    ctrl.on_transcript("partial", "How does spectral imaging")
    identity = ctrl._partial_id
    assert len(previews)==1 and not ctrl._transcripts and not ctrl.window._entries
    assert ctrl.window.live_caption.toPlainText() == "How does spectral imaging"
    preview_key=previews[0][0]
    ctrl.post("translation_preview", (ctrl._session, preview_key, "How does spectral imaging", "实时中文"))
    ctrl.pump()
    assert not ctrl.window._entries
    assert ctrl._live_data['zh']=='实时中文'
    ctrl.on_transcript("final", "How does spectral imaging help inverse rendering?")
    assert len(finals) == 1
    ctrl.post("translation", (ctrl._session, finals[0][0], "正式中文译文"))
    ctrl.post("translation_preview", (ctrl._session, preview_key, "How does spectral imaging", "延迟到达的预览"))
    ctrl.pump()
    assert ctrl.window._entries[identity].chinese.text() == "正式中文译文"
    assert ctrl._transcripts[0]["zh"] == "正式中文译文"
    assert ctrl._live_data['zh']=='正式中文译文'


def test_partial_updates_do_not_change_confirmed_history(controller):
    ctrl = controller
    ctrl.on_transcript("final", "Could you explain your method?")
    identity = ctrl._transcripts[0]["id"]
    ctrl.on_transcript("partial", "And how")
    ctrl.on_transcript("partial", "And how does it handle noise?")
    assert ctrl.window._entries[identity].english.text() == "Could you explain your method?"
    assert ctrl.window.live_caption.toPlainText().rstrip('?.!').endswith("And how does it handle noise")


def test_short_pause_windows_share_caption_and_stale_translation_is_ignored(controller):
    ctrl = controller
    finals, previews = [], []
    ctrl.translation = SimpleNamespace(submit=lambda *args: finals.append(args),
        submit_preview=lambda *args: previews.append(args), stop=lambda: None)
    ctrl.on_transcript("final", "How would you combine spectral measurements")
    first = ctrl._transcripts[0]["id"]
    old_key = finals[0][0]
    ctrl.on_transcript("partial", "with polarization information")
    assert ctrl._partial_id == first
    assert len(ctrl.window._entries) == 1
    assert ctrl.window.live_caption.toPlainText().endswith("with polarization information")
    ctrl.post("translation", (ctrl._session, old_key, "过时的短译文"))
    ctrl.pump()
    # This translation still matches the confirmed history; partial words live
    # in a separate pane and cannot invalidate a confirmed sentence's Chinese.
    assert ctrl.window._entries[first].chinese.text() == "过时的短译文"
    ctrl.on_transcript("final", "with polarization information for inverse rendering?")
    assert len(ctrl._transcripts) == 1
    assert ctrl._transcripts[0]["en"] == (
        "How would you combine spectral measurements with polarization information for inverse rendering?")
    ctrl.post("translation", (ctrl._session, finals[-1][0], "完整问题译文"))
    ctrl.pump()
    assert ctrl._transcripts[0]["zh"] == "完整问题译文"


def test_presentation_cannot_auto_answer_and_entering_qa_is_one_action(controller):
    ctrl = controller
    ctrl.listening = True
    ctrl.window.set_settings({"interview_phase": "presentation", "auto_answer": True})
    ctrl.save_settings(ctrl.window.get_settings())
    ctrl.on_transcript("final", "How does this calibration work?")
    ctrl.assembler.last_text = ctrl.assembler.last_voice = 0
    ctrl.auto_answer()
    assert not ctrl.busy and not ctrl.test_work
    ctrl.window.qa_phase_button.click()
    assert ctrl.pref.interview_phase == "qa" and ctrl.pref.auto_answer
    assert not ctrl.assembler.text  # A presentation question cannot leak into QA.
    ctrl.on_transcript("final", "Why would that be identifiable?")
    ctrl.assembler.last_text = ctrl.assembler.last_voice = 0
    ctrl.auto_answer()
    assert ctrl.busy and len(ctrl.test_work) == 1
    ctrl.window.presentation_phase_button.click()
    assert ctrl._cancel.is_set()


def test_new_automatic_question_replaces_obsolete_automatic_queue(controller):
    ctrl = controller
    ctrl.change_interview_phase("qa")
    ctrl.request_answer("First automatic question?", automatic=True)
    ctrl.request_answer("Second automatic question?", automatic=True)
    ctrl.request_answer("Newest automatic question?", automatic=True)
    assert ctrl._cancel.is_set()
    assert [item[1] for item in ctrl._requests] == ["Newest automatic question?"]
    ctrl.test_run_worker()
    ctrl.test_run_callbacks()
    ctrl.test_run_worker()
    assert len(ctrl._answers) == 1 and ctrl._answers[0]["question"] == "Newest automatic question?"


def test_non_question_is_not_prepended_to_next_automatic_question(controller):
    ctrl = controller
    ctrl.change_interview_phase("qa")
    ctrl.listening = True
    ctrl.on_transcript("final", "We will now move to the next topic.")
    ctrl.assembler.last_text = ctrl.assembler.last_voice = 0
    ctrl.auto_answer()
    assert not ctrl.assembler.text
    ctrl.on_transcript("final", "How do you calibrate?")
    ctrl.assembler.last_text = ctrl.assembler.last_voice = 0
    ctrl.auto_answer()
    ctrl.test_run_worker()
    assert ctrl.service.calls[0]["question"] == "How do you calibrate?"


def test_stop_drains_last_audio_final_and_translation_before_invalidating_session(controller):
    ctrl = controller
    stopped = []
    original_session = ctrl._session
    ctrl.audio = SimpleNamespace(stop=lambda: None, thread=None)
    def speech_stop(drain=False):
        stopped.append(("speech", drain))
        if drain:
            ctrl.post("transcript", (original_session, "final", "The last calibration question?"))
    ctrl.speech = SimpleNamespace(stop=speech_stop, thread=None)
    ctrl.translation = SimpleNamespace(pending=0,
        submit=lambda key, text: ctrl.post("translation", (original_session, key, "最后一句译文")),
        stop=lambda: stopped.append(("translation", True)))
    ctrl.stop_listening()
    assert ctrl.stopping and ctrl._session == original_session
    ctrl.test_run_worker()
    assert ctrl._transcripts[0]["en"] == "The last calibration question?"
    assert ctrl._transcripts[0]["zh"] == "最后一句译文"
    ctrl.test_run_worker()
    assert not ctrl.stopping and ctrl._session > original_session
    assert stopped[0] == ("speech", True)


def test_cancel_model_preparation_before_capture_ignores_late_ready(controller, monkeypatch):
    ctrl = controller
    stopped = []
    monkeypatch.setattr(ctrl, "_make_translator", lambda *_: SimpleNamespace(ready=True))
    monkeypatch.setattr(controller_module, "TranslationWorker", lambda *a, **k:
        SimpleNamespace(start=lambda: None, stop=lambda: stopped.append("translation")))
    monkeypatch.setattr(controller_module, "SpeechWorker", lambda *a, **k:
        SimpleNamespace(start=lambda: None, stop=lambda: stopped.append("speech")))
    ctrl.start_listening()
    old_session = ctrl._session
    assert ctrl.preparing and ctrl.window.stop_button.isEnabled()
    assert not ctrl.window.start_button.isEnabled()
    ctrl.window.stop_button.click()
    assert not ctrl.preparing and not ctrl.listening and not ctrl.stopping
    assert ctrl._session > old_session
    assert "speech" in stopped and "translation" in stopped
    assert not ctrl.window.stop_button.isEnabled()
    ctrl.post("asr_ready", old_session)
    ctrl.post("translation_ready", old_session)
    ctrl.post("restoration_ready", old_session)
    ctrl.post("capture_ready", old_session)
    ctrl.pump()
    assert not ctrl.listening and not ctrl._capture_started


def test_cancelled_gemma_preparation_blocks_restart_until_old_workers_finish(controller, monkeypatch):
    ctrl = controller
    work = {}
    joined = []
    class FakeThread:
        def __init__(self, name): self.name = name
        def join(self): joined.append(self.name)
    def background(operation, name="PrismWorker"):
        work[name] = operation
        return FakeThread(name)
    translator = SimpleNamespace(ready=True, install_english_chinese=lambda *a: None,
                                 translate=lambda text: "翻译")
    monkeypatch.setattr(ctrl, "background", background)
    monkeypatch.setattr(ctrl, "_make_translator", lambda *_: translator)
    monkeypatch.setattr(controller_module, "TranslationWorker", lambda *a, **k:
        SimpleNamespace(start=lambda: None, stop=lambda: None))
    speech = SimpleNamespace(start=lambda: None, stop=lambda: None, streaming=None)
    class SpeechThread(FakeThread):
        def join(self):
            super().join()
            # Reproduce a child created between stop and speech-thread exit.
            speech.streaming = SimpleNamespace(thread=FakeThread("streaming"))
    speech.thread = SpeechThread("speech")
    monkeypatch.setattr(controller_module, "SpeechWorker", lambda *a, **k: speech)
    ctrl.window.set_settings({"translation_engine": "gemma"})
    ctrl.start_listening()
    old_session = ctrl._session
    assert "PrismPrepareMT" in work
    ctrl.window.stop_button.click()
    assert ctrl._cancelled_preparation_pending and not ctrl.window.start_button.isEnabled()
    ctrl.start_listening()
    assert ctrl._session > old_session and not ctrl.preparing
    work["PrismPrepareMT"]()  # Late Gemma warm-up completion from cancelled session.
    ctrl.pump()
    assert not ctrl._capture_started and not ctrl.listening
    work["PrismCancelledPreparation"]()
    ctrl.pump()
    assert {"PrismPrepareText", "PrismPrepareMT", "speech", "streaming"} <= set(joined)
    assert joined.index("speech") < joined.index("streaming")
    assert not ctrl._cancelled_preparation_pending and ctrl.window.start_button.isEnabled()


def test_rejected_final_removes_unconfirmed_card_and_blocks_auto_guess(controller):
    ctrl = controller
    ctrl.change_interview_phase("qa")
    ctrl.listening = True
    ctrl.on_transcript("partial", "An unconfirmed question?")
    assert ctrl._partial_id is not None
    ctrl.on_transcript("rejected", "")
    assert ctrl._partial_id is None and not ctrl.window._entries and not ctrl._transcripts
    assert ctrl._question_uncertain
    ctrl.assembler.last_text = ctrl.assembler.last_voice = 0
    ctrl.auto_answer()  # Retire the incomplete turn at a clear idle boundary.
    assert not ctrl.busy and not ctrl._question_uncertain
    ctrl.on_transcript("final", "Why is the matrix identifiable?")
    ctrl.assembler.last_text = ctrl.assembler.last_voice = 0
    ctrl.auto_answer()
    assert ctrl.busy and not ctrl._question_uncertain


def test_old_phase_final_is_captioned_but_never_joins_new_qa_question(controller):
    ctrl = controller
    old_epoch = ctrl._phase_epoch
    ctrl.change_interview_phase("qa")
    ctrl.listening = True
    ctrl.post("transcript", (ctrl._session, "final", "Old presentation question?",
                             {"phase_epoch": old_epoch, "word_timed": True}))
    ctrl.pump()
    ctrl.assembler.last_text = ctrl.assembler.last_voice = 0
    ctrl.auto_answer()
    assert not ctrl.busy and not ctrl.assembler.text
    assert ctrl._transcripts[0]["en"] == "Old presentation question?"
    ctrl.on_transcript("final", "Actual professor question?",
                       {"phase_epoch": ctrl._phase_epoch, "word_timed": True})
    ctrl.assembler.last_text = ctrl.assembler.last_voice = 0
    ctrl.auto_answer()
    ctrl.test_run_worker()
    assert ctrl.service.calls[0]["question"] == "Actual professor question?"


@pytest.mark.parametrize("has_preview", [True, False])
def test_rejected_middle_never_auto_answers_incomplete_question(controller, has_preview):
    ctrl = controller
    ctrl.change_interview_phase("qa")
    ctrl.listening = True
    ctrl.on_transcript("final", "How do you calibrate the instrument")
    if has_preview:
        ctrl.on_transcript("partial", "under uncertain conditions")
    ctrl.on_transcript("rejected", "")
    ctrl.on_transcript("final", "and validate the material estimates?")
    ctrl.assembler.last_text = ctrl.assembler.last_voice = 0
    ctrl.auto_answer()
    assert not ctrl.busy and not ctrl.test_work and not ctrl.assembler.text


def test_timed_independent_finals_preserve_professor_repeated_noun_phrase(controller):
    ctrl = controller
    context = {"phase_epoch": ctrl._phase_epoch, "word_timed": True}
    ctrl.on_transcript("final", "What is the calibration matrix?", context)
    ctrl.on_transcript("final", "The calibration matrix maps incident Stokes to intensity.", context)
    expected = "What is the calibration matrix? The calibration matrix maps incident Stokes to intensity."
    assert ctrl.assembler.text == expected
    assert ' '.join(ctrl._transcripts[0]["en"].split()) == expected
    assert ctrl._transcripts[0]['raw_en']==expected

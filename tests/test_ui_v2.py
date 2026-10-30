"""Controller-facing GUI regressions; no audio, model, network, or fake inference."""
from __future__ import annotations

import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from PySide6.QtCore import QPoint
from PySide6.QtGui import QFontDatabase
from pathlib import Path

from zoom_copilot.ui import CopilotWindow


class CopilotWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        # Offscreen Windows does not enumerate installed fonts. Use actual
        # glyph metrics in either test order so clipped first lines are found.
        if os.name == "nt":
            for filename in ("segoeui.ttf", "msyh.ttc", "msjh.ttc"):
                path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / filename
                if path.is_file():
                    QFontDatabase.addApplicationFont(str(path))

    def setUp(self):
        self.window = CopilotWindow()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()

    def _eventually(self, condition, timeout_ms=1000):
        """Qt wrapping/layout can take multiple event turns under build load."""
        deadline = time.monotonic() + timeout_ms / 1000
        while not condition() and time.monotonic() < deadline:
            QTest.qWait(10)
        self.assertTrue(condition(), "Qt layout did not settle within the bounded wait")

    def test_programmatic_settings_never_send_save_events(self):
        emitted = []
        self.window.settings_saved.connect(emitted.append)
        self.window.set_devices([("a", "Speakers A"), ("b", "Speakers B")])
        self.window.set_settings({"speaker_id": "b", "auto_answer": True,
                                  "provider": "codex", "cloud_consent": True,
                                  "translation_engine": "nllb",
                                  "top_k": 7, "api_key": "session-only-secret"})
        self.assertEqual(emitted, [])
        settings = self.window.get_settings()
        self.assertEqual(settings["speaker_id"], "b")
        self.assertEqual(settings["top_k"], 7)
        self.assertEqual(settings["translation_engine"], "nllb")
        self.assertEqual(settings["api_key"], "session-only-secret")
        self.assertTrue(settings["auto_answer"])

    def test_answer_signal_contains_current_edited_question(self):
        questions = []
        self.window.answer_requested.connect(questions.append)
        self.window.set_question("  How is the inverse problem constrained?  ")
        self.window.answer_button.click()
        self.assertEqual(questions, ["How is the inverse problem constrained?"])
        self.window.set_question("")
        self.window.answer_button.click()
        self.assertEqual(len(questions), 1)

    def test_partial_and_final_with_same_id_reuse_single_card(self):
        self.window.append_transcript("utterance-1", "How do", partial=True)
        self.window.append_transcript("utterance-1", "How do you calibrate?", "如何标定？", partial=False)
        self.window.update_translation("utterance-1", "你如何进行标定？")
        self.assertEqual(len(self.window._entries), 1)
        item = self.window._entries["utterance-1"]
        self.assertEqual(item.objectName(), "transcriptItem")
        self.assertEqual(item.english.text(), "How do you calibrate?")
        self.assertEqual(item.chinese.text(), "你如何进行标定？")

    def test_untrusted_model_html_is_displayed_as_literal_text(self):
        payload = '<img src="https://not-a-real-model-output.invalid/leak">'
        # Security rendering is checked on a complete field; incomplete legacy
        # streams now hold punctuationless fragments until they finish.
        self.window.set_answer(payload, "中文", sources=[{"name": "paper.pdf", "text": "evidence"}])
        self.assertEqual(payload, self.window.answer_english.text())
        self.assertEqual(self.window.answer_english.textFormat().name, "PlainText")
        self.assertTrue(self.window.copy_button.isEnabled())
        self.window._copy_answer()
        self.assertEqual(QApplication.clipboard().text(), payload)
        self.window.set_answer("next delta", sources=None)
        self.assertEqual(len(self.window._sources), 1)
        self.window.set_answer("new response", sources=[])
        self.assertEqual(len(self.window._sources), 0)
        self.assertFalse(self.window.source_button.isEnabled())

    def test_busy_and_listening_controls_are_independent(self):
        self.window.set_devices([("a", "Speakers A")])
        self.window.set_listening(True)
        self.window.set_busy(True)
        self.assertFalse(self.window.start_button.isEnabled())
        self.assertTrue(self.window.stop_button.isEnabled())
        self.assertFalse(self.window.answer_button.isEnabled())
        self.assertFalse(self.window.device_select.isEnabled())
        self.window.set_busy(False)
        self.assertTrue(self.window.answer_button.isEnabled())
        self.assertTrue(self.window.stop_button.isEnabled())
        self.window.set_listening(False)
        self.assertTrue(self.window.start_button.isEnabled())
        self.assertFalse(self.window.stop_button.isEnabled())

    def test_manually_entered_model_does_not_reuse_stale_item_data(self):
        self.window.ollama_select.setEditText("my-private-model:latest")
        self.window.whisper_select.setEditText("large-v3-turbo")
        self.assertEqual(self.window.get_settings()["ollama_model"], "my-private-model:latest")
        self.assertEqual(self.window.get_settings()["whisper_size"], "large-v3-turbo")

    def test_short_window_continuous_bilingual_reader_never_overlaps(self):
        self.window.resize(1120, 720)
        self.window.set_answer("Research answer. " * 100, "中文回答。" * 100,
                               question_zh="如何约束逆问题？")
        self.window.show()
        self.app.processEvents()
        pair = self.window._answer_pairs[0]
        self.assertLessEqual(pair.english.geometry().bottom(), pair.chinese.geometry().top())
        self.assertIs(self.window.answer_chinese_scroll, self.window.answer_scroll)
        self.window.chinese_toggle.setChecked(False)
        self.app.processEvents()
        self.assertFalse(pair.chinese.isVisible())
        self.assertTrue(pair.english.isVisible())
        self.assertTrue(self.window.answer_scroll.isVisible())

    def test_prepare_translation_applies_selected_engine_before_download(self):
        events = []
        self.window.settings_saved.connect(lambda settings: events.append(settings["translation_engine"]))
        self.window.install_translation_requested.connect(lambda: events.append("download"))
        self.window.translation_engine_select.setCurrentIndex(self.window.translation_engine_select.findData("nllb"))
        self.window._prepare_translation()
        self.assertEqual(events, ["nllb", "download"])

    def test_streamed_answers_start_at_top_and_do_not_follow_the_end(self):
        self.window.resize(1280, 850)
        self.window.show()
        english = "English research sentence. " * 300
        chinese = "中文科研解释。" * 300
        self.window.set_answer(english[:800], streaming=True)
        QTest.qWait(30)
        self.window.set_answer(english, chinese, question_zh="后到达的问题中文释义。", streaming=True)
        self._eventually(lambda: self.window.answer_scroll.verticalScrollBar().maximum() > 0
            and self.window.answer_chinese_scroll.verticalScrollBar().maximum() > 0)
        self.assertGreater(self.window.answer_scroll.verticalScrollBar().maximum(), 0)
        self.assertGreater(self.window.answer_chinese_scroll.verticalScrollBar().maximum(), 0)
        self.assertEqual(self.window.answer_scroll.verticalScrollBar().value(), 0)
        self.assertEqual(self.window.answer_chinese_scroll.verticalScrollBar().value(), 0)
        old_en_max = self.window.answer_scroll.verticalScrollBar().maximum()
        old_zh_max = self.window.answer_chinese_scroll.verticalScrollBar().maximum()
        self.window.set_answer(english + english, chinese + chinese, question_zh="后到达的问题中文释义。", streaming=False)
        self._eventually(lambda: self.window.answer_scroll.verticalScrollBar().maximum() > old_en_max
            and self.window.answer_chinese_scroll.verticalScrollBar().maximum() > old_zh_max)
        self.assertEqual(self.window.answer_scroll.verticalScrollBar().value(), 0)
        self.assertEqual(self.window.answer_chinese_scroll.verticalScrollBar().value(), 0)

    def test_stream_preserves_continuous_manual_reading_anchor(self):
        self.window.resize(1280, 850)
        self.window.show()
        english = "English research sentence. " * 300
        chinese = "中文科研解释。" * 300
        self.window.set_answer(english, chinese, streaming=True)
        self._eventually(lambda: self.window.answer_scroll.verticalScrollBar().maximum() > 0
            and self.window.answer_chinese_scroll.verticalScrollBar().maximum() > 0)
        bar = self.window.answer_scroll.verticalScrollBar()
        bar.setValue(bar.maximum() // 3)
        paragraph = self.window._answer_pairs[0].english
        anchor = paragraph.mapTo(self.window.answer_scroll.viewport(), QPoint(0, 0)).y()
        self.window.set_answer(english + english, chinese + chinese,
                               question_zh="问题释义新出现后，阅读位置也应保留。", streaming=True)
        QTest.qWait(30)
        self.assertEqual(paragraph.mapTo(self.window.answer_scroll.viewport(), QPoint(0, 0)).y(), anchor)

    def _populate_timeline(self):
        self.window.resize(1280, 850)
        self.window.show()
        self.window.history_toggle.setChecked(True)
        for i in range(12):
            self.window.append_transcript(str(i), f"Question {i}: " + "A research sentence. " * 8)
        # Newly inserted labels can momentarily have zero height while the
        # scroll range is already positive. Do not use that provisional range
        # as the baseline for a later translation's height change.
        self._eventually(lambda: self.window.transcript_scroll.verticalScrollBar().maximum() > 100
            and self._timeline_layout_is_settled() and self._latest_timeline_is_visible())
        return self.window.transcript_scroll.verticalScrollBar()

    def _timeline_layout_is_settled(self):
        return (not self.window._timeline_follow_pending
            and not self.window._timeline_position_timer.isActive()
            and not self.window._timeline_finish_timer.isActive()
            and self.window.transcript_scroll.viewport().updatesEnabled())

    def _latest_timeline_is_visible(self):
        latest = self.window._latest_timeline_unit()
        if latest is None:
            return False
        view = self.window.transcript_scroll.viewport()
        if latest.height() > view.height():
            # Natural bottom-follow shows the newest tail of an over-screen
            # message; the full English and Chinese remain in the reader.
            bottom = latest.mapTo(view, QPoint(0, latest.height())).y()
            return 0 < bottom <= view.height()+2
        english_y = latest.english.mapTo(view, QPoint()).y()
        if not 0 <= english_y < view.height():
            return False
        return latest.mapTo(view, QPoint(0, latest.height())).y() <= view.height()+2

    def test_async_translation_follows_latest_by_default(self):
        bar = self._populate_timeline()
        self.assertGreater(bar.maximum(), 0)
        self.assertTrue(self._latest_timeline_is_visible())
        old_maximum = bar.maximum()
        self.window.update_translation("11", "译文异步到达并增加高度。" * 100)
        self._eventually(lambda: bar.maximum() > old_maximum
            and self._timeline_layout_is_settled() and self._latest_timeline_is_visible())
        self.assertGreater(bar.maximum(), old_maximum)
        self.assertTrue(self._latest_timeline_is_visible())

    def test_async_subtitle_updates_follow_latest_even_after_reading_old_subtitles(self):
        bar = self._populate_timeline()
        bar.setValue(0)
        self.window.update_translation("11", "译文异步到达并增加高度。" * 100)
        self.window.append_transcript("12", "A new research question. " * 20)
        self._eventually(lambda: self._timeline_layout_is_settled()
            and self._latest_timeline_is_visible())
        self.assertTrue(self._latest_timeline_is_visible())

    def test_presentation_phase_preserves_auto_preference_but_disables_auto_control(self):
        phases, saves = [], []
        self.window.interview_phase_changed.connect(phases.append)
        self.window.settings_saved.connect(saves.append)
        self.window.set_settings({"auto_answer": True, "interview_phase": "presentation"})
        self.assertTrue(self.window.auto_checkbox.isChecked())
        self.assertFalse(self.window.auto_checkbox.isEnabled())
        self.assertTrue(self.window.presentation_phase_button.isChecked())
        self.assertEqual(self.window.get_settings()["interview_phase"], "presentation")
        self.assertEqual(phases, [])
        self.assertEqual(saves, [])
        self.window.set_interview_phase("qa")
        self.assertTrue(self.window.auto_checkbox.isEnabled())
        self.assertTrue(self.window.auto_checkbox.isChecked())
        self.assertEqual(phases, [])

    def test_user_qa_switch_emits_phase_then_accepts_controller_auto_setting(self):
        self.window.set_settings({"interview_phase": "presentation", "auto_answer": False})
        phases = []
        def enable_qa(phase):
            phases.append(phase)
            if phase == "qa":
                self.window.set_settings({"auto_answer": True})
        self.window.interview_phase_changed.connect(enable_qa)
        self.window.qa_phase_button.click()
        self.assertEqual(phases, ["qa"])
        self.assertEqual(self.window.get_settings()["interview_phase"], "qa")
        self.assertTrue(self.window.auto_checkbox.isChecked())
        self.window.auto_checkbox.click()
        self.assertFalse(self.window.get_settings()["auto_answer"])
        self.assertEqual(phases, ["qa"])
        self.window.presentation_phase_button.click()
        self.assertEqual(phases, ["qa", "presentation"])
        self.assertFalse(self.window.auto_checkbox.isEnabled())

    def test_cloud_catalog_keeps_blank_default_and_only_uses_real_identifiers(self):
        self.assertEqual(self.window.answer_model_input.count(), 0)
        self.assertEqual(self.window.get_settings()["answer_model"], "")
        self.window.set_answer_models([{"model": "actual-model-a"}, {"id": "actual-model-b"},
                                       {"displayName": "no usable identifier"}, "actual-model-a"])
        self.assertEqual(self.window.answer_model_input.count(), 2)
        self.assertEqual(self.window.get_settings()["answer_model"], "")
        self.window.answer_model_input.setCurrentIndex(1)
        self.assertEqual(self.window.get_settings()["answer_model"], "actual-model-b")
        self.window.set_answer_models(["actual-model-c"])
        self.assertEqual(self.window.get_settings()["answer_model"], "actual-model-b")
        self.window.set_answer_model("Codex / actual-model-b")
        self.assertIn("Codex / actual-model-b", self.window.answer_model_label.text())
        self.window.set_answer_model("")
        self.assertIn("尚未确认", self.window.answer_model_label.text())

    def test_fast_answer_service_is_explicit_and_only_editable_for_codex(self):
        self.window.set_settings({"provider": "codex"})
        self.assertTrue(self.window.answer_speed_select.isEnabled())
        self.assertEqual(self.window.get_settings()["answer_speed"], "standard")
        self.assertIn("2.5", self.window.answer_speed_select.itemText(1))
        self.window.set_settings({"answer_speed": "fast"})
        self.assertEqual(self.window.get_settings()["answer_speed"], "fast")
        self.window.set_settings({"provider": "ollama"})
        self.assertFalse(self.window.answer_speed_select.isEnabled())
        self.window.set_settings({"provider": "codex"})
        self.assertTrue(self.window.answer_speed_select.isEnabled())
        self.assertEqual(self.window.get_settings()["answer_speed"], "fast")


if __name__ == "__main__":
    unittest.main()

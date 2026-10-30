"""Visible conversation/desktop-lyrics behavior using isolated sample text."""
import os
import time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from zoom_copilot.ui import CopilotWindow
from zoom_copilot.overlay import SubtitleOverlay


def application():
    return QApplication.instance() or QApplication([])


def test_acoustic_turns_group_segments_and_keep_translations_addressable():
    app = application()
    w = CopilotWindow()
    try:
        w.append_transcript("one", "How do you model", speaker_id="speaker_1", speaker_label="说话人 1", turn_id="turn-1")
        w.append_transcript("two", "scattering?", speaker_id="speaker_1", speaker_label="说话人 1", turn_id="turn-1")
        w.append_transcript("three", "I use a forward model.", speaker_id="speaker_2", speaker_label="说话人 2", turn_id="turn-2")
        assert w._entries["one"] is w._entries["two"]
        assert w._entries["two"] is not w._entries["three"]
        w.update_translation("one", "你如何建模")
        w.update_translation("two", "散射？")
        assert w._entries["one"].english.text() == "How do you model scattering?"
        assert w._entries["one"].chinese.text() == "你如何建模\n散射？"
        assert w._entries["three"].speaker.text() == "说话人 2"
        w.append_transcript("four", "A new question.", speaker_id="speaker_1", speaker_label="说话人 1", turn_id="turn-3")
        assert w._entries["four"] is not w._entries["one"]
    finally:
        w.close()
        w.deleteLater()
        app.processEvents()


def test_unknown_speaker_is_not_mislabeled_and_very_long_turn_is_bounded():
    app = application()
    w = CopilotWindow()
    try:
        w.append_transcript("a", "Word. " * 200, turn_id="long")
        w.append_transcript("b", "Continuation.", turn_id="long")
        assert w._entries["a"] is not w._entries["b"]
        assert w._entries["a"].speaker.text() == "说话人未确认"
        w.update_translation("a", "旧段译文仍能更新。")
        assert w._entries["a"].chinese.text() == "旧段译文仍能更新。"
        w.remove_transcript("a")
        assert "b" in w._entries
    finally:
        w.close()
        w.deleteLater()
        app.processEvents()


def test_follow_latest_translation_resumes_on_every_update_after_manual_scroll():
    app = application()
    w = CopilotWindow()
    try:
        w.resize(1280, 850)
        w.show()
        w.history_toggle.setChecked(True)
        for i in range(15):
            w.append_transcript(str(i), "A complete English research sentence. " * 8)
        QTest.qWait(40)
        bar = w.transcript_scroll.verticalScrollBar()
        def assert_latest_visible():
            deadline = time.monotonic()+1.5
            while time.monotonic() < deadline:
                QTest.qWait(10)
                if (not w._timeline_follow_pending and not w._timeline_position_timer.isActive()
                        and not w._timeline_finish_timer.isActive()):
                    break
            latest = w._latest_timeline_unit()
            view = w.transcript_scroll.viewport()
            english_y = latest.english.mapTo(view, QPoint()).y()
            if latest.height() <= view.height():
                assert 0 <= english_y < view.height()
                assert latest.mapTo(view, QPoint(0, latest.height())).y() <= view.height()+2
            else:
                assert 0 < latest.mapTo(view, QPoint(0, latest.height())).y() <= view.height()+2
            assert bar.value()==bar.maximum()
        assert w.follow_transcript.text() == "跟随最新对话"
        assert w.follow_transcript.isChecked()
        assert_latest_visible()
        w.update_translation("14", "最后一段的新译文。" * 80)
        QTest.qWait(40)
        assert_latest_visible()
        bar.setValue(bar.maximum() // 3)
        assert bar.value() < bar.maximum()
        w.append_transcript("new", "A different question. " * 20)
        w.update_translation("new", "新的问题。" * 20)
        QTest.qWait(40)
        assert_latest_visible()
        bar.setValue(0)
        w.update_translation("1", "旧消息的异步译文也不会关闭跟随。" * 30)
        QTest.qWait(40)
        assert_latest_visible()
        w.latest_button.click()
        QTest.qWait(40)
        assert_latest_visible()
    finally:
        w.close()
        w.deleteLater()
        app.processEvents()


def test_reading_focus_expands_each_panel_without_losing_text():
    app = application()
    w = CopilotWindow()
    try:
        w.resize(1280, 850)
        w.show()
        w.set_question("A long question. " * 40)
        w.set_answer("An answer. " * 100, "中文答案。" * 100)
        app.processEvents()
        width = w.timeline_panel.width()
        w.reading_mode.setCurrentIndex(1)
        app.processEvents()
        assert w.timeline_panel.width() > width
        assert not w.reply_panel.isVisible()
        assert not w.sidebar.isVisible()
        w.reading_mode.setCurrentIndex(2)
        app.processEvents()
        assert w.reply_panel.isVisible() and not w.timeline_panel.isVisible()
        assert w.answer_english.text().startswith("An answer.")
        w.reading_mode.setCurrentIndex(0)
        app.processEvents()
        assert w.timeline_panel.isVisible() and w.reply_panel.isVisible()
        assert w.current_question().startswith("A long question.")
    finally:
        w.close()
        w.deleteLater()
        app.processEvents()


def test_model_button_emits_and_citations_are_removed_from_all_visible_answers():
    app = application()
    w = CopilotWindow()
    try:
        events = []
        w.model_picker_requested.connect(lambda: events.append(True))
        w.model_picker_button.click()
        assert events == [True]
        w.set_answer("Model scattering [27cfd86b8b24972a6b8c].", "不确定性 [cb2efbe05243cb10d734]。", streaming=True)
        assert "27cfd" not in w.answer_english.text()
        assert "cb2ef" not in w.answer_chinese.text()
        w._copy_answer()
        assert "27cfd" not in QApplication.clipboard().text()
    finally:
        w.close()
        w.deleteLater()
        app.processEvents()


def test_provider_keys_never_cross_services_and_region_and_diarization_roundtrip():
    app = application()
    w = CopilotWindow()
    try:
        w.set_settings({"provider": "deepseek", "api_key": "deepseek-test-key",
                        "api_base_url": "https://api.deepseek.com/v1", "answer_model": "deepseek-flash",
                        "qwen_region": "ap-southeast-1", "speaker_diarization": False})
        w.provider_select.setCurrentIndex(w.provider_select.findData("qwen"))
        assert w.api_key_input.text() == ""
        assert w.api_base_input.text() == ""
        assert w.answer_model_input.text() == ""
        w.api_key_input.setText("qwen-test-key")
        w.provider_select.setCurrentIndex(w.provider_select.findData("openai_compatible"))
        assert w.api_key_input.text() == ""
        w.api_key_input.setText("compatible-test-key")
        w.provider_select.setCurrentIndex(w.provider_select.findData("deepseek"))
        assert w.api_key_input.text() == "deepseek-test-key"
        assert w.api_base_input.text() == "https://api.deepseek.com/v1"
        assert w.answer_model_input.text() == "deepseek-flash"
        settings = w.get_settings()
        assert settings["qwen_region"] == "ap-southeast-1"
        assert settings["speaker_diarization"] is False
        assert settings["session_api_keys"]["qwen"] == "qwen-test-key"
        assert settings["session_api_keys"]["compatible"] == "compatible-test-key"
    finally:
        w.close()
        w.deleteLater()
        app.processEvents()


def test_small_laptop_starts_with_answer_priority_and_retains_full_reader_choice():
    app = application()
    w = CopilotWindow()
    try:
        w.resize(1120, 720)
        w.set_answer("An answer. " * 100, "中文。" * 100)
        w.show()
        app.processEvents()
        assert w.reading_mode.currentData() == "split"
        assert w.reply_panel.width() > w.timeline_panel.width()
        w.reading_mode.setCurrentIndex(2)
        assert w.answer_scroll.height() >= 100
        assert not w.timeline_panel.isVisible()
        w.reading_mode.setCurrentIndex(1)
        assert w.timeline_panel.isVisible() and not w.question_panel.isVisible()
    finally:
        w.close()
        w.deleteLater()
        app.processEvents()


def test_prepare_speaker_model_is_an_explicit_signal_without_network():
    app = application()
    w = CopilotWindow()
    try:
        calls = []
        w.install_speaker_requested.connect(lambda: calls.append("prepare"))
        assert calls == []
        w.prepare_speaker_button.click()
        assert calls == ["prepare"]
    finally:
        w.close()
        w.deleteLater()
        app.processEvents()


def test_lyrics_transparent_modes_paging_and_full_question_survive_updates():
    app = application()
    overlay = SubtitleOverlay()
    try:
        overlay.show()
        overlay.resize(900, 300)
        assert overlay.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        assert overlay._transparent
        overlay.set_caption("A professor asks a question.", "教授在提问。")
        overlay.set_question("The complete question across a long pause?")
        english, chinese = "Research answer. " * 200, "中文解释。" * 200
        overlay.set_answer(english, chinese)
        assert overlay.caption.toPlainText().endswith("教授在提问。")
        overlay.mode.setCurrentIndex(2)
        QTest.qWait(20)
        assert not overlay.caption.isVisible() and overlay.answer.isVisible()
        assert english.strip() in overlay.answer.toPlainText() and chinese in overlay.answer.toPlainText()
        overlay.next_page.click()
        assert overlay.answer.verticalScrollBar().value() > 0
        position = overlay.answer.verticalScrollBar().value()
        overlay.set_answer(english + "More.", chinese)
        assert overlay.answer.verticalScrollBar().value() == position
        overlay.mode.setCurrentIndex(3)
        assert "complete question" in overlay.caption.toPlainText()
        overlay.set_transparent(False)
        assert not overlay._transparent
        overlay.set_font_size(100)
        assert overlay._font_size == 44
        overlay.move_to_bottom()
        assert overlay._anchor == "bottom"
    finally:
        overlay.hide()
        overlay.deleteLater()
        app.processEvents()

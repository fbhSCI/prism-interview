"""Real Qt presentation regressions for paired sources and optional following."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from zoom_copilot.ui import CopilotWindow
from zoom_copilot.overlay import SubtitleOverlay


def application():
    return QApplication.instance() or QApplication([])


def test_translation_units_stay_interleaved_by_source_id_when_sentence_counts_differ():
    app = application()
    window = CopilotWindow()
    try:
        window.show()
        # Two English sentences can legitimately become one Chinese sentence.
        window.append_transcript("a", "We calibrate M. We then validate it.", "标定 M 后再进行验证。",
            turn_id="voice", speaker_label="面试官")
        window.append_transcript("b", "What did you observe?", "你观察到了什么？", turn_id="voice")
        QTest.qWait(30)
        card = window._entries["a"]
        assert card is window._entries["b"]
        first, second = card._unit_widgets.values()
        assert first.english.text() == "We calibrate M. We then validate it."
        assert first.chinese.text() == "标定 M 后再进行验证。"
        assert second.english.text() == "What did you observe?"
        assert second.chinese.text() == "你观察到了什么？"
        assert first.english.isVisible() and first.chinese.isVisible()
        assert first.english.y() < first.chinese.y()
        assert first.y() + first.height() <= second.y()
        # A short pair must not create phantom scroll space that clips English
        # above a large empty region when following is enabled.
        if card.height() <= window.transcript_scroll.viewport().height():
            assert window.transcript_scroll.verticalScrollBar().maximum() == 0
        # Revising one unit never assigns its translation to its neighbour.
        window.update_message("a", "We calibrate M. We then validate it on reference targets.")
        assert card._unit_widgets["a"] is first
        assert first.chinese.text() == "标定 M 后再进行验证。"
        assert first.pending.isVisible()
        window.update_translation("a", "标定 M 后，使用参考目标验证。")
        assert not first.pending.isVisible()
        assert second.chinese.text() == "你观察到了什么？"
        assert first.chinese.text() == "标定 M 后，使用参考目标验证。"
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_follow_can_be_disabled_without_losing_reading_anchor_on_old_translation_growth():
    app = application()
    window = CopilotWindow()
    try:
        window.resize(1440, 940)
        window.show()
        for number in range(18):
            window.append_transcript(str(number), f"Source {number}: " + "Research words. " * 20,
                "译文。" * 10, turn_id=str(number))
        # Nested word wrapping is posted by QScrollArea after showing; start
        # the synthetic scroll only once the history has a stable real range.
        for _ in range(5):
            app.processEvents()
            QTest.qWait(20)
        window.follow_transcript.setChecked(False)
        bar = window.transcript_scroll.verticalScrollBar()
        target = window._entries["10"]
        bar.setValue(target.y() + 14)
        QTest.qWait(20)
        offset = target.y() - bar.value()
        window.append_transcript("latest", "A new research question?", "一个新问题？")
        window.update_translation("1", "旧译文增加高度。" * 160)
        QTest.qWait(40)
        assert not window.get_settings()["follow_latest"]
        assert target.y() - bar.value() == offset
        assert bar.value() < bar.maximum()
        # Following resumes deliberately and survives another delayed update.
        window.latest_button.click()
        QTest.qWait(30)
        assert window.get_settings()["follow_latest"]
        latest=window._latest_timeline_unit();view=window.transcript_scroll.viewport()
        assert 0 <= latest.english.mapTo(view,QPoint()).y() < view.height()
        window.update_translation("latest", "新译文。" * 30)
        QTest.qWait(30)
        latest=window._latest_timeline_unit();view=window.transcript_scroll.viewport()
        assert 0 <= latest.english.mapTo(view,QPoint()).y() < view.height()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_each_main_reader_has_a_reversible_large_workspace_with_live_contents():
    app = application()
    window = CopilotWindow()
    try:
        window.resize(1440, 940)
        window.show()
        window.append_transcript("a", "How do you calibrate?", "你如何标定？")
        window.set_question("How do you calibrate and validate your measurements? " * 15)
        window.set_answer("I use a calibrated forward model. " * 10, "使用经过标定的前向模型。" * 10)
        QTest.qWait(30)
        split_width = window.timeline_panel.width()
        window.expand_transcript_button.click()
        QTest.qWait(20)
        assert window.timeline_panel.width() > split_width
        assert window.expand_transcript_button.text() == "停靠"
        assert window.reading_docks["transcript"].isFloating()
        window.expand_transcript_button.click()
        window.expand_answer_button.click()
        QTest.qWait(20)
        assert window.timeline_panel.isVisible()
        assert window._answer_pairs[0].english.isVisible()
        assert window.reading_docks["answer"].isFloating()
        window.expand_answer_button.click()
        window.expand_question_button.click()
        QTest.qWait(20)
        assert window.reading_docks["question"].isFloating()
        assert window.timeline_panel.isVisible()
        assert window.question_panel.height() > 500
        assert window.question_input.isVisible()
        assert window.answer_button.isVisible()
        window.set_question("A later complete question?")
        QTest.qWait(20)
        assert window.question_input.toPlainText() == "A later complete question?"
        window.expand_question_button.click()
        assert window.timeline_panel.isVisible() and window.reply_panel.isVisible()
        assert window.current_question() == "A later complete question?"
        window.set_answer("A reply.", "回复。", sources=[{"text": "Reference excerpt."}])
        window.source_button.click()
        assert window._sources_dialog.windowFlags() & Qt.WindowType.WindowMaximizeButtonHint
    finally:
        if window._sources_dialog:
            window._sources_dialog.close()
        window.close()
        window.deleteLater()
        app.processEvents()


def test_remote_count_and_personal_profile_and_follow_settings_roundtrip_without_mic():
    app = application()
    window = CopilotWindow()
    try:
        assert window.capture_mode_select.currentData() == "single_remote"
        assert window.get_settings()["prefer_prepared_answers"] is True
        window.set_settings({"candidate_profile": "My thesis is on spectral-polarimetric imaging.",
            "follow_latest": False, "caption_speaker_mode": "acoustic"})
        window.prefer_prepared_answers_checkbox.setChecked(False)
        assert not window.get_settings()["prefer_prepared_answers"]
        assert window.capture_mode_select.currentData() == "acoustic"
        assert window.get_settings()["candidate_profile"].startswith("My thesis")
        assert not window.follow_transcript.isChecked()
        events = []
        window.settings_saved.connect(events.append)
        window.capture_mode_select.setCurrentIndex(window.capture_mode_select.findData("single_remote"))
        assert len(events) == 1
        assert events[0]["caption_speaker_mode"] == "single_remote"
        assert "不采集本机麦克风" in window.capture_mode_select.toolTip()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_overlay_uses_explicit_source_pairs_and_optional_follow_and_maximize():
    app = application()
    overlay = SubtitleOverlay()
    try:
        overlay.show()
        overlay.set_caption("First. Second.", "第一。第二。", speaker_label="面试官",
            units=[("First.", "第一。"), ("Second.", "第二。")])
        assert overlay.caption.toPlainText() == "面试官 · 已成句\nFirst.\n第一。\n\nSecond.\n第二。"
        overlay.follow_action.setChecked(False)
        overlay.set_caption("A long source. " * 200, "很长的译文。" * 200)
        overlay.caption.verticalScrollBar().setValue(0)
        overlay.set_caption("A longer source. " * 220, "很长的更新译文。" * 220)
        assert overlay.caption.verticalScrollBar().value() == 0
        overlay.maximize_button.click()
        assert overlay.isMaximized()
        overlay.maximize_button.click()
        assert not overlay.isMaximized()
    finally:
        overlay.close()
        overlay.deleteLater()
        app.processEvents()

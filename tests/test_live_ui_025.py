"""Stable speaker-card presentation, using synthetic display examples."""
import os
import time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication
from PySide6.QtWidgets import QComboBox
from PySide6.QtCore import QTimer, QPoint
from PySide6.QtTest import QTest

from zoom_copilot.ui import CopilotWindow
from zoom_copilot.overlay import SubtitleOverlay


def application():
    return QApplication.instance() or QApplication([])


def test_speaker_cards_are_primary_and_revisable_draft_is_compact_and_optional():
    app = application()
    w = CopilotWindow()
    try:
        w.resize(1440, 920)
        w.show()
        w.set_live_translation("I am not sure which metric to use.", "我不确定该使用哪种指标。",
            "说话人 2", "正在听译", raw="i'm not sure uh if i use any metric", turn_id="t", speaker_id="speaker_2")
        QTest.qWait(30)
        assert w.transcript_stack.isVisible()
        assert w.history_toggle.isChecked()
        assert not w.draft_panel.isVisible()
        assert not w.draft_toggle.isChecked()
        w.draft_toggle.click()
        QTest.qWait(10)
        assert w.draft_panel.isVisible()
        assert w.draft_panel.height() <= 205
        assert w.transcript_stack.height() >= 70
        assert w.live_caption.toPlainText() == "I am not sure which metric to use."
        assert w.live_translation.toPlainText() == "我不确定该使用哪种指标。"
        assert w.live_speaker.text() == "说话人 2"
        assert w.live_sentence_state.text() == "正在听译"
        assert w.live_raw.toPlainText().startswith("i'm not sure uh")
        assert not w.live_raw.isVisible()
        w.live_raw_toggle.click()
        assert w.live_raw.isVisible()
        w.set_live_translation("I am not sure which metric to use. We can inspect confidence.",
            "我不确定该使用哪种指标。可以查看置信度。", "说话人 2", "已成句",
            raw="i'm not sure uh if i use any metric but we can inspect confidence", turn_id="t")
        assert w.live_sentence_state.text() == "已成句"
        assert "uh" in w.live_raw.toPlainText()
        assert "uh" not in w.live_caption.toPlainText()
    finally:
        w.close()
        w.deleteLater()
        app.processEvents()


def test_old_history_refinement_never_replaces_current_speech_or_its_translation():
    app = application()
    w = CopilotWindow()
    try:
        w.append_transcript("a", "An older question?", turn_id="old", raw_en="an older um question")
        w.append_transcript("b", "The latest question?", turn_id="latest", speaker_id="speaker_2")
        w.update_translation("a", "旧问题。")
        w.update_message("a", "An older corrected question?", raw_en="an older um question", turn_id="old")
        assert w.live_caption.toPlainText() == "The latest question?"
        assert not w.live_translation.toPlainText()
        w.update_translation("b", "最新的问题？")
        assert w.live_translation.toPlainText() == "最新的问题？"
        assert w._entries["a"].raw_text.text() == "an older um question"
    finally:
        w.close()
        w.deleteLater()
        app.processEvents()


def test_acoustic_reassignment_updates_all_turn_segments_without_editing_original():
    app = application()
    w = CopilotWindow()
    try:
        w.append_transcript("a", "Got it.", turn_id="pending", speaker_label="说话人待补认", raw_en="Got it.")
        w.update_speaker("pending", "speaker_2", "说话人 2", .82)
        assert w._entries["a"].speaker.text() == "说话人 2"
        assert w.live_speaker.text() == "说话人 2"
        assert w.live_caption.toPlainText() == "Got it."
        assert w._entries["a"].raw_text.text() == "Got it."
        w.set_preparation_status("已准备 12 份资料 · 本地检索就绪")
        assert "12" in w.preparation_status.text()
    finally:
        w.close()
        w.deleteLater()
        app.processEvents()


def test_overlay_live_speaker_and_revisable_status_stay_with_the_caption():
    app = application()
    overlay = SubtitleOverlay()
    try:
        overlay.draft_action.setChecked(True)
        overlay.set_caption("How can you separate illumination?", "如何分离照明？",
            speaker_label="说话人 1", state="正在听译")
        assert overlay.caption.toPlainText().startswith("说话人 1 · 正在听译")
        overlay.set_caption("How can you separate illumination and reflectance?", "如何分离照明与反射特性？",
            speaker_label="说话人 1", state="已成句")
        assert "已成句" in overlay.caption.toPlainText()
        assert "reflectance" in overlay.caption.toPlainText()
    finally:
        overlay.hide()
        overlay.deleteLater()
        app.processEvents()


def test_renaming_a_known_voice_retains_its_acoustic_id_and_future_role():
    app = application()
    w = CopilotWindow()
    try:
        w.append_transcript("a", "What would you measure?", turn_id="t", speaker_id="speaker_2", speaker_label="说话人 2")
        labels, roles = [], []
        w.speaker_assigned.connect(lambda *args: labels.append(args))
        w.speaker_role_changed.connect(lambda *args: roles.append(args))
        def fill():
            dialog = QApplication.activeModalWidget()
            dialog.findChild(QComboBox, "speakerAssignmentChoice").setEditText("面试官")
            dialog.findChild(QComboBox, "speakerAssignmentRole").setCurrentIndex(1)
            dialog.accept()
        QTimer.singleShot(0, fill)
        w._show_speaker_assignment("t")
        assert labels == [("t", "speaker_2", "面试官")]
        assert roles == [("speaker_2", "interviewer")]
        assert w.live_speaker.text() == "面试官"
    finally:
        w.close()
        w.deleteLater()
        app.processEvents()


def test_optional_grammar_refinement_roundtrips_without_changing_live_text():
    app = application()
    w = CopilotWindow()
    try:
        assert w.get_settings()["grammar_refinement"] is False
        w.set_live_translation("The original words stay here.", "原话保留。", raw="the original uh words stay here")
        w.set_settings({"grammar_refinement": True})
        assert w.grammar_refinement_checkbox.isChecked()
        assert w.get_settings()["grammar_refinement"] is True
        assert w.live_caption.toPlainText() == "The original words stay here."
        assert w.live_raw.toPlainText() == "the original uh words stay here"
        assert "后台" in w.grammar_refinement_checkbox.toolTip()
        assert "跳过" in w.grammar_refinement_checkbox.toolTip()
        w.set_settings({"grammar_refinement": False})
        assert not w.grammar_refinement_checkbox.isChecked()
    finally:
        w.close()
        w.deleteLater()
        app.processEvents()


def test_manual_scroll_does_not_cancel_queued_or_future_follow_callbacks():
    app = application()
    w = CopilotWindow()
    try:
        w.resize(1280, 850)
        w.show()
        w.history_toggle.setChecked(True)
        for i in range(15):
            w.append_transcript(str(i), "A longer research sentence. " * 12)
        bar = w.transcript_scroll.verticalScrollBar()
        def latest_visible():
            unit = w._latest_timeline_unit()
            top = unit.mapTo(w.transcript_scroll.viewport(), QPoint()).y()
            bottom = unit.mapTo(w.transcript_scroll.viewport(), QPoint(0, unit.height())).y()
            height = w.transcript_scroll.viewport().height()
            return 0 <= top < height and (unit.height() > height or bottom <= height + 2)
        def wait_latest():
            limit = time.monotonic() + 0.5
            while not latest_visible() and time.monotonic() < limit:
                QTest.qWait(10)
            assert latest_visible()
        # Wrapping many long cards can require more than one Qt layout turn;
        # wait for the overflow this test needs, rather than assuming 40 ms.
        deadline = time.monotonic() + 1.0
        while (bar.maximum() <= 100 or not latest_visible()) and time.monotonic() < deadline:
            QTest.qWait(10)
        assert bar.maximum() > 100
        assert latest_visible()
        # A manual scroll never disables following, including when a queued
        # translation relayout has not run yet.
        w.update_translation("14", "译文增加高度。" * 100)
        assert w._timeline_follow_pending
        bar.setValue(0)
        assert w._timeline_follow_pending
        wait_latest()
        bar.setValue(0)
        w.append_transcript("new", "A new question? " * 20)
        wait_latest()
    finally:
        w.close()
        w.deleteLater()
        app.processEvents()

"""Continuous reading and detachable workspace; synthetic text, no model calls."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from zoom_copilot.ui import CopilotWindow


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    widget = CopilotWindow()
    widget.resize(1366, 768)
    widget.show()
    QTest.qWait(30)
    yield widget
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_fresh_window_starts_qa_auto_without_starting_capture_and_respects_loaded_mode(window):
    assert window.get_settings()["interview_phase"] == "qa"
    assert window.auto_checkbox.isChecked()
    assert not window._listening
    assert not window.stop_button.isEnabled()
    window.set_settings({"interview_phase": "presentation", "auto_answer": False})
    assert window.get_settings()["interview_phase"] == "presentation"
    assert not window.auto_checkbox.isChecked()
    window.set_settings({"topmost": False})
    assert window.get_settings()["interview_phase"] == "presentation"
    assert not window.auto_checkbox.isChecked()


def test_quick_topmost_switch_tracks_settings_and_does_not_change_overlay(window):
    from zoom_copilot.overlay import SubtitleOverlay
    overlay = SubtitleOverlay()
    window.set_settings({"topmost": True})
    assert window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    assert window.topmost_quick_button.isChecked()
    saved = []
    window.settings_saved.connect(saved.append)
    window.topmost_quick_button.click()
    assert not window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    assert not window.topmost_checkbox.isChecked()
    assert saved[-1]["topmost"] is False
    assert overlay.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    window.topmost_quick_button.click()
    assert window.windowFlags() & Qt.WindowType.WindowStaysOnTopHint
    assert saved[-1]["topmost"] is True
    overlay.close()


@pytest.mark.parametrize("size", [(1366, 768), (1920, 1080)])
def test_default_layout_favors_full_height_answer_without_clipped_bilingual_question(window, size):
    window.resize(*size)
    question = "Some components explicitly model motion. " * 20 + "Could a neural network replace them?"
    chinese = "部分组件显式建模运动。" * 25 + "神经网络能够替代它们吗？"
    window.set_answer("A complete response.", "完整回答。", chinese, question_en=question)
    window.restore_reading_layout()
    QTest.qWait(50)
    assert window.reading_docks["answer"].width() > window.reading_docks["transcript"].width()
    assert window.reading_docks["answer"].height() > window.reading_docks["question"].height()
    assert window.answer_scroll.horizontalScrollBar().maximum() == 0
    for label, text in [(window.question_original, question), (window.question_translation, chinese)]:
        assert text in label.text()
        assert label.height() >= label.heightForWidth(label.width())
        assert "…" not in label.text()


@pytest.mark.parametrize("name", ["transcript", "answer", "question"])
def test_each_reader_detaches_resizes_updates_and_redocks_when_closed(window, name):
    window.set_question("Original question?")
    window.set_answer("Original answer.", "原回答。", question_en="Original question?")
    dock = window.reading_docks[name]
    window._float_reading_panel(name)
    QTest.qWait(30)
    assert dock.isFloating() and dock.isVisible()
    assert dock.windowFlags() & Qt.WindowType.WindowMaximizeButtonHint
    dock.resize(960, 700)
    dock.move(80, 60)
    window.set_question("A later question?")
    window.set_answer("Later answer.", "稍后回答。", question_en="A later question?")
    window.append_transcript("late", "New caption.", "新译文。", speaker_label="老师", turn_id="new")
    QTest.qWait(30)
    assert dock.width() == 960 and dock.height() == 700
    assert window._answer_pairs[0].english.text() == "Later answer."
    assert window.current_question() == "A later question?"
    assert window._entries["late"].chinese.text() == "新译文。"
    dock.close()
    QTest.qWait(30)
    assert not dock.isFloating() and dock.isVisible()
    assert window._answer_pairs[0].english.text() == "Later answer."


def test_restore_layout_recovers_hidden_and_floating_readers_without_losing_data(window):
    window.set_answer("Remember this.", "保留此回答。", question_en="Keep this question?")
    window.reading_docks["transcript"].close()
    window._float_reading_panel("answer")
    window.focus_question_button.click()
    QTest.qWait(40)
    assert all(dock.isVisible() and not dock.isFloating() for dock in window.reading_docks.values())
    assert window._answer_pairs[0].english.text() == "Remember this."
    assert "Keep this question?" in window.question_original.text()


def test_mismatched_answer_paragraphs_keep_full_translation_without_false_alignment(window):
    english = "First finding.\n\nSecond finding."
    chinese = "中文合并描述两个结论。"
    window.set_answer(english, chinese, question_en="What did you find?")
    assert [pair.english.text() for pair in window._answer_pairs if pair.english.text()] == ["First finding.", "Second finding."]
    assert [pair.chinese.text() for pair in window._answer_pairs if pair.chinese.text()] == [chinese]
    assert window._answer_pairs[0].chinese.text() == ""
    window.chinese_toggle.setChecked(False)
    assert window.answer_scroll.isVisible()
    assert not any(pair.chinese.isVisible() for pair in window._answer_pairs)


def test_late_translation_and_completion_keep_visible_paragraph_anchor_and_widget(window):
    english = ("First research paragraph. " * 60) + "\n\n" + ("Second explanation. " * 60)
    chinese = "第一个中文段落。" * 80 + "\n\n" + "第二个中文段落。" * 80
    window.set_answer(english, streaming=True, question_en="How is this validated?", request_id="test")
    QTest.qWait(40)
    paragraph = window._answer_pairs[1].english
    bar = window.answer_scroll.verticalScrollBar()
    bar.setValue(paragraph.mapTo(window.answer_scroll.widget(), QPoint(0, 0)).y())
    anchor = paragraph.mapTo(window.answer_scroll.viewport(), QPoint(0, 0)).y()
    window.set_answer(english, chinese, "如何验证？" * 60, streaming=True, request_id="test")
    QTest.qWait(40)
    assert window._answer_pairs[1].english is paragraph
    assert paragraph.mapTo(window.answer_scroll.viewport(), QPoint(0, 0)).y() == anchor
    window.set_answer(english, chinese, "如何验证？" * 60, streaming=False, request_id="test")
    QTest.qWait(40)
    assert paragraph.mapTo(window.answer_scroll.viewport(), QPoint(0, 0)).y() == anchor
    rows = [(pair.english.text(), pair.chinese.text()) for pair in window._answer_pairs]
    window.set_answer("Another answer.", "另一个答案。", question_en="New question?", request_id="next")
    window.answer_new_button.click()
    window.answer_history_select.setCurrentIndex(0)
    QTest.qWait(40)
    assert rows == [(pair.english.text(), pair.chinese.text()) for pair in window._answer_pairs]
    assert paragraph.mapTo(window.answer_scroll.viewport(), QPoint(0, 0)).y() == anchor


def test_floating_readers_hide_with_workspace_and_reappear_on_restore(window):
    window._float_reading_panel("answer")
    dock = window.reading_docks["answer"]
    window.hide()
    QTest.qWait(30)
    assert not dock.isVisible()
    window.show()
    QTest.qWait(30)
    assert dock.isVisible() and dock.isFloating()
    window._navigate(1)
    assert not dock.isVisible()
    window._navigate(0)
    assert dock.isVisible()

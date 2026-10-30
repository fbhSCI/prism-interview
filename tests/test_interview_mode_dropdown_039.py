"""Compact anchored mode selection, without moving any reading panels."""
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
    value = CopilotWindow()
    value.resize(1366, 768)
    value.show()
    QTest.qWait(30)
    yield value
    value.interview_controls_panel.hide()
    value.close()
    value.deleteLater()
    app.processEvents()


def open_dropdown(window):
    window.focus_controls_button.click()
    QTest.qWait(30)
    return window.interview_controls_panel


def test_mode_dropdown_is_compact_below_button_and_does_not_move_reading_panels(window):
    before = {key: dock.geometry() for key, dock in window.reading_docks.items()}
    panel = open_dropdown(window)
    button = window.focus_controls_button
    below = button.mapToGlobal(QPoint(0, button.height()))
    assert panel.isVisible()
    assert panel.windowFlags() & Qt.WindowType.Popup
    assert panel.width() == 280
    assert panel.height() < 260
    screen = button.screen().availableGeometry()
    assert panel.pos().x() == max(screen.left(), min(below.x(), screen.right() - panel.width() + 1))
    assert panel.pos().y() == below.y() + 6
    assert window.pages.widget(0).layout().indexOf(panel) == -1
    assert before == {key: dock.geometry() for key, dock in window.reading_docks.items()}
    window.focus_controls_button.click()
    assert not panel.isVisible()


def test_dropdown_phase_selection_emits_once_saves_and_closes(window):
    window.set_settings({"interview_phase": "presentation", "auto_answer": True})
    events, settings = [], []
    window.interview_phase_changed.connect(events.append)
    window.settings_saved.connect(settings.append)
    panel = open_dropdown(window)
    assert window.inline_presentation_button.isChecked()
    assert not window.dropdown_auto_checkbox.isEnabled()
    window.inline_qa_button.click()
    assert events == ["qa"]
    assert settings[-1]["interview_phase"] == "qa"
    assert not panel.isVisible()
    assert window.inline_qa_button.isChecked()
    assert window.auto_checkbox.isEnabled()
    assert window.dropdown_auto_checkbox.isEnabled()
    open_dropdown(window)
    window.inline_qa_button.click()
    assert events == ["qa"]
    assert window.inline_qa_button.isChecked()


def test_popup_auto_toggle_tracks_existing_control_and_saved_preferences(window):
    assert window.auto_checkbox.isChecked()
    settings = []
    window.settings_saved.connect(settings.append)
    open_dropdown(window)
    window.dropdown_auto_checkbox.click()
    assert not window.auto_checkbox.isChecked()
    assert not window.get_settings()["auto_answer"]
    assert settings[-1]["auto_answer"] is False
    assert "手动回答" in window.focus_controls_button.text()
    window.auto_checkbox.setChecked(True)
    assert window.dropdown_auto_checkbox.isChecked()
    assert "自动辅助" in window.focus_controls_button.text()


def test_keyboard_escape_and_phase_selection_work_without_a_mouse(window):
    panel = open_dropdown(window)
    assert QApplication.focusWidget() is window.inline_qa_button
    QTest.keyClick(window.inline_qa_button, Qt.Key.Key_Escape)
    assert not panel.isVisible()
    open_dropdown(window)
    window.inline_presentation_button.setFocus()
    QTest.keyClick(window.inline_presentation_button, Qt.Key.Key_Space)
    assert not panel.isVisible()
    assert window.get_settings()["interview_phase"] == "presentation"
    assert window.inline_presentation_button.isChecked()


def test_dropdown_footer_preserves_answer_and_next_signals_and_closes(window):
    from PySide6.QtWidgets import QPushButton
    window.set_question("How is the inverse problem constrained?")
    answers, next_questions = [], []
    window.answer_requested.connect(answers.append)
    window.new_question_requested.connect(lambda: next_questions.append(True))
    panel = open_dropdown(window)
    buttons = {button.text(): button for button in panel.findChildren(QPushButton)}
    buttons["生成当前回答"].click()
    assert answers == ["How is the inverse problem constrained?"]
    assert not panel.isVisible()
    open_dropdown(window)
    buttons["下一题"].click()
    assert next_questions == [True]
    assert not panel.isVisible()


def test_loading_settings_and_silent_phase_updates_keep_dropdown_in_sync(window):
    events = []
    window.interview_phase_changed.connect(events.append)
    window.set_settings({"interview_phase": "presentation", "auto_answer": False})
    assert not window.dropdown_auto_checkbox.isEnabled()
    assert not window.dropdown_auto_checkbox.isChecked()
    window.set_interview_phase("qa")
    assert events == []
    assert window.inline_qa_button.isChecked()
    assert window.dropdown_auto_checkbox.isEnabled()
    assert "手动回答" in window.focus_controls_button.text()


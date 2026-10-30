"""Default answer follow: wait for words, retain request bindings and history."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from zoom_copilot.ui import CopilotWindow


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    widget = CopilotWindow()
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    app.processEvents()


def first_answer(window):
    window.set_answer("First English answer.", "第一题中文回答。", "第一题中文问题？",
                      question_en="First English question?", request_id="one",
                      sources=[{"name": "first.pdf", "text": "First source."}])
    window.set_answer_model("First model")


def pending_second(window):
    window.set_answer(question_en="Second English question?", streaming=True, request_id="two", sources=[])
    window.set_answer_model("Second model")


def test_follow_is_enabled_by_default_but_empty_request_retains_all_first_answer_fields(window):
    assert window.follow_answers_checkbox.isChecked()
    assert "默认开启" in window.follow_answers_checkbox.toolTip()
    first_answer(window)
    pending_second(window)
    assert window._answer_displayed_id == "one"
    assert window.answer_english.text() == "First English answer."
    assert window.answer_chinese.text() == "第一题中文回答。"
    assert "First English question?" in window.question_original.text()
    assert "第一题中文问题？" in window.question_translation.text()
    assert window._sources[0]["name"] == "first.pdf"
    assert "First model" in window.answer_model_label.text()


@pytest.mark.parametrize("english,chinese", [(" \n", "\t"), ("[27cfd86b8b24972a6b8c]", ""), ("[27cfd86", "")])
def test_empty_or_hidden_stream_tokens_do_not_count_as_answer_words(window, english, chinese):
    first_answer(window)
    pending_second(window)
    window.set_answer(english, chinese, streaming=True, request_id="two")
    assert window._answer_displayed_id == "one"
    window.follow_answers_checkbox.setChecked(False)
    window.follow_answers_checkbox.setChecked(True)
    assert window._answer_displayed_id == "one"


def test_first_new_english_switches_all_bindings_and_later_chinese_stays_with_new_question(window):
    first_answer(window)
    pending_second(window)
    window.set_answer("Second English begins.", streaming=True, request_id="two")
    assert window._answer_displayed_id == "two"
    assert window.answer_english.text() == "Second English begins."
    assert "第一题" not in window.answer_chinese.text()
    assert "Second English question?" in window.question_original.text()
    assert "第一题" not in window.question_translation.text()
    assert window._sources == []
    assert "Second model" in window.answer_model_label.text()
    window.set_answer("Second English begins.", "第二题中文回答。", "第二题中文问题？", request_id="two")
    assert window.answer_chinese.text() == "第二题中文回答。"
    assert "第二题中文问题？" in window.question_translation.text()
    assert window._answer_history["one"]["en"] == "First English answer."


def test_manual_history_pins_then_rechecking_follow_returns_to_latest_valid_answer(window):
    first_answer(window)
    pending_second(window)
    window.set_answer("Second answer.", "第二题回答。", request_id="two", streaming=True)
    window.answer_history_select.setCurrentIndex(0)
    assert not window.follow_answers_checkbox.isChecked()
    window.set_answer("Second answer with more words.", "第二题更多文字。", request_id="two")
    assert window._answer_displayed_id == "one"
    window.follow_answers_checkbox.setChecked(True)
    assert window._answer_displayed_id == "two"
    assert window.answer_english.text() == "Second answer with more words."
    assert window.answer_chinese.text() == "第二题更多文字。"


def test_delayed_old_translation_cannot_reselect_old_request_or_mix_question(window):
    first_answer(window)
    pending_second(window)
    window.set_answer("Second answer.", request_id="two", streaming=True)
    window.set_answer("First English answer.", "第一题晚到的修订译文。", "第一题中文问题？", request_id="one")
    assert window._answer_active_id == window._answer_displayed_id == "two"
    assert window.answer_english.text() == "Second answer."
    assert "第一题" not in window.answer_chinese.text()
    assert "Second English question?" in window.question_original.text()
    window.answer_history_select.setCurrentIndex(0)
    assert window.answer_chinese.text() == "第一题晚到的修订译文。"


def test_chinese_first_is_valid_and_cancellation_restores_previous_completed_answer(window):
    first_answer(window)
    pending_second(window)
    window.set_answer("", "第二题先到的中文。", "第二题问题？", request_id="two", streaming=True)
    assert window._answer_displayed_id == "two"
    assert window.answer_english.text() == ""
    assert "Second English question?" in window.question_original.text()
    window.set_answer(request_id="two", cancelled=True)
    assert window._answer_displayed_id == "one"
    assert window.answer_english.text() == "First English answer."

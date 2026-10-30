"""Answer question snapshots remain complete, bilingual and bound while listening."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt, QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPlainTextEdit

from zoom_copilot.ui import CopilotWindow


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    widget = CopilotWindow()
    widget.resize(1280, 850)
    widget.show()
    app.processEvents()
    yield widget
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_stream_question_is_bound_to_answer_not_new_live_question(window):
    original = "Would a fully end-to-end network be a good alternative to motion modeling?"
    window.set_question(original)
    window.set_answer(question_en=original, sources=[], streaming=True)
    window.set_question("How did you calibrate your ghost imaging system?")
    window.set_answer("I would compare physical consistency.", "我会比较物理一致性。",
                      "完全端到端网络能否替代运动建模？", streaming=True)
    assert original in window.question_original.full_text
    assert "ghost imaging" not in window.question_original.full_text
    assert "完全端到端" in window.question_translation.full_text


def test_laptop_reading_layout_shows_question_and_can_restore_closed_dock(window):
    window.resize(1366, 768)
    window.reading_mode.setCurrentIndex(0)
    question = "Should we ignore explicit motion modeling and use a neural network instead?"
    window.set_question(question)
    window.set_answer("A sufficiently detailed research answer. " * 60, "有依据的科研回答。" * 60,
                      "应该用神经网络替代显式运动建模吗？", question_en=question)
    QTest.qWait(40)
    assert window.question_panel.isVisible()
    assert window.focus_controls_button.isVisible()
    assert window.focus_question_button.isVisible()
    assert window.answer_scroll.height() >= window.answer_english.fontMetrics().lineSpacing() * 3
    assert window.answer_chinese_scroll.height() >= window.answer_chinese.fontMetrics().lineSpacing() * 3
    window.reading_docks["question"].close()
    assert not window.question_panel.isVisible()
    window.focus_question_button.click()
    QTest.qWait(40)
    assert window.question_panel.isVisible()
    assert window.current_question() == question
    assert window.answer_button.isVisible()
    assert window.current_question() == question


def test_complete_question_uses_frozen_background_and_actual_ask(window):
    background = "Alex, one question about your last work. Some parts use neural networks. "
    actual_ask = "Should we completely ignore motion modeling and have a network hallucinate the video?"
    window.set_answer(question_en=background + actual_ask, question_preview_en=actual_ask, streaming=True)
    window.set_question("A newer live question?")
    window.set_answer("The model must be assessed.", question_zh="是否应该让网络完全替代运动建模？",
                      streaming=True)
    assert actual_ask in window.question_original.full_text
    assert background.strip() not in window.question_original.text()
    assert background.strip() in window.answer_question_background.text()
    assert window.answer_question_background_card.isVisible()
    window.answer_question_expand.click()
    dialog = window._answer_question_dialog
    assert dialog.findChild(QPlainTextEdit, "answerQuestionEnglishFull").toPlainText() == background + actual_ask
    dialog.close()


def test_long_question_grows_naturally_without_truncation_or_required_expansion(window):
    original = ("A neural renderer can synthesize videos from still images. " * 20
                + "Should we ignore the explicit motion model?")
    translation = "神经渲染可以从静态图像合成视频。" * 40 + "我们应该忽略显式运动模型吗？"
    window.set_answer("A research answer. " * 100, "科研回答。" * 100,
                      translation, question_en=original)
    QTest.qWait(40)
    assert window.answer_question_card.height() > 140
    assert original in window.question_original.text()
    assert translation in window.question_translation.text()
    assert "…" not in window.question_original.text()
    assert "…" not in window.question_translation.text()
    assert window.question_original.height() >= window.question_original.heightForWidth(window.question_original.width())
    assert window.question_translation.height() >= window.question_translation.heightForWidth(window.question_translation.width())
    assert window.answer_scroll.height() > 30
    assert window.answer_chinese_scroll.height() > 20
    window.answer_question_expand.click()
    dialog = window._answer_question_dialog
    assert dialog is not None and dialog.isVisible()
    assert dialog.findChild(QPlainTextEdit, "answerQuestionEnglishFull").toPlainText() == original
    assert dialog.findChild(QPlainTextEdit, "answerQuestionChineseFull").toPlainText() == translation
    assert dialog.windowFlags() & Qt.WindowType.WindowMaximizeButtonHint
    dialog.close()


def test_late_translation_grows_full_question_and_preserves_visible_answer_anchor(window):
    window.set_answer("A research answer. " * 300, question_en="Is an end-to-end network sufficient?",
                      streaming=True)
    QTest.qWait(40)
    initial_height = window.answer_question_card.height()
    bar = window.answer_scroll.verticalScrollBar()
    bar.setValue(bar.maximum() // 2)
    paragraph = window._answer_pairs[0].english
    initial_anchor = paragraph.mapTo(window.answer_scroll.viewport(), QPoint(0, 0)).y()
    window.set_answer("A research answer. " * 300, question_zh="端到端网络足够吗？" * 90,
                      streaming=True)
    QTest.qWait(40)
    assert window.answer_question_card.height() > initial_height
    assert paragraph.mapTo(window.answer_scroll.viewport(), QPoint(0, 0)).y() == initial_anchor


def test_expanded_question_receives_only_translation_for_its_own_snapshot(window):
    first = "Does a network replace motion modeling?"
    window.set_answer(question_en=first, streaming=True)
    window.answer_question_expand.click()
    dialog = window._answer_question_dialog
    reader = dialog.findChild(QPlainTextEdit, "answerQuestionChineseFull")
    window.set_question("What is your next research direction?")
    window.set_answer("First answer.", question_zh="网络能否替代运动建模？", streaming=True)
    assert reader.toPlainText() == "网络能否替代运动建模？"
    window.set_answer(question_en="What is your next research direction?", streaming=True)
    window.set_answer("Second answer.", question_zh="下一步研究方向是什么？", streaming=True)
    assert reader.toPlainText() == "网络能否替代运动建模？"
    assert dialog.findChild(QPlainTextEdit, "answerQuestionEnglishFull").toPlainText() == first
    dialog.close()


def test_cancellation_clears_question_and_new_request_does_not_reuse_translation(window):
    window.set_answer("Answer.", "回答。", "旧问题？", question_en="Old question?")
    window.set_answer(sources=[], streaming=False, question_en="")
    assert not window.answer_question_card.isVisible()
    assert window.answer_stack.currentIndex() == 0
    window.set_answer(question_en="New question?", sources=[], streaming=True)
    assert "旧问题" not in window.question_translation.full_text
    assert "New question?" in window.question_original.full_text


def test_question_html_is_literal_text_in_compact_and_expanded_reader(window):
    payload = '<img src="https://untrusted.invalid/collect"> What is the motion model?'
    window.set_answer("Answer.", question_en=payload, question_zh="如何建模？")
    assert window.question_original.textFormat() == Qt.TextFormat.PlainText
    window.answer_question_expand.click()
    dialog = window._answer_question_dialog
    assert dialog.findChild(QPlainTextEdit, "answerQuestionEnglishFull").toPlainText() == payload
    dialog.close()

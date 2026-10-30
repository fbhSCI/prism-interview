"""Current Question stays uncluttered while model/answer question data survives."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import QApplication

from zoom_copilot.ui import CopilotWindow


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    value = CopilotWindow()
    value.resize(1366, 768)
    value.show()
    app.processEvents()
    yield value
    value.close()
    value.deleteLater()
    app.processEvents()


@pytest.mark.parametrize("semantic", [False, True])
def test_current_question_omits_chinese_without_removing_source_or_answer_translation(window, semantic):
    question = "How do you validate the reconstructed Stokes components?"
    translation = "你如何验证重建的斯托克斯分量？"
    raw = "How do you, um, validate the reconstructed Stokes components?"
    packet = {"question_en": question, "question_zh": translation, "raw_question_en": raw,
        "turns": [{"label": "面试官 A", "speaker_id": "a", "role": "interviewer",
                   "clean_en": question}], "subquestions": [question]}
    if semantic:
        packet.update(background_en="The illumination changes between measurements.",
                      answer_question_en=question)
    window.set_question_packet(packet)
    text = window.question_packet_view.toPlainText()
    assert question in text
    assert "面试官 A · 提问" in text
    assert translation not in text and "中文问题" not in text
    assert window.current_question() == question
    assert window._question_packet == packet
    assert packet["question_zh"] == translation

    window.set_answer("Check calibration and independent measurements.", "核查标定与独立测量。",
                      question_en=question, question_zh=translation, request_id="stokes")
    assert question in window.question_original.text()
    assert translation in window.question_translation.text()
    assert translation == window._answer_history["stokes"]["question_zh"]
    assert window._question_packet["raw_question_en"] == raw


def test_late_current_question_translation_does_not_repaint_or_reset_reading_selection(window):
    question = "Why use polarization?"
    window.set_question_packet({"question_en": question, "answer_question_en": question})
    reader = window.question_packet_view
    cursor = reader.textCursor()
    cursor.setPosition(reader.toPlainText().index(question))
    cursor.movePosition(QTextCursor.MoveOperation.End, QTextCursor.MoveMode.KeepAnchor)
    reader.setTextCursor(cursor)
    before = reader.toPlainText()
    selected = reader.textCursor().selectedText()
    changes = []
    reader.textChanged.connect(lambda: changes.append(True))

    assert window.set_current_question_translation(question, "为什么使用偏振？")
    assert window._question_packet["question_zh"] == "为什么使用偏振？"
    assert reader.toPlainText() == before
    assert reader.textCursor().selectedText() == selected
    assert not changes

    # A later packet refresh preserves the matched translation internally.
    window.set_question_packet({"question_en": question})
    assert window._question_packet["question_zh"] == "为什么使用偏振？"
    assert "为什么使用偏振？" not in reader.toPlainText()
    window.set_question_packet({"question_en": "What about scattering?"})
    assert not window.set_current_question_translation(question, "不应覆盖新问题。")
    assert "question_zh" not in window._question_packet

"""Show actual asks first without truncating or changing the model's source."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from zoom_copilot.question_packet import QuestionContext
from zoom_copilot.ui import CopilotWindow


@pytest.fixture
def window(tmp_path,monkeypatch):
    monkeypatch.setenv('PRISM_INTERVIEW_HOME',str(tmp_path))
    app=QApplication.instance() or QApplication([])
    w=CopilotWindow();w.resize(1366,850);w.show();QTest.qWait(40)
    yield w
    w.close();w.deleteLater();app.processEvents()


def test_source_setup_moves_below_answer_but_full_source_stays_in_request(window):
    setup='The acquisition uses calibrated spectral channels. '*12
    ask='How do you estimate uncertainty under changing illumination?'
    source=setup+ask
    context=QuestionContext();packet=context.build(source)
    assert packet.question_en==source
    window.set_answer('An answer.','回答。','如何评估不确定性？',question_en=source,
                      question_preview_en=source,request_id='legacy')
    QTest.qWait(40);baseline=window.answer_question_card.height()
    window.set_answer('An answer.','回答。','如何评估不确定性？',question_en=source,
                      question_preview_en=packet.answer_question_en,request_id='focused')
    QTest.qWait(40)
    assert ask in window.question_original.text()
    assert 'calibrated spectral' not in window.question_original.text()
    assert window.answer_question_background.text()==setup.strip()
    assert window.answer_question_background_card.isVisible()
    assert window.answer_question_card.height()<baseline
    assert window.answer_question_background_card.mapTo(window.answer_article,QPoint()).y() > window.answer_paragraphs.mapTo(window.answer_article,QPoint()).y()
    assert window._answer_history['focused']['question_en']==source
    assert window._answer_question_en==source
    assert '含提问背景' in window.question_translation.text()
    assert packet.to_prompt_data()['background_en']==setup.strip()
    assert packet.to_prompt_data()['answer_question_en']==ask
    assert '…' not in window.question_original.text()+window.answer_question_background.text()


def test_real_two_part_choice_and_all_followup_asks_remain_in_foreground(window):
    source=('Consider reconstruction and neural rendering. '
            'Which direction would you prioritize? Or fixing them with neural rendering. '
            'What measurements would you add?')
    packet=QuestionContext().build(source)
    window.set_answer('Answer.',question_en=source,question_preview_en=packet.answer_question_en)
    assert 'Which direction would you prioritize?' in window.question_original.text()
    assert 'Or fixing them with neural rendering.' in window.question_original.text()
    assert 'What measurements would you add?' in window.question_original.text()
    assert window.answer_question_background.text()=='Consider reconstruction and neural rendering.'


def test_unverified_or_partial_focus_cannot_hide_real_asks(window):
    source='What are the limits? How do you calibrate?'
    window.set_answer('Answer.',question_en=source,question_preview_en='What are the limits?')
    assert source in window.question_original.text()
    assert not window.answer_question_background_card.isVisible()
    # Unknown declarative/indirect phrasing remains in full rather than being
    # erased because an extractor could not confidently identify a question.
    unknown='Regarding wavelength-dependent attenuation, your explanation please.'
    window.set_answer('Answer.',question_en=unknown,question_preview_en='attenuation')
    assert unknown in window.question_original.text()
    assert not window.answer_question_background_card.isVisible()


def test_background_and_translation_are_bound_to_selected_answer_history(window):
    first='The instrument measures S0. How do you validate S0?'
    packet=QuestionContext().build(first)
    window.set_answer('First.','第一。','仪器测量S0，如何验证S0？',question_en=first,
                      question_preview_en=packet.answer_question_en,request_id='first')
    window.set_answer('Second.','第二。','如何验证S3？',question_en='How do you validate S3?',request_id='second')
    assert not window.answer_question_background_card.isVisible()
    window.answer_history_select.setCurrentIndex(window.answer_history_select.findData('first'))
    assert 'How do you validate S0?' in window.question_original.text()
    assert 'The instrument measures S0.'==window.answer_question_background.text()
    assert '仪器测量S0' in window.question_translation.text()
    assert window.answer_question_background_card.isVisible()

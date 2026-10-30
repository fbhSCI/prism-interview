"""Persistent answer reading, explicit mode controls and focused bilingual asks."""
import os
from pathlib import Path
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPlainTextEdit

from zoom_copilot.ui import CopilotWindow


@pytest.fixture
def window():
    app=QApplication.instance() or QApplication([])
    previous_font=app.font()
    # Qt's offscreen backend does not discover the Windows fonts automatically;
    # use the same installed fonts as the real desktop for layout assertions.
    for filename in ("segoeui.ttf", "msyh.ttc", "msjh.ttc"):
        path=Path(os.environ.get("WINDIR", "C:/Windows"))/"Fonts"/filename
        if path.exists():
            QFontDatabase.addApplicationFont(str(path))
    app.setFont(QFont("Microsoft YaHei",10))
    value=CopilotWindow()
    value.resize(1366,768)
    value.show()
    app.processEvents()
    yield value
    value.close()
    value.deleteLater()
    app.processEvents()
    app.setFont(previous_font)


def completed(window, key="1", en="A complete first answer.", zh="第一题完整回答。"):
    window.set_answer(en,zh,"第一题问题？",question_en="First question?",request_id=key,
                      sources=[{"name":"first-paper.pdf","text":"First evidence."}])
    window.set_answer_model("First model")


def test_next_request_keeps_completed_answer_question_sources_and_model_visible(window):
    window.follow_answers_checkbox.setChecked(False)
    completed(window)
    window.set_answer(question_en="Second question?",question_preview_en="Second question?",
                      sources=[],streaming=True,request_id="2")
    window.set_answer_model("Second model")
    assert window.answer_english.text()=="A complete first answer."
    assert window.answer_chinese.text()=="第一题完整回答。"
    assert "First question?" in window.question_original.full_text
    assert window._sources[0]["name"]=="first-paper.pdf"
    assert "First model" in window.answer_model_label.text()
    assert window.answer_new_button.isVisible()
    window.set_answer("Second answer streaming.","第二题流式回答。","第二题？",
                      streaming=True,request_id="2")
    assert window.answer_english.text()=="A complete first answer."
    window.answer_new_button.click()
    assert window.answer_english.text()=="Second answer streaming."
    assert "Second question?" in window.question_original.full_text
    assert "Second model" in window.answer_model_label.text()
    assert window._sources==[]
    window.answer_history_select.setCurrentIndex(0)
    assert window.answer_english.text()=="A complete first answer."
    assert window._sources[0]["name"]=="first-paper.pdf"


def test_cancelled_later_request_does_not_erase_completed_answer(window):
    completed(window)
    window.set_answer(question_en="Unfinished next question?",streaming=True,request_id="2")
    window.set_answer(question_en="",sources=[],request_id="2",cancelled=True)
    assert window.answer_english.text()=="A complete first answer."
    assert window.answer_chinese.text()=="第一题完整回答。"
    assert not window.answer_new_button.isVisible()
    assert window._answer_history["1"]["en"]=="A complete first answer."


def test_follow_option_waits_for_answer_text_and_old_answers_remain_available(window):
    completed(window)
    window.follow_answers_checkbox.setChecked(True)
    window.set_answer(question_en="Second question?",streaming=True,request_id="2")
    assert window.answer_english.text()=="A complete first answer."
    window.set_answer("New English.",streaming=True,request_id="2")
    assert window.answer_english.text()=="New English."
    window.set_answer("New English.","新的中文。","第二题？",request_id="2")
    window.answer_history_select.setCurrentIndex(0)
    assert not window.follow_answers_checkbox.isChecked()
    assert window.answer_english.text()=="A complete first answer."


def test_completed_answer_reading_position_survives_background_generation(window):
    window.follow_answers_checkbox.setChecked(False)
    completed(window,en="A full research explanation. "*160,zh="完整的科研解释。"*160)
    QTest.qWait(80)
    enbar=window.answer_scroll.verticalScrollBar()
    zhbar=window.answer_chinese_scroll.verticalScrollBar()
    enbar.setValue(enbar.maximum()//2)
    zhbar.setValue(zhbar.maximum()//2)
    before=(enbar.value(),zhbar.value())
    window.set_answer(question_en="Next question?",streaming=True,request_id="2")
    for i in range(4):
        window.set_answer("New text. "*(i+1),streaming=True,request_id="2")
    QTest.qWait(40)
    assert (enbar.value(),zhbar.value())==before
    window.answer_new_button.click()
    window.answer_history_select.setCurrentIndex(0)
    QTest.qWait(40)
    assert (enbar.value(),zhbar.value())==before


def test_full_question_starts_with_actual_ask_and_background_raw_are_optional(window):
    setup="Alex, the last work is amazing. A comparison is possible. "*10
    ask="Should we ignore motion modeling and let the network hallucinate the video?"
    raw="rick uh amazing do you think we should ignore the motion modeling the video"
    window.set_answer("An answer.","回答。","应该忽略运动建模而让网络生成视频吗？",
                      question_en=setup+ask,question_preview_en=ask,question_raw_en=raw,request_id="1")
    window.answer_question_expand.click()
    dialog=window._answer_question_dialog
    focus=dialog.findChild(QPlainTextEdit,"answerQuestionEnglishFocus")
    assert focus.isVisible() and focus.toPlainText()==ask
    background=dialog.findChild(QPlainTextEdit,"answerQuestionEnglishFull")
    assert not background.isVisible() and background.toPlainText()==setup+ask
    original=dialog.findChild(QPlainTextEdit,"answerQuestionRawFull")
    assert not original.isVisible() and original.toPlainText()==raw
    dialog.close()


def test_same_question_repeated_with_new_request_cannot_retranslate_old_dialog(window):
    window.set_answer("First answer.","第一回答。","第一释义。",question_en="Same question?",request_id="1")
    window.answer_question_expand.click()
    dialog=window._answer_question_dialog
    window.follow_answers_checkbox.setChecked(True)
    window.set_answer("Second answer.","第二回答。","第二释义。",question_en="Same question?",request_id="2")
    assert dialog.findChild(QPlainTextEdit,"answerQuestionChineseFull").toPlainText()=="第一释义。"
    dialog.close()


def test_interview_mode_controls_are_painted_inline_and_choose_qa(window):
    window.set_settings({"interview_phase":"presentation"})
    events=[]
    window.interview_phase_changed.connect(events.append)
    assert window.focus_controls_button.menu() is None
    window.focus_controls_button.click()
    assert window.interview_controls_panel.isVisible()
    assert window.inline_qa_button.isVisible()
    window.inline_qa_button.click()
    assert events==["qa"]
    assert window.inline_qa_button.isChecked()
    assert not window.interview_controls_panel.isVisible()


def test_only_explicit_session_reset_removes_answer_history(window):
    completed(window)
    window.set_answer(question_en="Next question?",streaming=True,request_id="2")
    assert window.answer_history_select.count()==2
    window.clear_answer_history()
    assert not window._answer_history
    assert not window.answer_history_bar.isVisible()
    assert window.answer_english.text()==""
    assert window.answer_stack.currentIndex()==0


def test_pinned_answer_and_bilingual_question_fit_laptop_split_view(window):
    completed(window,en="A complete research explanation. "*160,zh="完整的科研解释。"*160)
    QTest.qWait(80)
    assert window.question_original.lines is None and window.question_translation.lines is None
    assert window.question_original.text()==window.question_original.full_text
    assert window.question_translation.text()==window.question_translation.full_text
    assert window.reading_docks["answer"].width()>window.reading_docks["transcript"].width()
    assert window.answer_scroll.height()>=window.answer_english.fontMetrics().lineSpacing()*3
    assert window.answer_chinese_scroll.height()>=window.answer_chinese.fontMetrics().lineSpacing()*3


def test_typical_actual_ask_keeps_important_final_words_visible(window):
    ask=("Do you think that is a good alternative that we completely ignore the motion modeling "
         "and just have a network hallucinate the video?")
    window.set_answer("A readable answer.","可读的回答。", "完全忽略运动建模，让网络生成视频是一个好的替代方案吗？",
                      question_en=ask,question_preview_en=ask,request_id="1")
    QTest.qWait(80)
    visible=" ".join(window.question_original.text().split())
    assert visible.endswith("hallucinate the video?")
    assert "…" not in visible


def test_overlay_question_remains_paired_with_selected_answer(window):
    window.follow_answers_checkbox.setChecked(False)
    completed(window)
    window.set_question("Second question?")
    window.show_subtitle_overlay()
    overlay=window._overlay
    overlay.mode.setCurrentIndex(overlay.mode.findData("question"))
    assert overlay._question_text=="First question?"
    assert overlay._answer_en=="A complete first answer."
    window.set_answer("Second answer.", "第二回答。", "第二题？", question_en="Second question?",request_id="2")
    window.set_question_packet({"question_en":"Third question?"})
    assert overlay._question_text=="First question?"
    assert overlay._answer_en=="A complete first answer."
    window.answer_new_button.click()
    assert overlay._question_text=="Second question?"
    assert overlay._answer_en=="Second answer."
    window.answer_history_select.setCurrentIndex(0)
    assert overlay._question_text=="First question?"
    assert overlay._answer_en=="A complete first answer."
    window.clear_answer_history()
    assert overlay._question_text=="Third question?"
    assert not overlay._answer_en


def test_overlay_receives_stable_speakers_and_updates_translation_in_same_card(window):
    window.append_transcript("one-a","How does this model work?","这个模型如何工作？",
                             speaker_id="s1",speaker_label="提问者",turn_id="t1")
    window.append_transcript("one-b","Does it use calibrated measurements?","",
                             speaker_id="s1",speaker_label="提问者",turn_id="t1")
    window.append_transcript("two","It uses calibrated measurements.","它使用校准测量。",
                             speaker_id="s2",speaker_label="演讲者",turn_id="t2")
    window.show_subtitle_overlay()
    overlay=window._overlay
    assert list(overlay.caption._cards)==["one-a","two"]
    first=overlay.caption._cards["one-a"]
    second=overlay.caption._cards["two"]
    assert first.pairs[0].english.text()=="How does this model work?"
    window.update_translation("one-b","它使用校准测量吗？")
    assert overlay.caption._cards["one-a"] is first
    assert overlay.caption._cards["two"] is second
    assert first.pairs[1].chinese.text()=="它使用校准测量吗？"
    window.update_speaker("t1","s1","老师")
    assert "老师" in first.header.text()
    stable=overlay.caption.toPlainText()
    window.append_transcript("draft","Could we also",partial=True,
                             speaker_id="s1",speaker_label="老师",turn_id="t3")
    assert overlay.caption.toPlainText().startswith(stable)
    assert overlay.caption._cards['one-a'] is first
    assert overlay.caption._cards['two'] is second
    assert '__prism_live_caption__' not in overlay.caption._cards
    assert overlay.caption.toPlainText()==stable


def test_ui_sends_only_eight_recent_distinct_cards_to_overlay(window):
    for i in range(12):
        window.append_transcript(f"p{i}",f"Question {i}?",f"问题{i}？",
                                 speaker_id=f"s{i%2}",speaker_label=f"说话人 {i%2+1}",turn_id=f"t{i}")
    window.show_subtitle_overlay()
    assert list(window._overlay.caption._cards)==[f"p{i}" for i in range(4,12)]


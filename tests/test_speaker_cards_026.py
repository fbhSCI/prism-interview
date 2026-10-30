"""Conversation cards are stable; question formatting mirrors the request packet."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtTest import QTest
from PySide6.QtCore import QPoint
from PySide6.QtWidgets import QApplication
from zoom_copilot.ui import CopilotWindow


def application():
    return QApplication.instance() or QApplication([])


def test_drafts_do_not_insert_reorder_or_resize_committed_cards():
    app = application()
    window = CopilotWindow()
    try:
        window.resize(1440, 920)
        window.show()
        window.append_transcript("a", "What do you measure?", "你测量什么？",
            speaker_id="speaker_1", speaker_label="说话人 1", turn_id="t1")
        QTest.qWait(30)
        card = window._entries["a"]
        before = card.geometry()
        for i in range(20):
            window.set_live_translation("A draft that keeps changing " + "word " * i,
                "不停修订的草稿" * i, "说话人 2", turn_id="t2")
            window.append_transcript("draft-only", "Changing words " * i, partial=True)
            app.processEvents()
        assert list(window._entries) == ["a"]
        assert card.geometry() == before
        assert card.english.text() == "What do you measure?"
        assert card.chinese.text() == "你测量什么？"
        assert window.conversation_title.text() == "说话人 1 · 双语对话"
        assert not window.draft_panel.isVisible()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_card_sentence_lines_preserve_numbers_and_original():
    app = application()
    window = CopilotWindow()
    try:
        raw = "uh we measured 0.95 but i am not sure then what's next"
        window.append_transcript("a", "We measured 0.95. I am not sure. What comes next?",
            "我们测到0.95。我不确定。下一步是什么？", raw_en=raw,
            speaker_id="speaker_1", speaker_label="说话人 1", turn_id="t1")
        card = window._entries["a"]
        assert card.english.text() == "We measured 0.95.\nI am not sure.\nWhat comes next?"
        assert card.chinese.text() == "我们测到0.95。\n我不确定。\n下一步是什么？"
        assert card.raw_text.text() == raw
        assert card.speaker.text() == "说话人 1"
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_old_translation_growth_follows_latest_without_rebuilding_cards():
    app = application()
    window = CopilotWindow()
    try:
        window.resize(1440, 920)
        window.show()
        for i in range(18):
            window.append_transcript(str(i), f"Question {i}: What is the measurement model? " * 3,
                "这是一条测试译文。", turn_id=f"t{i}")
        QTest.qWait(40)
        bar = window.transcript_scroll.verticalScrollBar()
        target = window._entries["10"]
        cards = dict(window._entries)
        bar.setValue(target.y() + 14)
        QTest.qWait(20)
        window.update_translation("1", "旧段译文增加了一些解释。" * 80)
        QTest.qWait(50)
        latest = window._latest_timeline_unit()
        top = latest.mapTo(window.transcript_scroll.viewport(), QPoint()).y()
        bottom = latest.mapTo(window.transcript_scroll.viewport(), QPoint(0, latest.height())).y()
        height = window.transcript_scroll.viewport().height()
        assert 0 <= top < height
        assert latest.height() > height or bottom <= height + 2
        assert all(window._entries[key] is card for key, card in cards.items())
        assert window._entries["1"].chinese.text().startswith("旧段译文")
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_question_packet_shows_source_speaker_and_separate_subquestions():
    app = application()
    window = CopilotWindow()
    try:
        question = "What did you measure? How did you validate it?"
        packet = {"question_en": question, "raw_question_en": "what uh did you measure how did you validate it",
            "turns": [{"label": "说话人 2", "speaker_id": "speaker_2", "role": "interviewer",
                "sentences": ["What did you measure?", "How did you validate it?"]}],
            "subquestions": ["What did you measure?", "How did you validate it?"]}
        window.set_question_packet(packet)
        assert window.current_question() == question
        assert window.question_reader_stack.currentIndex() == 1
        display = window.question_packet_view.toPlainText()
        assert "说话人 2 · 提问\nWhat did you measure?\nHow did you validate it?" in display
        assert display.count("What did you measure?") == 1
        assert window._question_packet["raw_question_en"] == packet["raw_question_en"]
        window.edit_question_button.click()
        assert window.question_reader_stack.currentIndex() == 0
        window.question_input.setPlainText("An edited complete question?")
        questions = []
        window.answer_requested.connect(questions.append)
        window.answer_button.click()
        assert questions == ["An edited complete question?"]
        window.edit_question_button.click()
        assert window.question_packet_view.toPlainText() == "当前问题（手动编辑） · 提问\nAn edited complete question?"
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_question_fragments_keep_one_speaker_heading_and_use_restored_whole_question():
    app = application()
    window = CopilotWindow()
    try:
        question = "How do you model scattering? How do you validate it?"
        window.set_question_packet({"question_en": question, "turns": [
            {"turn_id": "t1", "speaker_id": "speaker_1", "label": "说话人 1", "role": "interviewer", "clean_en": "How do you model"},
            {"turn_id": "t1", "speaker_id": "speaker_1", "label": "说话人 1", "role": "interviewer", "clean_en": "scattering how do you validate it"}]})
        assert window.question_packet_view.toPlainText() == (
            "说话人 1 · 提问\nHow do you model scattering?\nHow do you validate it?")
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_background_audio_packets_do_not_overwrite_a_manual_question_before_send():
    app = application()
    window = CopilotWindow()
    try:
        original = {"question_en": "How did you measure this?", "turns": [
            {"speaker_id": "speaker_1", "label": "说话人 1", "role": "interviewer",
             "sentences": ["How did you measure this?"]}]}
        window.set_question_packet(original)
        window.edit_question_button.click()
        manual = "How did you calibrate and validate the polarization measurements?"
        window.question_input.setPlainText(manual)
        for i, latest in enumerate(("How did you measure this? And validate it?",
                                   "What was the calibration error?",
                                   "Which reference targets did you use?")):
            incoming = {"question_en": latest, "preparation": {"state": f"update-{i}"},
                "turns": [{"speaker_id": "speaker_2", "label": "说话人 2", "role": "interviewer",
                    "sentences": [latest]}]}
            window.set_question_packet(incoming)
            app.processEvents()
            assert window.current_question() == manual
            assert window.question_reader_stack.currentIndex() == 0
        assert window._background_question_packet["question_en"] == "Which reference targets did you use?"
        window.edit_question_button.click()
        assert window.question_reader_stack.currentIndex() == 1
        assert window.question_packet_view.toPlainText() == "当前问题（手动编辑） · 提问\n" + manual
        # A speaker relabel or library-ready refresh can also arrive after Done.
        window.set_question_packet({"question_en": "A later automatic question?"})
        assert window.current_question() == manual
        assert manual in window.question_packet_view.toPlainText()
        sent = []
        window.answer_requested.connect(sent.append)
        window.answer_button.click()
        assert sent == [manual]
        # Explicit Next question intentionally clears the manual selection.
        window.set_question("")
        window.set_question_packet({"question_en": "", "turns": []})
        assert window.current_question() == ""
        assert not window._manual_question_active
        window.set_question_packet({"question_en": "A fresh automatic question?"})
        assert window.current_question() == "A fresh automatic question?"
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_opening_editor_without_typing_still_accepts_new_audio_questions():
    app = application()
    window = CopilotWindow()
    try:
        window.set_question_packet({"question_en": "First question?"})
        window.edit_question_button.click()
        window.set_question_packet({"question_en": "First question? An additional subquestion?"})
        assert window.current_question() == "First question? An additional subquestion?"
        assert not window._manual_question_active
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_long_automatic_question_keeps_primary_speaker_cards_large():
    app = application()
    window = CopilotWindow()
    try:
        window.resize(1440, 940)
        window.show()
        assert window.question_panel.isVisible()
        QTest.qWait(30)
        original_sizes = {name: dock.size() for name, dock in window.reading_docks.items()}
        window.append_transcript("a", "How do you validate the model?", "如何验证模型？", turn_id="t1")
        window.set_question_packet({"question_en": "How do you validate the model? " * 70,
            "turns": [{"label": "说话人 1", "speaker_id": "speaker_1", "role": "interviewer",
                "sentences": ["How do you validate the model?"] * 70}]})
        QTest.qWait(50)
        assert {name: dock.size() for name, dock in window.reading_docks.items()} == original_sizes
        assert window.transcript_scroll.viewport().height() >= 100
        assert "validate" in window.current_question()
        window.expand_question_button.click()
        QTest.qWait(30)
        assert window.reading_docks["question"].isFloating()
        assert window.question_panel.height() > 300
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_chinese_translation_splits_ascii_endings_without_splitting_decimals_or_terms():
    app = application()
    window = CopilotWindow()
    try:
        chinese = "置信度为0.95. 这是S0与S1的比较? 然后呢!这还不是定论.下一步请比较e.g. reference A."
        raw = "the confidence is 0.95 and this is S0 and S1"
        window.append_transcript("a", "Confidence is 0.95. Compare S0 and S1.", chinese, raw_en=raw)
        card = window._entries["a"]
        assert card.chinese.text() == ("置信度为0.95.\n这是S0与S1的比较?\n然后呢!\n"
            "这还不是定论.\n下一步请比较e.g. reference A.")
        assert card.raw_text.text() == raw
        assert "0.95" in card.english.text()
        assert "S0" in card.english.text()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()

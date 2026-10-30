"""Always-follow display and semantic question sections use the selected input."""
import os
import time
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QAbstractSlider

from zoom_copilot.ui import CopilotWindow
from zoom_copilot.question_packet import QuestionContext


def application():
    return QApplication.instance() or QApplication([])


def test_wheel_and_scrollbar_never_disable_follow_on_new_confirmed_content():
    app = application()
    window = CopilotWindow()
    try:
        window.resize(1440, 940)
        window.show()
        for number in range(16):
            window.append_transcript(str(number), "A complete research sentence. " * 8,
                "这是一句科研讨论。" * 8, turn_id=str(number))
        QTest.qWait(40)
        bar = window.transcript_scroll.verticalScrollBar()
        viewport = window.transcript_scroll.viewport()
        def assert_latest_visible():
            deadline = time.monotonic()+1.5
            while time.monotonic() < deadline:
                QTest.qWait(10)
                if (not window._timeline_follow_pending and not window._timeline_position_timer.isActive()
                        and not window._timeline_finish_timer.isActive()):
                    break
            latest = window._latest_timeline_unit()
            english_y = latest.english.mapTo(viewport, QPoint()).y()
            if latest.height() <= viewport.height():
                assert 0 <= english_y < viewport.height()
                assert latest.mapTo(viewport, QPoint(0, latest.height())).y() <= viewport.height()+2
            else:
                assert 0 < latest.mapTo(viewport, QPoint(0, latest.height())).y() <= viewport.height()+2
            assert bar.value()==bar.maximum()
        event = QWheelEvent(QPointF(100, 100), QPointF(viewport.mapToGlobal(QPoint(100, 100))),
            QPoint(), QPoint(0, 120), Qt.MouseButton.NoButton, Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase, False)
        QApplication.sendEvent(viewport, event)
        bar.triggerAction(QAbstractSlider.SliderAction.SliderToMinimum)
        assert bar.value() == 0
        cards = dict(window._entries)
        window.append_transcript("next", "What is your next experiment?", "下一个实验是什么？",
            speaker_id="speaker_2", speaker_label="说话人 2", turn_id="new-turn")
        QTest.qWait(40)
        assert_latest_visible()
        assert all(window._entries[key] is card for key, card in cards.items())
        bar.triggerAction(QAbstractSlider.SliderAction.SliderToMinimum)
        window.update_translation("next", "你接下来打算做什么实验？" * 10)
        QTest.qWait(40)
        assert_latest_visible()
        # A narrower reader wraps existing cards over more lines; the newest
        # confirmed translation remains visible through that layout change.
        window.resize(1120, 720)
        QTest.qWait(40)
        assert_latest_visible()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_question_view_separates_game_rules_from_the_actual_coordination_question():
    app = application()
    window = CopilotWindow()
    try:
        background = "Let us play a small game. We will take turns naming one model each."
        core = "Do you want to start, or do you want me to start?"
        raw = "So let. US play small game. One model. Each. Do. You want me to start?"
        full = background + " " + core
        window.set_question_packet({"question_en": full, "raw_question_en": raw,
            "background_en": background, "answer_question_en": core,
            "interaction_kind": "coordination", "turns": [
                {"label": "说话人 2", "role": "interviewer", "speaker_id": "speaker_2"}]})
        display = window.question_packet_view.toPlainText()
        assert display == ("说话人 2 · 对话协调\n\n需要回答\n" + core +
            "\n\n背景\nLet us play a small game.\nWe will take turns naming one model each.")
        assert "1." not in display and "2." not in display
        assert window.current_question() == full
        assert window._question_packet["raw_question_en"] == raw
        # Human editing replaces semantic selection as well as visible text.
        window.edit_question_button.click()
        window.question_input.setPlainText("How do you validate your model?")
        window.edit_question_button.click()
        assert "small game" not in window.question_packet_view.toPlainText()
        assert "start" not in window.question_packet_view.toPlainText()
        assert "validate" in window.question_packet_view.toPlainText()
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_short_confirmation_window_roundtrips_without_integer_truncation():
    app = application()
    window = CopilotWindow()
    try:
        window.set_settings({})
        assert window.get_settings()["caption_window_seconds"] == 3.6
        for seconds in (3.6, 3.2, 6.0, 12.0):
            window.set_settings({"caption_window_seconds": seconds})
            assert window.get_settings()["caption_window_seconds"] == seconds
        assert not window.get_settings()["live_translation"]
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_multiple_interviewers_keep_their_own_scientific_question_sources():
    app = application()
    window = CopilotWindow()
    try:
        context = QuestionContext()
        first = "We have four linear analyzer channels. How would you calibrate M?"
        second = "How would you validate the reconstructed Stokes parameters?"
        # One question crosses an ASR window; source text must remain coherent
        # without combining the two interviewers' words under one speaker.
        context.record("a1", "We have four linear analyzer channels. How would you", accepted=True,
            speaker_id="a", speaker_label="面试官 A", turn_id="a1")
        context.record("a2", "calibrate M?", accepted=True,
            speaker_id="a", speaker_label="面试官 A", turn_id="a2")
        context.record("b1", second, accepted=True,
            speaker_id="b", speaker_label="面试官 B", turn_id="b")
        packet = context.build(first + " " + second, raw_question=first + " " + second)
        window.set_question_packet(packet.to_ui_data())
        display = window.question_packet_view.toPlainText()
        assert display.startswith("多位说话人 · 提问\n\n需要回答（合并整理，逐人来源见下方）")
        assert display.index("需要回答") < display.index("逐人来源")
        sources = display.split("逐人来源\n", 1)[1]
        assert sources == ("面试官 A · 提问\nWe have four linear analyzer channels.\n"
            "How would you calibrate M?\n\n面试官 B · 提问\n" + second)
        assert "1." not in sources and "2." not in sources
        assert window.current_question() == packet.question_en
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()


def test_setup_confirmation_waits_without_presenting_background_as_an_answerable_question():
    app = application()
    window = CopilotWindow()
    try:
        setup = "We will take turns naming a model. The person who cannot name one loses, right?"
        window.set_question_packet({"question_en": setup, "raw_question_en": setup,
            "background_en": setup, "answer_question_en": "", "interaction_kind": "setup",
            "turns": [{"speaker_id": "speaker_2", "label": "说话人 2", "role": "unknown",
                "clean_en": setup}]})
        display = window.question_packet_view.toPlainText()
        assert display.startswith("背景 · 等待实际提问\n\n正在等待对方提出完整问题…")
        assert "需要回答" not in display
        assert "背景\nWe will take turns naming a model.\nThe person who cannot name one loses, right?" in display
        assert window.current_question() == setup
        assert window._question_packet["raw_question_en"] == setup
    finally:
        window.close()
        window.deleteLater()
        app.processEvents()

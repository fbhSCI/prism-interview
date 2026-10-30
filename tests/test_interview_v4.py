import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from zoom_copilot.turns import QuestionAssembler
from zoom_copilot.ui import CopilotWindow
from test_controller_v2 import controller


def test_pause_after_auto_answer_retains_prefix_and_revises_request(controller):
    c = controller
    c.listening = True
    c.window.qa_phase_button.click()
    c.on_transcript("final", "How would you estimate the material reflectance?")
    c.assembler.last_voice -= 4
    c.assembler.last_text -= 4
    c.auto_answer()
    assert c.busy
    prefix = c.assembler.text
    assert prefix.startswith("How would")
    c.assembler.voice()
    c.on_transcript("final", "Especially when the illumination is unknown and there is scattering.")
    assert c.window.current_question().startswith(prefix)
    c.assembler.last_voice -= 4
    c.assembler.last_text -= 4
    c.auto_answer()
    assert c._cancel.is_set()
    assert c._requests[-1][1] == prefix + " Especially when the illumination is unknown and there is scattering."
    c.auto_answer()
    assert len(c._requests) == 1


def test_preface_survives_pause_before_actual_question(controller):
    c = controller
    c.listening = True
    c.window.qa_phase_button.click()
    c.on_transcript("final", "The incident polarization is unknown in this scene.")
    c.assembler.last_voice -= 4
    c.assembler.last_text -= 4
    c.auto_answer()
    assert c.assembler.text and not c.busy
    c.on_transcript("final", "How would you calibrate the instrument?")
    assert c.window.current_question().startswith("The incident polarization")


def test_new_question_long_gap_and_explicit_boundary():
    a = QuestionAssembler()
    a.voice(1)
    a.add("Why is the inverse problem ill posed?", at=2)
    a.consume(retain=True)
    a.voice(6)
    a.add("In particular under scattering.", at=7)
    assert a.text.startswith("Why is")
    a.consume(retain=True)
    a.voice(22)
    a.add("What is the calibration matrix?", at=23)
    assert a.text == "What is the calibration matrix?"
    a.reset()
    assert not a.text


def test_question_ending_in_conjunction_waits_for_completion():
    a = QuestionAssembler()
    a.add("How would you separate illumination and", at=1)
    assert not a.ready(now=5)
    a.add("reflectance?", at=5)
    assert a.ready(now=8)


def test_overlay_restore_resize_growth_and_stable_history():
    app = QApplication.instance() or QApplication([])
    w = CopilotWindow()
    w.resize(1280, 900)
    w.show()
    # Incoming words preserve the user's dock allocation. Each pane can float.
    QTest.qWait(30)
    before = w.reading_docks["question"].height()
    w.set_question("Please explain the measurement model and the proposed validation. " * 14)
    QTest.qWait(40)
    assert w.reading_docks["question"].height() == before
    w.expand_question_button.click()
    QTest.qWait(30)
    assert w.reading_docks["question"].isFloating()
    assert w.question_input.height() > 65
    w.expand_question_button.click()
    # Confirmed speaker cards stay primary; revisable draft never moves them.
    assert w.transcript_stack.isVisible()
    assert not w.draft_panel.isVisible()
    for i in range(20):
        w.append_transcript(str(i), "Confirmed English words. " * 10)
    QTest.qWait(100)
    bar = w.transcript_scroll.verticalScrollBar()
    assert bar.maximum() > 100
    bar.setValue(100)
    for i in range(12):
        w.set_live_caption("New partial words. " * (i + 1))
    QTest.qWait(30)
    assert bar.value() == 100
    w.set_answer("Suggested English.", "对应中文。")
    w.show_subtitle_overlay()
    QTest.qWait(30)
    assert w._overlay.isVisible() and not w.isVisible()
    assert "Suggested English.\n\n对应中文。" == w._overlay.answer.toPlainText()
    w._overlay.move(30, 35)
    w._overlay.resize(700, 260)
    QTest.qWait(20)
    assert w._overlay.width() == 700
    w._overlay.close()
    QTest.qWait(20)
    assert w.isVisible() and not w._overlay.isVisible()
    w.close()

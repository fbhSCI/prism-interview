"""Chronological conversation reading: append below, optionally follow its end."""
import os
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PySide6.QtCore import QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from zoom_copilot.ui import CopilotWindow


def settled(window):
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        QTest.qWait(10)
        if (not window._timeline_follow_pending
                and not window._timeline_position_timer.isActive()
                and not window._timeline_finish_timer.isActive()
                and window.transcript_scroll.viewport().updatesEnabled()):
            return
    pytest.fail('Conversation layout did not settle')


@pytest.fixture
def window(tmp_path, monkeypatch):
    monkeypatch.setenv('PRISM_INTERVIEW_HOME', str(tmp_path))
    app = QApplication.instance() or QApplication([])
    value = CopilotWindow()
    value.resize(1366, 850)
    value.show()
    QTest.qWait(60)
    yield value
    value.close()
    value.deleteLater()
    app.processEvents()


def append(window, key):
    window.append_transcript(key, f'Question {key}: How do you calibrate the optical system?',
                             f'问题 {key}：你如何标定光学系统？', turn_id=key,
                             speaker_label=f'说话人 {key}')


def test_new_messages_appear_below_history_at_bottom_not_at_reading_top(window):
    window.reading_mode.setCurrentIndex(window.reading_mode.findData('transcript'))
    QTest.qWait(40)
    for i in range(8):
        append(window, str(i))
    settled(window)
    previous = window._entries['7']
    append(window, '8')
    settled(window)
    newest = window._entries['8']
    view = window.transcript_scroll.viewport()
    bar = window.transcript_scroll.verticalScrollBar()
    assert newest.y() >= previous.y()+previous.height()
    assert bar.value() == bar.maximum() > 0
    # Both the preceding message and the newest pair are readable together;
    # 039's reserved screenful + 12px top anchor fails these assertions.
    assert previous.mapTo(view, QPoint(0, previous.height())).y() > 0
    assert newest.mapTo(view, QPoint()).y() > view.height()//3
    assert newest.mapTo(view, QPoint(0, newest.height())).y() <= view.height()+2
    assert newest._unit_widgets['8'].chinese.isVisible()


def test_follow_off_keeps_history_then_resume_returns_to_bottom(window):
    for i in range(12):
        append(window, str(i))
    settled(window)
    window.follow_transcript.setChecked(False)
    bar = window.transcript_scroll.verticalScrollBar()
    target = window._entries['3']
    bar.setValue(target.y()+6)
    before = target.mapTo(window.transcript_scroll.viewport(), QPoint()).y()
    append(window, '12')
    window.update_translation('0', '一段更长的早期译文。'*40)
    settled(window)
    assert target.mapTo(window.transcript_scroll.viewport(), QPoint()).y() == before
    assert bar.value() < bar.maximum()
    assert window.latest_button.text() == '↓ 回到最新'
    window.latest_button.click()
    settled(window)
    assert window.follow_transcript.isChecked()
    assert bar.value() == bar.maximum()
    assert window.latest_button.text() == '已在最新位置'


def test_resize_and_late_translation_keep_the_newest_end_visible(window):
    for i in range(9):
        append(window, str(i))
    settled(window)
    window.update_translation('8', '这段译文随确认的英语更新。'*30)
    settled(window)
    window.resize(1200, 730)
    settled(window)
    bar = window.transcript_scroll.verticalScrollBar()
    newest = window._latest_timeline_unit()
    view = window.transcript_scroll.viewport()
    assert bar.value() == bar.maximum()
    assert 0 < newest.mapTo(view, QPoint(0, newest.height())).y() <= view.height()+2
    assert newest.chinese.text() == '这段译文随确认的英语更新。'*30

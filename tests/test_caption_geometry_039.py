"""Actual Qt viewport geometry, not just caption string equality."""
import os
import time
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QPoint, QCoreApplication, QEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
import pytest
from zoom_copilot.ui import CopilotWindow
from zoom_copilot.overlay import SubtitleOverlay


@pytest.fixture
def window(tmp_path, monkeypatch):
    monkeypatch.setenv('PRISM_INTERVIEW_HOME', str(tmp_path))
    app=QApplication.instance() or QApplication([])
    value=CopilotWindow();value.resize(1366,850);value.show();QTest.qWait(100)
    yield value
    value.close();value.deleteLater();app.processEvents()


def point(widget, viewport):
    return widget.mapTo(viewport,QPoint()).y()


def history(window):
    for i in range(7):
        window.append_transcript(str(i),'A previous confirmed calibration question. '*3,
                                 '此前已经确认的问题。'*4,turn_id=str(i))


def test_words_and_late_same_height_chinese_do_not_move_readable_english(window):
    history(window)
    window.append_transcript('latest','Could you explain the measurement?', '你能解释测量吗？',turn_id='latest')
    QTest.qWait(100)
    unit=window._entries['latest']._unit_widgets['latest']
    viewport=window.transcript_scroll.viewport()
    bar=window.transcript_scroll.verticalScrollBar()
    positions=[];moves=[]
    bar.valueChanged.connect(moves.append)
    initial=point(unit.english,viewport)
    old_height=unit.height()
    window.update_message('latest','Could you explain the measurement?', '你能解释测量吗？',translation_pending=True)
    for _ in range(5):QTest.qWait(10);positions.append(point(unit.english,viewport))
    window.update_translation('latest','你能说明测量吗？')
    for _ in range(5):QTest.qWait(10);positions.append(point(unit.english,viewport))
    for i in range(12):
        window.update_message('draft', 'An unconfirmed word '+str(i), partial=True)
        QTest.qWait(5);positions.append(point(unit.english,viewport))
    assert 0 <= initial < viewport.height()
    assert positions == [initial]*len(positions)
    assert unit.height()==old_height
    assert moves==[]
    assert not window.draft_toggle.isChecked()


def test_confirmed_growth_that_still_fits_follows_bottom_without_reverse_bounce(window):
    history(window)
    window.reading_mode.setCurrentIndex(window.reading_mode.findData('transcript'))
    QTest.qWait(80)
    window.append_transcript('latest','How do you calibrate?', '你如何校准？',turn_id='latest')
    QTest.qWait(100)
    unit=window._entries['latest']._unit_widgets['latest'];view=window.transcript_scroll.viewport()
    initial=point(unit.english,view)
    positions=[];scrolls=[]
    window.transcript_scroll.verticalScrollBar().valueChanged.connect(scrolls.append)
    for text in ('How do you calibrate the system?',
                 'How do you calibrate the system? How do you estimate uncertainty?',
                 'How do you calibrate the system? How do you estimate uncertainty under changing illumination?'):
        window.update_message('latest',text,'你如何校准？',translation_pending=True)
        for _ in range(5):QTest.qWait(10);positions.append(point(unit.english,view))
    assert unit.height()<view.height()
    assert positions==sorted(positions,reverse=True)
    assert positions[-1]<=initial
    assert scrolls==sorted(scrolls)
    assert point(unit,view)+unit.height()<=view.height()
    bar=window.transcript_scroll.verticalScrollBar()
    assert bar.value()==bar.maximum()


def test_growing_overscreen_unit_follows_newest_tail_and_retains_complete_text(window):
    history(window)
    en='Could you explain the physical measurement model?'
    zh='你能解释一下物理测量模型吗？'
    window.append_transcript('latest',en,zh,turn_id='latest');QTest.qWait(100)
    initial=point(window._latest_timeline_unit().english,window.transcript_scroll.viewport())
    bar=window.transcript_scroll.verticalScrollBar();moves=[]
    bar.valueChanged.connect(moves.append)
    for i,suffix in enumerate((' We measure reflectance under controlled illumination.',
       ' We then estimate uncertainty from repeated observations.',
       ' The model retains the measured spectral and polarization channels.')):
        en+=suffix
        window.update_message('latest',en,zh,translation_pending=True)
        QTest.qWait(80)
        zh='你能解释一下物理测量模型吗？'+'我们测量反射特性，并评估照明变化带来的不确定性。'*(i+1)
        window.update_translation('latest',zh);QTest.qWait(80)
    # Once per settled confirmed English/Chinese update, never a reverse jump.
    assert len(moves)<=6
    assert moves==sorted(moves)
    unit=window._latest_timeline_unit();view=window.transcript_scroll.viewport()
    assert point(unit.english,view)<=initial
    assert 0<point(unit,view)+unit.height()<=view.height()+2
    assert bar.value()==bar.maximum()
    assert unit.height()>view.height()
    assert window._entries['latest']._segments['latest'][0]==en
    assert window._entries['latest']._segments['latest'][1]==zh


def test_disabled_follow_preserves_visible_card_pixel_during_old_translation_layout(window):
    history(window)
    window.append_transcript('last','A latest source unit.', '最后一段。',turn_id='last')
    QTest.qWait(100)
    window.follow_transcript.setChecked(False)
    target=window._entries['3'];bar=window.transcript_scroll.verticalScrollBar()
    bar.setValue(target.y()+8)
    initial=point(target,window.transcript_scroll.viewport())
    window.update_translation('0','这条旧译文增加高度，但不应该移动正在阅读的卡片。'*35)
    samples=[]
    for _ in range(12):QTest.qWait(10);samples.append(point(target,window.transcript_scroll.viewport()))
    assert max(abs(value-initial) for value in samples)<=2
    assert not window.follow_transcript.isChecked()


def test_workspace_scrollbar_width_is_reserved_before_history_overflows(window):
    window.append_transcript('first','Short.', '简短。')
    QTest.qWait(80)
    width=window.transcript_scroll.viewport().width()
    history(window);QTest.qWait(100)
    assert window.transcript_scroll.verticalScrollBar().maximum()>0
    assert window.transcript_scroll.viewport().width()==width


def test_overlay_late_chinese_keeps_pixel_anchor_and_scrollbar_width():
    app=QApplication.instance() or QApplication([]);overlay=SubtitleOverlay()
    try:
        overlay.auto_height_action.setChecked(False);overlay.resize(1000,600);overlay.show()
        row={'entry_id':'same','en':'Could you explain the physical measurement model?', 'zh':'请解释物理测量模型。'}
        overlay.set_conversation([row]);QTest.qWait(100)
        pair=overlay.caption._cards['same'].pairs[0];view=overlay.caption.viewport()
        width=view.width();initial=point(pair.english,view);positions=[]
        for i in range(3):
            row['zh']+='我们还需要保留测量中的不确定性。'
            overlay.set_conversation([row])
            for _ in range(6):QTest.qWait(10);positions.append(point(pair.english,view))
        assert pair.height()<view.height()
        assert positions==[initial]*len(positions)
        assert view.width()==width
        assert pair.chinese.text()==row['zh']
        assert not overlay.draft_action.isChecked()
    finally:
        overlay.hide();overlay.deleteLater();app.processEvents()


def test_batch_initialization_does_not_latch_a_zero_range_offscreen_anchor(window):
    for number in range(20):
        window.append_transcript(str(number), 'A complete calibration sentence. '*12,
                                 '这是一段测量条件的译文。'*8,turn_id=str(number))
    # Twenty wrapped cards can need several Qt event turns under suite/build
    # load. Observe the final range and owned layout timers, not a fixed delay.
    deadline=time.monotonic()+1.5
    while time.monotonic()<deadline:
        QTest.qWait(10)
        if (window.transcript_scroll.verticalScrollBar().maximum()>1000
                and not window._timeline_follow_pending
                and not window._timeline_position_timer.isActive()
                and not window._timeline_finish_timer.isActive()):
            break
    latest=window._latest_timeline_unit();view=window.transcript_scroll.viewport()
    bar=window.transcript_scroll.verticalScrollBar()
    assert bar.maximum()>1000
    assert bar.value()>0
    assert bar.value()==bar.maximum()
    assert 0<latest.mapTo(view,QPoint(0,latest.height())).y()<=view.height()+2
    assert view.updatesEnabled()
    # A deliberate scroll with follow still on must not latch a new offscreen
    # anchor when a truly new confirmed source arrives.
    bar.setValue(0)
    window.append_transcript('newest','What is your next experiment?', '下一项实验是什么？',turn_id='newest')
    QTest.qWait(80)
    newest=window._latest_timeline_unit()
    assert 0<=newest.english.mapTo(view,QPoint()).y()<view.height()
    assert newest.mapTo(view,QPoint(0,newest.height())).y()<=view.height()+2


@pytest.mark.parametrize('floating',[False,True])
def test_destroying_reader_with_pending_final_layout_has_no_detached_callback(floating):
    app=QApplication.instance() or QApplication([])
    value=SubtitleOverlay() if floating else CopilotWindow()
    value.show();QTest.qWait(30)
    destroyed=[];value.destroyed.connect(lambda:destroyed.append(True))
    if floating:
        value.set_conversation([{'entry_id':'pending','en':'Confirmed sentence.','zh':'确认译文。'}])
        value.caption._apply_position()
        assert value.caption._position_finish_timer.isActive()
        assert value.caption._position_finish_timer.parent() is value.caption
    else:
        value.append_transcript('pending','Confirmed sentence.','确认译文。')
        value._apply_timeline_position()
        assert value._timeline_finish_timer.isActive()
        assert value._timeline_finish_timer.parent() is value
    value.hide();value.deleteLater()
    QCoreApplication.sendPostedEvents(None,QEvent.Type.DeferredDelete)
    assert destroyed
    # Processing future events used to call a receiverless lambda against the
    # deleted QVBoxLayout and fail in the next unrelated UI test.
    QTest.qWait(80)
    app.processEvents()

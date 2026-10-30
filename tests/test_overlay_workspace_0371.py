"""A hidden main workspace publishes confirmed English and its late translation."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtCore import QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from zoom_copilot.ui import CopilotWindow

def test_main_window_hidden_overlay_gets_confirmed_words_and_late_translation():
    app=QApplication.instance() or QApplication([])
    window=CopilotWindow()
    try:
        window.show()
        window.update_message('first','Previous complete question.','上一道完整问题。',
                              speaker_label='面试官',turn_id='turn1')
        window.show_subtitle_overlay()
        QTest.qWait(70)
        assert not window.isVisible()
        overlay=window._overlay
        first=overlay.caption._cards['first']
        window.update_message('partial','Could we estimate',partial=True,
                              speaker_label='面试官',turn_id='turn2')
        QTest.qWait(70)
        assert '__prism_live_caption__' not in overlay.caption._cards
        assert not overlay.draft_action.isChecked()
        assert overlay.caption._cards['first'] is first
        window.update_message('second','Could we estimate reflectance?',speaker_label='面试官',turn_id='turn2')
        window.update_translation('second','能否估计反射特性？')
        QTest.qWait(70)
        assert overlay.caption._cards['first'] is first
        assert '__prism_live_caption__' not in overlay.caption._cards
        pair=overlay.caption._cards['second'].pairs[0]
        assert pair.english.text()=='Could we estimate reflectance?'
        assert pair.chinese.text()=='能否估计反射特性？'
        assert pair.english.mapTo(overlay.caption.viewport(),QPoint()).y()>=0
    finally:
        window.close();window.deleteLater();app.processEvents()

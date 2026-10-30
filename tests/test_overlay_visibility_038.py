"""Actual viewport visibility and per-pixel desktop transparency regressions."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from PySide6.QtCore import QPoint
from PySide6.QtGui import QImage, QColor
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from zoom_copilot.overlay import SubtitleOverlay

def app():
    return QApplication.instance() or QApplication([])

def release(window):
    window.hide(); window.deleteLater(); app().processEvents()

def test_default_follow_keeps_english_on_screen_when_pair_fits():
    application=app(); window=SubtitleOverlay()
    try:
        window.show()
        window.set_conversation([{'entry_id':'short-en-long-zh','en':'Could you explain this method?',
            'zh':'The corresponding translation is significantly longer. '*8}])
        QTest.qWait(150)
        pair=window.caption._cards['short-en-long-zh'].pairs[0]
        english_top=pair.english.mapTo(window.caption.viewport(),QPoint(0,0)).y()
        assert pair.height()<window.caption.viewport().height()
        assert english_top>=0
        assert english_top+pair.english.height()<=window.caption.viewport().height()
        window.set_conversation([{'entry_id':'short-en-long-zh','en':'Could you explain this method?',
            'zh':'The corresponding translation is significantly longer. '*10}])
        QTest.qWait(150)
        assert pair.english.mapTo(window.caption.viewport(),QPoint(0,0)).y()>=0
    finally:release(window)

def test_provisional_english_revises_in_place_with_history_and_translation_pairs():
    application=app(); window=SubtitleOverlay()
    try:
        window.draft_action.setChecked(True)
        window.show(); window.set_conversation([{'entry_id':'old','en':'Previous question.','zh':'上一题。'}])
        old=window.caption._cards['old']
        window.set_caption('Can we estimate',speaker_label='面试官',state='正在听译')
        QTest.qWait(60)
        live=window.caption._cards['__prism_live_caption__']
        pair=live.pairs[0]
        assert pair.english.isVisible()
        assert pair.english.text()=='Can we estimate'
        window.set_caption('Can we estimate reflectance?', '能否估计反射特性？',speaker_label='面试官',state='正在听译')
        QTest.qWait(60)
        assert window.caption._cards['old'] is old
        assert window.caption._cards['__prism_live_caption__'] is live
        assert live.pairs[0] is pair
        assert pair.english.text()=='Can we estimate reflectance?'
        assert pair.chinese.text()=='能否估计反射特性？'
        assert pair.english.mapTo(window.caption.viewport(),QPoint()).y()>=0
        window.set_caption('')
        application.processEvents()
        assert '__prism_live_caption__' not in window.caption._cards
        assert window.caption._cards['old'] is old
    finally:release(window)

def card_alpha(window):
    # Sample a text-free inner margin; whole-window opacity must remain 1.0.
    image=window.grab().toImage().convertToFormat(QImage.Format.Format_ARGB32)
    card=window.caption._cards['alpha']
    point=card.mapTo(window,QPoint(8,card.height()//2))
    ratio=image.devicePixelRatio()
    return image.pixelColor(round(point.x()*ratio),round(point.y()*ratio)).alpha()

def test_background_is_transparent_while_text_keeps_full_opacity():
    application=app(); window=SubtitleOverlay()
    try:
        window.show(); window.set_conversation([{'entry_id':'alpha','en':'Visible English','zh':'可见的中文'}])
        QTest.qWait(100)
        assert window.transparency_slider.value()==80
        assert window.windowOpacity()==1.0
        default_alpha=card_alpha(window)
        assert 0<default_alpha<90
        window.set_background_transparency(100);QTest.qWait(30)
        assert window.transparency_slider.value()==100
        assert card_alpha(window)==0
        window.set_background_transparency(0);QTest.qWait(30)
        assert card_alpha(window)==255
        pair=window.caption._cards['alpha'].pairs[0]
        assert pair.english.palette().windowText().color().alpha()==255
        assert pair.chinese.palette().windowText().color().alpha()==255
        assert pair.english.graphicsEffect() is not None
        window.set_transparent(False);assert not window.transparency_slider.isEnabled()
        window.set_transparent(True);assert window.transparency_slider.isEnabled()
    finally:release(window)

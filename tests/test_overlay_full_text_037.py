"""Interview captions keep complete words visible without an expand action."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtCore import QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QFont,QFontDatabase
from pathlib import Path
from zoom_copilot.overlay import SubtitleOverlay

def application():
    app=QApplication.instance() or QApplication([])
    for font in ('segoeui.ttf','msyh.ttc','msjh.ttc'):
        QFontDatabase.addApplicationFont(str(Path('C:/Windows/Fonts')/font))
    app.setFont(QFont('Microsoft YaHei',10))
    return app

def test_long_bilingual_pair_is_full_text_in_the_caption_widget():
    app=application()
    window=SubtitleOverlay()
    try:
        window.resize(920,320);window.show()
        en=('Can you distinguish material reflectance from varying illumination? '*9)+'End of the complete question.'
        zh=('如何区分材质反射特性和变化的照明？'*15)+'完整问题的末尾。'
        window.set_conversation([{'entry_id':'long','speaker_label':'面试官','en':en,'zh':zh,'units':[(en,zh)]}])
        QTest.qWait(80)
        pair=window.caption._cards['long'].pairs[0]
        assert pair.english.text()==en.strip() and pair.chinese.text()==zh
        assert not hasattr(pair,'expand')
        assert window.height()>320
        assert window.height()<=window.screen().availableGeometry().height()-24
        assert pair.english.height()>=pair.english.heightForWidth(pair.english.width())
        assert pair.chinese.height()>=pair.chinese.heightForWidth(pair.chinese.width())
        size=window.size()
        window.set_conversation([{'entry_id':'short','en':'Why?','zh':'为什么？'}]);QTest.qWait(30)
        assert window.size()==size
    finally:
        window.hide();window.deleteLater();app.processEvents()

def test_over_screen_unit_follows_latest_tail_and_preserves_scrollable_english():
    app=application()
    window=SubtitleOverlay()
    try:
        window.auto_height_action.setChecked(False)
        window.resize(800,360);window.show()
        en='Full English question with physical constraints. '*100
        zh='这里完整保留对应的中文译文。'*150
        window.set_conversation([{'entry_id':'huge','en':en,'zh':zh}]);QTest.qWait(100)
        pair=window.caption._cards['huge'].pairs[0]
        assert pair.height()>window.caption.viewport().height()
        view=window.caption.viewport();bar=window.caption.verticalScrollBar()
        # Latest-tail following includes Chinese that arrives after English.
        # A pair larger than the screen remains complete and scrollable; it
        # cannot show its beginning and end simultaneously in a finite reader.
        assert bar.value()==bar.maximum()
        assert 0<pair.chinese.mapTo(view,QPoint(0,pair.chinese.height())).y()<=view.height()
        window.follow_action.setChecked(False)
        bar.setValue(0)
        assert pair.english.mapTo(view,QPoint()).y()>=0
        assert pair.english.text()==en.strip() and pair.chinese.text()==zh
    finally:
        window.hide();window.deleteLater();app.processEvents()

def test_already_fitting_pair_does_not_expand_to_screen_height_during_first_layout():
    app=application();window=SubtitleOverlay()
    try:
        window.resize(1120,520);window.show()
        en=('Could inverse rendering estimate reflectance from images? '*7)
        zh='能否通过逆渲染从图像估计反射特性？'*8
        window.set_conversation([{'entry_id':'fits','en':en,'zh':zh}]);QTest.qWait(100)
        pair=window.caption._cards['fits'].pairs[0]
        assert pair.height()<window.caption.viewport().height()
        assert window.height()==520
    finally:
        window.hide();window.deleteLater();app.processEvents()


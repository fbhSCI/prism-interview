"""Floating-reader Qt paints and actual tail geometry; no native flicker claim."""
import os
import time
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest
from PySide6.QtCore import QEvent, QObject, QPoint
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from zoom_copilot.overlay import SubtitleOverlay


def settle(window):
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        QTest.qWait(10)
        if (not window.caption._position_timer.isActive()
                and not window.caption._position_finish_timer.isActive()
                and not window.caption._anchor_finish_timer.isActive()
                and not window._fit_timer.isActive()):
            QTest.qWait(30)
            return
    pytest.fail('Floating caption layout did not settle')


class PaintProbe(QObject):
    def __init__(self, parent):
        super().__init__(parent)
        self.events = []
        self.names = {}

    def watch(self, name, widget):
        self.names[id(widget)] = name
        widget.installEventFilter(self)

    def eventFilter(self, widget, event):
        if event.type() == QEvent.Type.Paint:
            self.events.append((self.names[id(widget)],
                event.region().boundingRect() == widget.rect()))
        return False


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    # The Windows offscreen plugin does not enumerate system fonts itself.
    for name in ('segoeui.ttf', 'msyh.ttc', 'msjh.ttc'):
        QFontDatabase.addApplicationFont(str(Path('C:/Windows/Fonts') / name))
    app.setFont(QFont('Microsoft YaHei UI', 10))
    value = SubtitleOverlay()
    value.auto_height_action.setChecked(False)
    value.resize(1000, 600)
    value.show()
    value.set_conversation([row('same', 'We measure light.', '我们测量光。')])
    settle(value)
    yield value
    value.hide(); value.deleteLater(); app.processEvents()


def row(identity, en, zh):
    return {'entry_id': identity, 'speaker_label': '面试官', 'state': '已成句',
            'en': en, 'zh': zh}


def probe(window):
    value = PaintProbe(window)
    card = window.caption._cards['same']
    pair = card.pairs[0]
    for name, widget in [('viewport', window.caption.viewport()), ('card', card),
            ('english', pair.english), ('chinese', pair.chinese), ('speaker', card.header)]:
        value.watch(name, widget)
    return value, card, pair


def bottom(widget, parent):
    return widget.mapTo(parent, QPoint(0, widget.height())).y()


def test_short_revisions_keep_updates_enabled_and_do_not_dirty_entire_reader(window):
    observer, card, pair = probe(window)
    height = pair.height()
    for en, zh in [('We measure reflected light.', '我们测量光。'),
                   ('We measure reflected light.', '我们测量反射光。')]:
        window.set_conversation([row('same', en, zh)])
        assert window.caption.viewport().updatesEnabled()
        settle(window)
    assert observer.events
    assert not any(name == 'viewport' and full for name, full in observer.events), observer.events
    assert card is window.caption._cards['same']
    assert pair is card.pairs[0]
    assert pair.height() == height


def test_identical_records_and_unconfirmed_words_do_not_repaint_or_restart_layout(window):
    window.auto_height_action.setChecked(True)
    settle(window)
    observer, card, pair = probe(window)
    revision = window.caption._revision
    for text in ('How', 'How do', 'How do we', 'How do we calibrate'):
        window.set_conversation([row('same', 'We measure light.', '我们测量光。')])
        window.set_caption(text, speaker_label='面试官', state='正在听译 · 词句会修订')
        assert not window._fit_timer.isActive()
        QTest.qWait(15)
    assert observer.events == []
    assert window.caption._revision == revision
    assert not window.draft_action.isChecked()
    assert window.caption._cards['same'] is card and card.pairs[0] is pair


def test_late_translation_over_screen_follows_chinese_tail_with_all_text_retained(window):
    en = 'Could you explain the spectral measurement model?'
    window.set_conversation([row('same', en, '请解释光谱测量模型。')])
    settle(window)
    card = window.caption._cards['same']; pair = card.pairs[0]
    moves = []; bar = window.caption.verticalScrollBar()
    bar.valueChanged.connect(moves.append)
    for count in (8, 25, 60):
        zh = '这里完整保留测量条件、材料反射特性和不确定性。' * count + '译文末尾。'
        window.set_conversation([row('same', en, zh)])
        settle(window)
        view = window.caption.viewport()
        assert 0 < bottom(pair.chinese, view) <= view.height()
        assert bar.value() == bar.maximum()
        assert pair.english.text() == en and pair.chinese.text() == zh
        assert pair.chinese.height() >= pair.chinese.heightForWidth(pair.chinese.width())
    assert pair.height() > view.height()
    assert moves == sorted(moves)
    assert len(moves) <= 6


def test_newest_card_is_below_history_and_fitting_pair_remains_complete(window):
    rows = [row(str(i), 'A previous measurement question. ' * 3, '此前的测量问题。' * 3)
            for i in range(7)]
    rows.append(row('last', 'What is your next experiment?', '下一项实验是什么？'))
    window.set_conversation(rows); settle(window)
    reader = window.caption; last = reader._cards['last']; pair = last.pairs[0]
    assert [reader._cards[key].y() for key in reader._order] == sorted(
        reader._cards[key].y() for key in reader._order)
    view = reader.viewport(); bar = reader.verticalScrollBar()
    assert 0 <= pair.english.mapTo(view, QPoint()).y()
    assert 0 < bottom(pair.chinese, view) <= view.height()
    assert bar.value() == bar.maximum()


def test_disabled_follow_keeps_history_anchor_while_older_translation_grows(window):
    rows = [row(str(i), 'A previous measurement question. ' * 3, '此前的测量问题。' * 3)
            for i in range(8)]
    window.set_conversation(rows); settle(window)
    window.follow_action.setChecked(False)
    target = window.caption._cards['4']; bar = window.caption.verticalScrollBar()
    bar.setValue(target.y() + 9)
    before = target.mapTo(window.caption.viewport(), QPoint()).y()
    rows[0]['zh'] = '增加的旧译文不应该移动当前阅读位置。' * 40
    window.set_conversation(rows); settle(window)
    assert abs(target.mapTo(window.caption.viewport(), QPoint()).y() - before) <= 2
    assert not window.follow_action.isChecked()
    window.follow_action.setChecked(True); settle(window)
    assert bar.value() == bar.maximum()


def test_auto_height_stays_on_screen_and_leaves_latest_chinese_above_footer(window):
    window.auto_height_action.setChecked(True)
    window.resize(800, 300)
    box = window.screen().availableGeometry()
    window.move(box.right()-window.width()+1, box.bottom()-window.height()+1)
    en = 'How do we estimate reflectance under varying illumination? ' * 20
    zh = '如何在变化的照明下估计反射特性，并保留测量不确定性？' * 50
    window.set_conversation([row('same', en, zh)]); settle(window)
    assert window.height() > 300
    assert box.contains(window.geometry())
    assert window.height() <= box.height() - 24
    pair = window.caption._cards['same'].pairs[0]
    assert 0 < bottom(pair.chinese, window.caption.viewport()) <= window.caption.viewport().height()
    assert bottom(window.caption.viewport(), window) <= window.footer.y()
    assert pair.english.text() == en.strip() and pair.chinese.text() == zh

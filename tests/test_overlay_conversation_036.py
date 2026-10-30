"""Stable, explicitly source-paired floating conversation rendering."""
import os
import time
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from PySide6.QtCore import QRect
from zoom_copilot.overlay import SubtitleOverlay


def application():
    return QApplication.instance() or QApplication([])


def records(count=2):
    return [{'entry_id':f'part-{i}', 'speaker_label':f'说话人 {i % 2 + 1}', 'state':'已成句',
             'en':f'Question {i}. A calibrated measurement.', 'zh':f'问题 {i}。校准测量。',
             'units':[(f'Question {i}.', f'问题 {i}。'), ('A calibrated measurement.', '校准测量。')]}
            for i in range(count)]


def test_overlay_keeps_stable_speaker_cards_and_explicit_source_translation_pairs():
    app=application(); window=SubtitleOverlay()
    try:
        window.show(); window.set_conversation(records()); app.processEvents()
        first=window.caption._cards['part-0']; second=window.caption._cards['part-1']
        pair=first.pairs[0]
        assert window.mode.currentData()=='captions'
        assert not window.answer.isVisible()
        assert pair.english.text()=='Question 0.' and pair.chinese.text()=='问题 0。'
        updated=records(); updated[1]['units'][1]=('A calibrated measurement.', '经过校准的测量。')
        window.set_conversation(updated); app.processEvents()
        assert window.caption._cards['part-0'] is first
        assert window.caption._cards['part-1'] is second
        assert first.pairs[0] is pair
        assert second.pairs[1].chinese.text()=='经过校准的测量。'
        assert '说话人 1' in window.caption.toPlainText() and '说话人 2' in window.caption.toPlainText()
    finally:window.hide();window.deleteLater();app.processEvents()


def test_provisional_live_words_never_replace_confirmed_overlay_history():
    app=application(); window=SubtitleOverlay()
    try:
        window.draft_action.setChecked(True)
        window.show(); window.set_conversation(records()); app.processEvents()
        stable=window.caption.toPlainText()
        first=window.caption._cards['part-0']
        window.set_caption('How could we separate', '', speaker_label='说话人 2', state='正在听译')
        app.processEvents()
        assert window.draft_action.isChecked()
        live=window.caption._cards['__prism_live_caption__']
        assert live.pairs[0].english.text()=='How could we separate'
        assert window.caption.toPlainText().startswith(stable)
        assert window.caption._cards['part-0'] is first
        window.set_caption('How could we separate illumination?', '如何分离照明？', state='已成句')
        app.processEvents()
        assert window.caption._cards['__prism_live_caption__'] is live
        assert live.pairs[0].chinese.text()=='如何分离照明？'
        window.set_conversation(records()+[{'entry_id':'confirmed','en':'How could we separate illumination?',
            'zh':'如何分离照明？'}]);app.processEvents()
        assert '__prism_live_caption__' not in window.caption._cards
        assert window.caption._cards['part-0'] is first
    finally:window.hide();window.deleteLater();app.processEvents()


def test_reader_anchor_survives_old_translation_growth_when_follow_disabled():
    app=application(); window=SubtitleOverlay()
    try:
        window.auto_height_action.setChecked(False)
        window.resize(820,420); window.show()
        rows=records(8)
        window.set_conversation(rows); QTest.qWait(30)
        window.follow_action.setChecked(False)
        target=window.caption._cards['part-3']
        window.caption.verticalScrollBar().setValue(target.y()+14)
        before=target.y()-window.caption.verticalScrollBar().value()
        rows[0]['units'][0]=('Question 0.', '校准后的误差来自测量条件，需要核对照明、噪声和散射。'*30)
        window.set_conversation(rows); QTest.qWait(30)
        assert abs(target.y()-window.caption.verticalScrollBar().value()-before)<=2
        window.follow_action.setChecked(True); app.processEvents()
        pair=window.caption._cards[window.caption._order[-1]].pairs[-1]
        assert pair.english.mapTo(window.caption.viewport(),pair.english.rect().topLeft()).y()>=0
    finally:window.hide();window.deleteLater();app.processEvents()


def test_recent_card_limit_and_long_text_are_scrollable_without_resizing_window():
    app=application(); window=SubtitleOverlay()
    try:
        window.auto_height_action.setChecked(False)
        window.resize(820,420); window.show()
        rows=records(12)
        rows[-1]['units']=[('A long technical sentence. '*90, '这是一段很长的科研问题译文。'*90)]
        before=window.size(); window.set_conversation(rows)
        # Wrapping eight cards and a very long pair requires multiple queued
        # height-for-width turns. Observe actual final geometry rather than a
        # fixed30ms delay, especially under concurrent build/full-suite load.
        deadline=time.monotonic()+1.5
        while time.monotonic()<deadline:
            QTest.qWait(10)
            reader=window.caption;bar=reader.verticalScrollBar()
            pair=reader._cards['part-11'].pairs[0]
            tail=pair.chinese.mapTo(reader.viewport(),pair.chinese.rect().bottomLeft()).y()
            if (bar.maximum()>0 and bar.value()==bar.maximum()
                    and not reader._position_timer.isActive()
                    and not reader._position_finish_timer.isActive()
                    and 0<tail<reader.viewport().height()):
                break
        assert len(window.caption._cards)==8
        assert window.caption._order[0]=='part-4'
        assert window.size()==before
        assert window.caption.verticalScrollBar().maximum()>0
        assert bar.value()==bar.maximum()
        assert 0<tail<reader.viewport().height()
        assert window.caption.horizontalScrollBar().maximum()==0
        assert 'A long technical sentence.' in window.caption.toPlainText()
        pair=window.caption._cards['part-11'].pairs[0]
        assert pair.english.text()==rows[-1]['units'][0][0].strip()
        assert pair.chinese.text()==rows[-1]['units'][0][1]
        assert not hasattr(pair,'expand')
    finally:window.hide();window.deleteLater();app.processEvents()


def test_overlay_placement_respects_explicit_secondary_screen_geometry():
    app=application();window=SubtitleOverlay()
    class Screen:
        def availableGeometry(self):return QRect(1366,40,1920,1040)
    try:
        target=Screen();box=target.availableGeometry()
        window.resize(900,500);window.move_to_top(target)
        assert window.x()>=box.x() and window.y()==box.y()+12
        assert window.x()+window.width()<=box.x()+box.width()
        window.move_to_bottom(target)
        assert window.y()+window.height()<=box.bottom()
        assert window._anchor=='bottom'
    finally:window.hide();window.deleteLater();app.processEvents()


def test_overlay_preserves_neutral_speaker_label_without_inventing_identity():
    app=application();window=SubtitleOverlay()
    try:
        window.set_conversation([{'entry_id':'unknown','en':'Could you explain the method?',
            'zh':'可以解释一下这个方法吗？','speaker_label':'当前发言 · 身份未指定','state':'已成句'}])
        assert window.caption._cards['unknown'].header.text()=='当前发言 · 身份未指定 · 已成句'
    finally:window.hide();window.deleteLater();app.processEvents()

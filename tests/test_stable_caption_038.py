"""Faithful controller-to-hidden-overlay caption publishing, without real audio.

The former default followed a new untranslated live hypothesis after each
committed card. This simulates the same public event ordering with local
translation delivery, not a hardcoded final screenshot or cloud request.
"""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from dataclasses import replace
from types import SimpleNamespace

from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from test_controller_v2 import controller
from zoom_copilot.caption_display import caption_english
from zoom_copilot.overlay import SubtitleOverlay
from zoom_copilot.ui import CopilotWindow


def application():
    return QApplication.instance() or QApplication([])


def context(c, end):
    return dict(phase_epoch=c._phase_epoch, stream_id='stable-038',
                word_timed=True, start_sample=0, end_sample=end,
                text_end_sample=end, speaker_label='面试官')


def test_controller_live_hypotheses_cannot_hide_late_committed_chinese(controller):
    c = controller
    c.pref.live_translation = False
    c.pref.caption_speaker_mode = 'single_remote'
    c._listen_pref = replace(c.pref)
    c.listening = True
    calls = []
    c.translation = SimpleNamespace(submit=lambda key, text: calls.append((key, text)), stop=lambda: None)
    c.window.show_subtitle_overlay()
    overlay = c.window._overlay
    try:
        c.on_transcript('final', 'How do you calibrate the optical system?', context(c, 16000))
        identity = c._transcripts[0]['id']
        card = overlay.caption._cards[identity]
        pair = card.pairs[0]
        original_en = pair.english.text()
        for words in ('What', 'What is', 'What is the', 'What is the uncertainty'):
            c.on_transcript('partial', words, context(c, 24000))
        # Real event ordering: translation of confirmed speech arrives while a
        # newer unconfirmed tail is being published on every decoder update.
        c.post('translation', (c._session, calls[0][0], '如何校准这套光学系统？'))
        c.pump()
        QTest.qWait(70)
        assert list(overlay.caption._cards) == [identity]
        assert overlay.caption._cards[identity] is card
        assert card.pairs[0] is pair
        assert pair.english.text() == original_en
        assert pair.chinese.text() == '如何校准这套光学系统？'
        assert pair.chinese.isVisible()
        assert '中文译文待生成' not in overlay.caption.toPlainText()
        assert not c.window.isVisible()
    finally:
        overlay.hide()


def test_unconfirmed_words_do_not_create_empty_translation_cards_or_relayout():
    app = application(); overlay = SubtitleOverlay()
    try:
        overlay.show()
        overlay.set_conversation([{'entry_id':'source', 'speaker_label':'面试官',
            'en':'Could you explain the reconstruction?', 'zh':'能解释一下重建过程吗？'}])
        QTest.qWait(80)
        card = overlay.caption._cards['source']
        revision = overlay.caption._revision
        geometry = card.geometry()
        for text in ('Can', 'Can we', 'Can we estimate', 'Can we estimate reflectance'):
            overlay.set_caption(text, speaker_label='面试官', state='正在听译 · 词句会修订')
            app.processEvents()
        assert not overlay.draft_action.isChecked()
        assert overlay.caption._revision == revision
        assert card.geometry() == geometry
        assert list(overlay.caption._cards) == ['source']
        assert 'Can we' not in overlay.caption.toPlainText()
        assert '能解释一下重建过程吗？' in overlay.caption.toPlainText()
    finally:
        overlay.hide(); overlay.deleteLater(); app.processEvents()


def test_first_speech_shows_stable_status_then_confirmed_english_without_waiting_for_zh():
    app = application(); overlay = SubtitleOverlay()
    try:
        overlay.set_caption('Could you', state='正在听译 · 词句会修订')
        assert overlay.caption.toPlainText() == ''
        assert overlay.caption_title.text() == '正在识别 · 确认后显示完整语段'
        overlay.set_caption('Could you explain your method?', state='已确认片段 · 正在更新中文')
        assert 'Could you explain your method?' in overlay.caption.toPlainText()
        pair = overlay.caption._cards['current-caption'].pairs[0]
        assert pair.english.text() == 'Could you explain your method?'
        assert pair.chinese.text() == ''
        assert pair.chinese.isHidden()
    finally:
        overlay.hide(); overlay.deleteLater(); app.processEvents()


def test_workspace_overlay_share_same_source_paired_paragraphs_and_keep_translation_prefix():
    app = application(); window = CopilotWindow()
    try:
        window.append_transcript('one', 'We estimate reflectance.\nWe retain uncertainty.',
            '我们估计反射特性，并保留不确定性。', turn_id='remote', speaker_label='面试官')
        window.append_transcript('two', 'How do you validate it?', '',
            turn_id='remote', translation_pending=True)
        window.show_subtitle_overlay()
        overlay = window._overlay
        card = overlay.caption._cards['one']
        main = window._entries['one']
        assert len(card.pairs) == len(main._unit_widgets) == 2
        for source, pair in zip(main._unit_widgets.values(), card.pairs):
            assert pair.english.text() == source.english.text()
            assert pair.chinese.text() == source.chinese.text()
        window.update_message('one', 'We estimate reflectance.\nWe retain uncertainty and validate it.',
            '我们估计反射特性，并保留不确定性。', translation_pending=True)
        assert card.pairs[0].chinese.text() == '我们估计反射特性，并保留不确定性。'
        window.update_translation('two', '如何验证？')
        assert card.pairs[1].chinese.text() == '如何验证？'
        assert main._unit_widgets['two'].chinese.text() == '如何验证？'
        assert main._segments['one'][0].count('\n') == 1
        assert caption_english(main._segments['one'][0]) == card.pairs[0].english.text()
    finally:
        window.close(); window.deleteLater(); app.processEvents()


def test_overlay_answer_pairs_stay_interleaved_and_never_guess_legacy_alignment():
    app = application(); overlay = SubtitleOverlay()
    try:
        overlay.set_answer('First.\n\nSecond.', '第一。\n\n第二。', answer_pairs=[
            {'id':'a', 'en':'First.', 'zh':'第一。'},
            {'id':'b', 'en':'Second.', 'zh':'第二。'}])
        assert overlay.answer.toPlainText() == 'First.\n第一。\n\nSecond.\n第二。'
        overlay.set_answer('Legacy one.\n\nLegacy two.', '旧译文一。\n\n旧译文二。')
        assert overlay.answer.toPlainText() == 'Legacy one.\n\nLegacy two.\n\n旧译文一。\n\n旧译文二。'
    finally:
        overlay.hide(); overlay.deleteLater(); app.processEvents()

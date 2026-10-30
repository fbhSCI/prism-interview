"""Qt dirty-region regressions; synthetic paint events, no native flicker claim."""
import os
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pytest
from PySide6.QtCore import QObject, QEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from zoom_copilot.ui import CopilotWindow


def settle(window):
    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        QTest.qWait(10)
        if (not window._timeline_follow_pending
                and not window._timeline_position_timer.isActive()
                and not window._timeline_finish_timer.isActive()):
            # Flush paints queued by the final geometry event before observing
            # the next operation's dirty regions.
            QTest.qWait(40)
            return
    pytest.fail('Caption layout did not settle')


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
def window(tmp_path, monkeypatch):
    monkeypatch.setenv('PRISM_INTERVIEW_HOME', str(tmp_path))
    app = QApplication.instance() or QApplication([])
    value = CopilotWindow()
    value.resize(1480, 950)
    value.show()
    value.reading_mode.setCurrentIndex(value.reading_mode.findData('transcript'))
    QTest.qWait(50)
    value.append_transcript('same', 'We measure light.', '我们测量光。',
        turn_id='turn', speaker_id='source', speaker_label='面试官')
    settle(value)
    yield value
    value.close()
    value.deleteLater()
    app.processEvents()


def probe_reader(window):
    probe = PaintProbe(window)
    card = window._entries['same']
    unit = card._unit_widgets['same']
    for name, widget in [('viewport', window.transcript_scroll.viewport()),
            ('card', card), ('english', unit.english), ('chinese', unit.chinese),
            ('speaker', card.speaker)]:
        probe.watch(name, widget)
    return probe, card, unit


def test_identical_controller_and_translation_callbacks_do_not_repaint_or_reposition(window):
    probe, card, unit = probe_reader(window)
    generation = window._timeline_scroll_generation
    geometry = card.geometry()
    for _ in range(4):
        window.append_transcript('same', 'We measure light.', '我们测量光。',
            turn_id='turn', speaker_id='source', speaker_label='面试官')
        window.update_translation('same', '我们测量光。')
        QTest.qWait(20)
    assert probe.events == []
    assert window._timeline_scroll_generation == generation
    assert window._entries['same'] is card
    assert card._unit_widgets['same'] is unit
    assert card.geometry() == geometry


def test_short_revisions_repaint_dirty_text_without_invalidating_entire_reader(window):
    probe, card, unit = probe_reader(window)
    viewport = window.transcript_scroll.viewport()
    old_height = unit.height()
    english_label, chinese_label = unit.english, unit.chinese
    window.append_transcript('same', 'We measure reflected light.', '我们测量光。',
        turn_id='turn', speaker_id='source', speaker_label='面试官')
    # Disabling then enabling QWidget updates invalidates all retained pixels,
    # and previously caused a full viewport repaint even for one extra word.
    assert viewport.updatesEnabled()
    settle(window)
    window.update_translation('same', '我们测量反射光。')
    assert viewport.updatesEnabled()
    settle(window)
    assert probe.events
    assert not any(name == 'viewport' and full for name, full in probe.events), probe.events
    assert unit.height() == old_height
    assert card._unit_widgets['same'] is unit
    assert unit.english is english_label and unit.chinese is chinese_label
    assert unit.english.text() == 'We measure reflected light.'
    assert unit.chinese.text() == '我们测量反射光。'


def test_unconfirmed_word_hypotheses_never_repaint_committed_conversation(window):
    probe, card, unit = probe_reader(window)
    source = unit.english.text()
    generation = window._timeline_scroll_generation
    for word in ('How', 'How do', 'How do we', 'How do we calibrate'):
        window.update_message('draft', word, partial=True, turn_id='draft-turn')
        QTest.qWait(10)
    assert probe.events == []
    assert set(window._entries) == {'same'}
    assert unit.english.text() == source
    assert window._timeline_scroll_generation == generation


def test_hidden_raw_revision_is_saved_without_repainting_formal_caption(window):
    probe, card, unit = probe_reader(window)
    window.append_transcript('same', 'We measure light.', '我们测量光。',
        turn_id='turn', speaker_id='source', speaker_label='面试官',
        raw_en='We, um, we measure light.')
    QTest.qWait(60)
    assert probe.events == []
    assert card.raw_text.text() == 'We, um, we measure light.'
    card.raw_toggle.setChecked(True)
    settle(window)
    assert card.raw_text.isVisible()
    assert unit.english.text() == 'We measure light.'

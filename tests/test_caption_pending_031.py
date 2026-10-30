"""Caption revisions keep their earlier Chinese visibly marked as incomplete."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from zoom_copilot.ui import CopilotWindow


@pytest.fixture
def window(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISM_INTERVIEW_HOME", str(tmp_path / "data"))
    application = QApplication.instance() or QApplication([])
    value = CopilotWindow()
    value.show()
    application.processEvents()
    try:
        yield value
    finally:
        value.close()
        value.deleteLater()
        application.processEvents()


@pytest.mark.parametrize("method", ["append_transcript", "update_message"])
def test_expanding_english_retains_old_chinese_with_explicit_pending(window, method):
    window.append_transcript("first", "The weights are learned", "权重通过学习获得。",
                             turn_id="remote", speaker_label="面试官")
    window.append_transcript("next", "We validate the result.", "我们验证结果。",
                             turn_id="remote")
    card = window._entries["first"]
    paragraph = card._unit_widgets["first"]
    neighbour = card._unit_widgets["next"]

    # A nonempty translation belongs to the earlier source version. It should
    # stay readable while the newly appended clause is being translated.
    getattr(window, method)("first", "The weights are learned and are back-propagated.",
        "权重通过学习获得。", translation_pending=True)
    assert paragraph is card._unit_widgets["first"]
    assert paragraph.chinese.text() == "权重通过学习获得。"
    assert paragraph.pending.isVisible()
    assert paragraph.pending.text() == "本段补充内容正在翻译…"
    assert "first" in card._pending_translations
    assert neighbour.chinese.text() == "我们验证结果。"
    assert not neighbour.pending.isVisible()

    window.update_translation("first", "权重通过学习获得，并进行反向传播。")
    assert paragraph.chinese.text() == "权重通过学习获得，并进行反向传播。"
    assert not paragraph.pending.isVisible()
    assert "first" not in card._pending_translations
    assert neighbour.chinese.text() == "我们验证结果。"


def test_explicit_pending_clear_and_legacy_empty_translation_path(window):
    window.append_transcript("first", "We estimate the field.", "我们估计场。")
    window.update_message("first", "We estimate the field and validate it.")
    card = window._entries["first"]
    paragraph = card._unit_widgets["first"]
    assert paragraph.pending.isVisible()
    assert paragraph.chinese.text() == "我们估计场。"
    window.update_message("first", "We estimate the field and validate it.",
                          "我们估计场并验证。", translation_pending=False)
    assert not paragraph.pending.isVisible()
    assert "first" not in card._pending_translations
    assert paragraph.chinese.text() == "我们估计场并验证。"


def test_caption_paragraph_wraps_naturally_without_modifying_sources(window):
    restored = "The system is learned end-to-end.\nThe field still follows a mathematical model."
    raw = "The system is learned end-to-end. The field still follows a mathematical model."
    window.append_transcript("first", restored, "系统端到端学习，场仍遵循数学模型。",
                             raw_en=raw, turn_id="remote")
    card = window._entries["first"]
    paragraph = card._unit_widgets["first"]
    assert paragraph.english.text() == restored.replace("\n", " ")
    assert paragraph.english.wordWrap()
    assert card._segments["first"][0] == restored
    assert card._raw_segments["first"] == raw
    assert paragraph.chinese.text() == "系统端到端学习，场仍遵循数学模型。"

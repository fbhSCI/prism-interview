"""Reader regressions: explicit translation identity, whole units and history."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QPoint
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from zoom_copilot.stable_text import completed_text
from zoom_copilot.ui import CopilotWindow


@pytest.fixture
def window():
    app = QApplication.instance() or QApplication([])
    value = CopilotWindow()
    value.resize(1366, 768)
    value.show()
    app.processEvents()
    yield value
    value.close()
    value.deleteLater()
    app.processEvents()


def pair(identity, english, chinese, complete=True):
    return {"id": identity, "en": english, "zh": chinese,
            "en_complete": complete, "zh_complete": complete}


@pytest.mark.parametrize("raw,expected", [
    ("A calibrated forw", ""),
    ("A calibrated model. We comp", "A calibrated model."),
    ("We use Dr. Chen's model. The next", "We use Dr. Chen's model."),
    ("The result is 3.14 and", ""),
    ("Use Fig. 2 for the result. Now", "Use Fig. 2 for the result."),
    ("We use e.g. intensity. Next", "We use e.g. intensity."),
    ("A bullet without punctuation\n\nA part", "A bullet without punctuation"),
    ("这是一整句。下一个未完", "这是一整句。"),
    ('Is this identifiable? "Yes!" New', 'Is this identifiable? "Yes!"'),
])
def test_stream_boundary_holds_partial_words_and_scientific_abbreviations(raw, expected):
    assert completed_text(raw) == expected
    assert completed_text(raw, final=True) == raw.strip()


def test_structured_paragraphs_keep_each_chinese_under_its_source_even_when_counts_differ(window):
    first = pair("measurement", "Use calibrated measurements.\n\nKeep the illumination fixed.",
                 "使用校准测量，并保持照明不变。")
    second = pair("ambiguity", "Check the remaining ambiguity.",
                  "检查剩余歧义。\n\n不要声称唯一解。")
    window.set_answer(answer_pairs=[first], streaming=True, request_id="one", question_en="Why?")
    first_widget = window._answer_pairs[0]
    window.set_answer(answer_pairs=[first, second], streaming=True, request_id="one")
    assert window._answer_pairs[0] is first_widget
    assert [(p.english.text(), p.chinese.text()) for p in window._answer_pairs] == [
        (first["en"], first["zh"]), (second["en"], second["zh"])]
    assert len(window._answer_pairs) == 2
    window.set_answer(answer_pairs=[first, second], request_id="one")
    assert window._answer_pairs[0] is first_widget


def test_flat_stream_does_not_emit_each_word_and_final_flushes_unpunctuated_tail(window):
    window.set_answer("Keep the physics. A new", streaming=True, request_id="one")
    widget = window._answer_pairs[0]
    for suffix in (" model", " is", " not", " finished"):
        window.set_answer("Keep the physics. A new" + suffix, streaming=True, request_id="one")
        assert window._answer_pairs[0] is widget
        assert widget.english.text() == "Keep the physics."
    window.set_answer("Keep the physics. A new model is not finished", request_id="one")
    assert widget.english.text() == "Keep the physics. A new model is not finished"


def test_incomplete_new_pair_does_not_auto_switch_but_complete_pair_does(window):
    old = pair("old", "Old complete answer.", "旧完整回答。")
    window.set_answer(answer_pairs=[old], question_en="Old question?", request_id="one")
    partial = pair("next", "The cali", "校", complete=False)
    window.set_answer(answer_pairs=[partial], streaming=True, question_en="Next question?", request_id="two")
    assert window._answer_displayed_id == "one"
    complete = pair("next", "The calibrated model works.", "校准后的模型可用。")
    window.set_answer(answer_pairs=[complete], streaming=True, request_id="two")
    assert window._answer_displayed_id == "two"
    assert window._answer_pairs[0].chinese.text() == complete["zh"]
    assert "Next question?" in window.question_original.text()


def test_pair_snapshots_are_owned_and_pinned_history_does_not_mix_late_translation(window):
    old = pair("a", "First answer.", "第一回答。")
    window.set_answer(answer_pairs=[old], question_en="First?", request_id="one")
    old["zh"] = "Outside mutation"
    new = pair("a", "Second answer.", "第二回答。")
    window.set_answer(answer_pairs=[new], question_en="Second?", request_id="two")
    window.answer_history_select.setCurrentIndex(0)
    widget = window._answer_pairs[0]
    assert widget.chinese.text() == "第一回答。"
    window.set_answer(answer_pairs=[pair("a", "Second answer.", "第二回答的迟到修订。")], request_id="two")
    assert window._answer_displayed_id == "one"
    assert window._answer_pairs[0] is widget
    assert widget.chinese.text() == "第一回答。"
    assert "First?" in window.question_original.text()


def test_late_pair_translation_preserves_reading_anchor_and_original_widget(window):
    first = pair("first", "First scientific paragraph. " * 100, "")
    second = pair("second", "Second scientific paragraph. " * 100, "第二段。")
    window.set_answer(answer_pairs=[first, second], streaming=True, request_id="one")
    QTest.qWait(50)
    widget = window._answer_pairs[1].english
    window.answer_scroll.verticalScrollBar().setValue(widget.mapTo(window.answer_article, QPoint(0, 0)).y())
    anchor = widget.mapTo(window.answer_scroll.viewport(), QPoint(0, 0)).y()
    translated = dict(first, zh="第一科学段落。" * 100)
    window.set_answer(answer_pairs=[translated, second], streaming=True, request_id="one")
    QTest.qWait(50)
    assert window._answer_pairs[1].english is widget
    assert widget.mapTo(window.answer_scroll.viewport(), QPoint(0, 0)).y() == anchor


def test_equal_legacy_paragraph_counts_are_not_used_as_translation_identity(window):
    en = "Measurement paragraph.\n\nInverse problem paragraph."
    zh = "未经对齐的第一块。\n\n未经对齐的第二块。"
    window.set_answer(en, zh, request_id="legacy")
    assert [(p.english.text(), p.chinese.text()) for p in window._answer_pairs] == [
        ("Measurement paragraph.", ""), ("Inverse problem paragraph.", ""), ("", zh)]


def test_explicit_pairs_keep_html_literal_and_hidden_citations_out(window):
    en = '<img src="https://untrusted.invalid/image"> Keep the forward model [27cfd86b8b24972a6b8c].'
    window.set_answer(answer_pairs=[pair("first", en, "保留物理模型。")], request_id="one")
    label = window._answer_pairs[0].english
    assert label.textFormat().name == "PlainText"
    assert '<img src="https://untrusted.invalid/image">' in label.text()
    assert "27cfd86" not in label.text()


def test_session_reset_discards_pair_widget_cache_and_never_reuses_previous_text(window):
    window.set_answer(answer_pairs=[pair("first", "Old.", "旧。")], request_id="one")
    old = window._answer_pairs[0]
    window.clear_answer_history()
    window.set_answer(answer_pairs=[pair("first", "New.", "新。")], request_id="one")
    assert window._answer_pairs[0] is not old
    assert window._answer_pairs[0].chinese.text() == "新。"


def test_final_safety_fallback_replaces_earlier_streamed_pairs_instead_of_showing_stale_claim(window):
    window.set_answer(answer_pairs=[pair("draft", "A provisional claim.", "临时结论。")],
                      streaming=True, request_id="one")
    window.set_answer("The evidence does not support that claim.", "资料不支持这一结论。",
                      request_id="one", streaming=False)
    assert window._answer_history["one"]["answer_pairs"] is None
    assert window._answer_pairs[0].english.text() == "The evidence does not support that claim."
    assert "临时" not in window._answer_pairs[0].chinese.text()


def test_early_question_chinese_is_retained_without_rebinding_old_answer(window):
    window.set_answer("Old answer.", "旧回答。", "旧问题？", question_en="Old?", request_id="old")
    window.set_question_packet({"question_en": "Why use polarization?", "answer_question_en": "Why use polarization?"})
    assert window.set_current_question_translation("Why use polarization?", "为什么使用偏振？")
    assert "中文问题" not in window.question_packet_view.toPlainText()
    assert window._question_packet["question_zh"] == "为什么使用偏振？"
    assert "Why use polarization?" in window.question_packet_view.toPlainText()
    assert "Old?" in window.question_original.text()
    assert window._answer_displayed_id == "old"
    window.set_question_packet({"question_en": "Why use polarization?"})
    assert "为什么使用偏振？" not in window.question_packet_view.toPlainText()
    assert window._question_packet["question_zh"] == "为什么使用偏振？"
    window.set_question_packet({"question_en": "What about scattering?"})
    assert not window.set_current_question_translation("Why use polarization?", "迟到的翻译。")
    assert "迟到" not in window.question_packet_view.toPlainText()
    assert "为什么" not in window.question_packet_view.toPlainText()


def test_early_question_translation_does_not_override_manual_edit_and_simple_packet_keeps_english(window):
    window.set_question("Simple question?")
    assert window.set_current_question_translation("Simple question?", "简单问题？")
    assert "Simple question?" in window.question_packet_view.toPlainText()
    window.edit_question_button.setChecked(True)
    assert not window.set_current_question_translation("Simple question?", "错误覆盖。")
    assert "错误覆盖" not in window.question_packet_view.toPlainText()

"""Bilingual paragraph protocol, verified offline without credentials/cloud."""
import json
import threading

import pytest

from zoom_copilot.answers import (AnswerService, PairedJsonStreamer, parse_reply,
    PAIRED_ANSWER_SCHEMA, build_prompt)
from zoom_copilot.codex_client import AnswerCancelled
from zoom_copilot.config import Preferences
from zoom_copilot.evidence_gap import EVIDENCE_GAP_EN
from zoom_copilot.knowledge import Passage
from test_controller_v2 import controller


def row(identity="p1", en="I would validate against an independent reference.",
        zh="我会用独立参考进行验证。", citations=None):
    return {"id": identity, "answer_en": en, "answer_zh": zh, "citations": citations or []}


def output(rows=None):
    return json.dumps({"question_zh": "如何验证重建并检查不确定性？",
        "answer_pairs": rows or [row(), row("p2", "I would report uncertainty separately.", "我会单独报告不确定性。")]
    }, ensure_ascii=False)


def test_question_is_complete_before_first_pair_and_no_token_refresh():
    events = []
    stream = PairedJsonStreamer(lambda *event: events.append(event))
    raw = output()
    for index, char in enumerate(raw):
        stream.feed(char)
        # Closed pair 1 must be visible before pair 2 even begins.
        if raw[index:index + 11] == ', {"id": "p2':
            assert [kind for kind, _ in events] == ["question_zh", "answer_pair"]
    assert [kind for kind, _ in events] == ["question_zh", "answer_pair", "answer_pair"]
    assert events[0][1] == "如何验证重建并检查不确定性？"
    assert [json.loads(value)["id"] for kind, value in events if kind == "answer_pair"] == ["p1", "p2"]
    assert all(json.loads(value)["en_complete"] and json.loads(value)["zh_complete"]
               for kind, value in events if kind == "answer_pair")


def test_stream_waits_for_matching_chinese_and_complete_row():
    events = []
    stream = PairedJsonStreamer(lambda *event: events.append(event))
    stream.feed('{"question_zh":"为什么？","answer_pairs":[{"id":"p1","answer_en":"Check noise.",')
    assert events == [("question_zh", "为什么？")]
    stream.feed('"answer_zh":"检查噪声。","citations":[]')
    assert len(events) == 1
    stream.feed('}')
    assert len(events) == 2 and json.loads(events[-1][1])["zh"] == "检查噪声。"


def test_quoted_json_field_names_escapes_and_unicode_never_fake_an_update():
    source = output([row(en='The string "question_zh":"fake" is data. Check \\ noise 😀.',
                         zh='字符串是数据。检查噪声😀。')])
    events = []
    stream = PairedJsonStreamer(lambda *event: events.append(event))
    for char in source:
        stream.feed(char)
    assert len(events) == 2 and events[0][1] != "fake"
    assert json.loads(events[1][1])["en"].endswith("😀.")


def test_explicit_pairs_preserve_sources_and_remove_unknown_citations():
    passages = [Passage("paper.pdf", "p.2", "Independent validation is required.", "known")]
    result = parse_reply(output([row(citations=["known", "made-up"]), row("p2")]), passages, strict=True)
    assert len(result.pairs) == 2
    assert result.pairs[0]["citations"] == ["known"]
    assert result.citations == ["known"] and len(result.evidence) == 1
    assert "made-up" in " ".join(result.warnings)
    assert result.answer_en == "\n\n".join(p["en"] for p in result.pairs)
    assert result.answer_zh == "\n\n".join(p["zh"] for p in result.pairs)


@pytest.mark.parametrize("bad", [
    {"id": "p1", "answer_en": "No translation.", "citations": []},
    row(zh=""), row(en="First paragraph.\n\nSecond paragraph."),
    row(en="word " * 141),
])
def test_incomplete_or_oversized_pair_never_exposed(bad):
    events = []
    raw = output([bad])
    stream = PairedJsonStreamer(lambda *event: events.append(event))
    stream.feed(raw)
    assert [kind for kind, _ in events] == ["question_zh"]
    with pytest.raises(ValueError):
        parse_reply(raw, [], strict=True)


def test_duplicate_ids_rejected_without_overwriting_a_paragraph():
    with pytest.raises(ValueError, match="重复"):
        parse_reply(output([row(), row()]), [], strict=True)


def test_legacy_flat_answer_not_guessed_into_paragraph_pairs():
    raw = json.dumps({"question_zh": "问题", "answer_en": "One.\n\nTwo.",
                      "answer_zh": "一段完整翻译。", "citations": []})
    result = parse_reply(raw, [], strict=True)
    assert not result.pairs and result.answer_en == "One.\n\nTwo."
    events = []
    stream = PairedJsonStreamer(lambda *event: events.append(event))
    stream.feed('{"question_zh":"问题","answer_en":"One')
    assert events == [("question_zh", "问题")]
    stream.feed('.","answer_zh":"完整翻译。","citations":[]}')
    assert events[-2:] == [("answer_en", "One."), ("answer_zh", "完整翻译。")]


def test_normal_service_uses_one_request_and_schema_keeps_question_first(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISM_INTERVIEW_HOME", str(tmp_path))
    prefs = Preferences(provider="codex", cloud_consent=True, prefer_prepared_answers=False)
    service, calls, events = AnswerService(prefs), [], []
    class Client:
        last_model = "offline-model"
        def complete(self, system, user, schema, on_delta, cancel_event):
            calls.append((system, user, schema))
            for char in output():
                on_delta(char)
            return output()
    monkeypatch.setattr(service, "_get_codex", lambda *_: Client())
    result = service.answer("How would you validate reconstruction?", [],
        request_id="request-1", on_delta=lambda *event: events.append(event))
    assert len(calls) == 1 and calls[0][2] is PAIRED_ANSWER_SCHEMA
    assert list(PAIRED_ANSWER_SCHEMA["properties"])[0] == "question_zh"
    assert "COMPLETE question FIRST" in calls[0][0]
    assert result.request_id == "request-1" and len(result.pairs) == 2
    assert [event[0] for event in events] == ["question_zh", "answer_pair", "answer_pair"]


def test_guard_releases_question_first_but_never_unsupported_personal_claim(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISM_INTERVIEW_HOME", str(tmp_path))
    prefs = Preferences(provider="qwen", cloud_consent=True, prefer_prepared_answers=False)
    events, calls = [], []
    raw = output([row(en="I achieved 99 percent in my completed enzyme trial.", zh="我已达到99%。")])
    class Client:
        model = last_model = "offline-model"
        def __init__(self, preferences): pass
        def complete(self, system, user, on_delta, cancel_event):
            calls.append(user)
            for char in raw:
                on_delta(char)
                assert all(kind == "question_zh" for kind, _ in events)
            return raw
    monkeypatch.setattr("zoom_copilot.providers.OpenAICompatibleClient", Client)
    result = AnswerService(prefs).answer("What did you personally measure in your completed enzyme trial?", [],
        on_delta=lambda *event: events.append(event))
    assert len(calls) == 1 and result.answer_en == EVIDENCE_GAP_EN
    assert len(result.pairs) == 1 and result.pairs[0]["id"] == "local-evidence-gap"
    assert not any("99 percent" in value for _, value in events)
    assert [kind for kind, _ in events] == ["question_zh", "answer_pair"]


def test_cancelled_pair_response_not_returned_or_retried(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISM_INTERVIEW_HOME", str(tmp_path))
    prefs = Preferences(provider="qwen", cloud_consent=True, prefer_prepared_answers=False)
    cancel, events, calls = threading.Event(), [], []
    class Client:
        model = last_model = "offline-model"
        def __init__(self, preferences): pass
        def complete(self, system, user, on_delta, cancel_event):
            calls.append(user)
            on_delta('{"question_zh":"怎样验证？",')
            cancel_event.set()
            # A misbehaving provider can still callback; service must not
            # expose post-cancel text even before the controller's ID filter.
            on_delta('"answer_pairs":[' + json.dumps(row(), ensure_ascii=False) + ']}')
            return output()
    monkeypatch.setattr("zoom_copilot.providers.OpenAICompatibleClient", Client)
    with pytest.raises(AnswerCancelled):
        AnswerService(prefs).answer("How would you validate reconstruction?", [],
            on_delta=lambda *event: events.append(event), cancel_event=cancel)
    assert len(calls) == 1 and events == [("question_zh", "怎样验证？")]


def test_controller_keeps_complete_pair_binding_and_early_question_translation(controller):
    ctrl = controller
    ctrl.window.set_answer("Old answer.", "旧答案。", "旧问题？", request_id="old", question_en="Old question?")
    question = "How would you validate reconstruction?"
    ctrl.window.set_question_packet({"question_en": question})
    ctrl.request_answer(question)
    identity = ctrl._active
    ctrl.post("answer_delta", (identity, "question_zh", "怎样验证重建？"))
    ctrl.pump()
    assert "怎样验证重建？" not in ctrl.window.question_packet_view.toPlainText()
    assert question in ctrl.window.question_packet_view.toPlainText()
    assert ctrl.window._question_packet["question_zh"] == "怎样验证重建？"
    assert ctrl.window._answer_history[identity]["question_zh"] == "怎样验证重建？"
    assert ctrl.window._answer_question_en == "Old question?"
    assert ctrl.window._answer_question_zh == "旧问题？"
    paired = parse_reply(output(), [], strict=True).pairs
    ctrl.post("answer_delta", (identity, "answer_pair", json.dumps(paired[0])))
    ctrl.pump()
    assert ctrl.window._answer_question_en == question
    assert ctrl.window._answer_question_zh == "怎样验证重建？"
    assert "怎样验证重建？" in ctrl.window.question_translation.text()
    assert ctrl.window._answer_visible_pairs == [paired[0]]
    ctrl.post("answer_delta", (identity, "answer_pair", json.dumps(paired[1])))
    ctrl.post("answer_delta", ("old", "answer_pair", json.dumps(dict(paired[0], en="STALE"))))
    ctrl.pump()
    assert ctrl._stream_pairs == paired
    assert ctrl.window._answer_visible_pairs == paired
    assert "STALE" not in ctrl.window._answer_en


def test_controller_cancelled_pairs_and_question_callbacks_do_not_leak(controller):
    ctrl = controller
    question = "How would you validate reconstruction?"
    ctrl.window.set_question_packet({"question_en": question})
    ctrl.request_answer(question)
    identity = ctrl._active
    ctrl.cancel_answer()
    pair = parse_reply(output(), [], strict=True).pairs[0]
    ctrl.post("answer_delta", (identity, "question_zh", "取消后不显示的问题？"))
    ctrl.post("answer_delta", (identity, "answer_pair", json.dumps(pair)))
    ctrl.pump()
    assert not ctrl._stream_pairs and "取消后" not in ctrl.window.question_packet_view.toPlainText()

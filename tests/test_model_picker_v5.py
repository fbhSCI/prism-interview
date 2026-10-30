"""Model switching/catalogs and opaque-citation cleanup without model inference."""
import io
import inspect
import json
import os
import threading
from urllib.error import HTTPError

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QObject, Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from zoom_copilot.answers import JsonFieldStreamer, parse_reply
from zoom_copilot.config import Preferences
from zoom_copilot.display_text import clean_answer_text
from zoom_copilot.knowledge import Passage
from zoom_copilot.model_catalog import choose_default_gpt, fetch_provider_models, supports_fast
from zoom_copilot.model_picker import ModelPickerDialog
from zoom_copilot.providers import OpenAICompatibleClient, provider_connection


@pytest.fixture(scope="module", autouse=True)
def _qt_application():
    # Keep one application alive for the entire module, and dispose every dialog
    # explicitly on the GUI thread rather than leave Python worker-thread GC to
    # reclaim cycles containing Qt wrappers.
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(autouse=True)
def _dispose_model_dialogs(_qt_application):
    yield
    for widget in _qt_application.topLevelWidgets():
        if isinstance(widget, ModelPickerDialog):
            widget.close()
            widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    _qt_application.processEvents()


def test_model_workers_never_capture_dialog_or_qobject(monkeypatch):
    captured = []
    class DeferredThread:
        def __init__(self, target, **kwargs):
            captured.append(target)
        def start(self):
            pass
    monkeypatch.setattr("zoom_copilot.model_picker.threading.Thread", DeferredThread)
    dialog = ModelPickerDialog(Preferences(provider="qwen", cloud_consent=True, api_key="test-key",
                                           answer_model="qwen3.8-flash", free_quota_confirmed=True),
                               catalog_loader=lambda prefs: [])
    dialog.refresh_catalog()
    dialog.start_answer_test()
    assert len(captured) == 2
    for worker in captured:
        closure = inspect.getclosurevars(worker).nonlocals
        assert "self" not in closure
        assert not any(isinstance(value, QObject) for value in closure.values())
    dialog.close()


def test_qwen_test_panel_fits_short_screen_with_actions_always_visible():
    dialog = ModelPickerDialog(Preferences(provider="qwen", cloud_consent=True))
    dialog.test_panel.show()
    dialog.test_output.setPlainText("English answer.\n\n中文回答。" * 20)
    dialog.resize(900, 650)
    dialog.show()
    QTest.qWait(30)
    assert dialog.height() == 650
    assert dialog.buttons.geometry().bottom() < dialog.height()
    assert dialog.status_label.geometry().bottom() < dialog.buttons.geometry().top()
    assert dialog.body_scroll.verticalScrollBar().maximum() > 0
    dialog.close()


def test_model_picker_can_maximize_minimize_and_restore_without_losing_the_selection():
    dialog = ModelPickerDialog(Preferences(provider="codex", cloud_consent=False))
    dialog._catalog_ready(dialog._generation, catalog(), "")
    dialog.show()
    QTest.qWait(25)
    dialog.model_list.setCurrentRow(1)
    selection = dialog.selected_preferences().answer_model
    assert dialog.windowFlags() & Qt.WindowType.WindowMaximizeButtonHint
    assert dialog.windowFlags() & Qt.WindowType.WindowMinimizeButtonHint
    assert dialog.isSizeGripEnabled()
    dialog.showMaximized()
    QTest.qWait(25)
    assert dialog.isMaximized()
    assert dialog.selected_preferences().answer_model == selection
    assert dialog.buttons.geometry().bottom() < dialog.height()
    dialog.showMinimized()
    QTest.qWait(25)
    assert dialog.isMinimized()
    dialog.showNormal()
    QTest.qWait(25)
    assert not dialog.isMinimized() and not dialog.isMaximized()
    assert dialog.selected_preferences().answer_model == selection
    dialog.close()


def catalog():
    return [{"id": "gpt-6.1-sol", "model": "gpt-6.1-sol", "isDefault": True,
             "serviceTiers": [{"id": "priority"}]},
            {"id": "gpt-5.6-luna", "model": "gpt-5.6-luna", "serviceTiers": [{"id": "priority"}]},
            {"model": "gpt-6-luna", "serviceTiers": [{"id": "priority"}]},
            {"model": "gpt-6-astra"}]


def test_auto_default_uses_measured_model_only_if_really_listed():
    assert choose_default_gpt(catalog()) == "gpt-5.6-luna"
    assert choose_default_gpt([catalog()[0], catalog()[2]]) == "gpt-6-luna"
    assert choose_default_gpt([catalog()[0], catalog()[3]]) == "gpt-6.1-sol"
    assert choose_default_gpt([]) == ""
    assert not supports_fast({"serviceTiers": "priority"})


def test_codex_catalog_reads_all_pages_and_refreshes_without_inference(monkeypatch):
    from test_answers_v2 import fake_codex
    client = fake_codex(monkeypatch)
    original = client._rpc
    calls = []
    def rpc(method, params, deadline, cancel_event=None):
        calls.append((method, dict(params)))
        if method == "model/list":
            if not params.get("cursor"):
                return {"data": [catalog()[0]], "nextCursor": "page-two"}
            assert params["cursor"] == "page-two"
            return {"data": [catalog()[0], catalog()[1]], "nextCursor": None}
        return original(method, params, deadline, cancel_event)
    monkeypatch.setattr(client, "_rpc", rpc)
    try:
        assert [m["model"] for m in client.models()] == ["gpt-6.1-sol", "gpt-5.6-luna"]
        client.models()
        assert len([method for method, _ in calls if method == "model/list"]) == 4
        assert not any(method in ("thread/start", "turn/start") for method, _ in calls)
    finally:
        client.close()


def test_codex_explicit_model_must_exist_in_official_catalog(monkeypatch):
    from test_answers_v2 import fake_codex
    from zoom_copilot.answers import ANSWER_SCHEMA
    from zoom_copilot.codex_client import CodexError
    client = fake_codex(monkeypatch)
    client.model = "nonexistent-model"
    original = client._rpc
    calls = []
    def rpc(method, *args, **kwargs):
        calls.append(method)
        return original(method, *args, **kwargs)
    monkeypatch.setattr(client, "_rpc", rpc)
    with pytest.raises(CodexError, match="不在当前订阅"):
        client.complete("system", "must not send", ANSWER_SCHEMA)
    assert "turn/start" not in calls and "thread/start" not in calls


def test_session_provider_keys_never_persist_and_saved_defaults_survive(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISM_INTERVIEW_HOME", str(tmp_path))
    prefs = Preferences(api_key="test-current-key", session_api_keys={"qwen": "test-qwen-key"})
    prefs.save()
    saved = (tmp_path / "preferences.json").read_text(encoding="utf-8")
    assert "test-current-key" not in saved and "test-qwen-key" not in saved
    assert "session_api_keys" not in saved
    # Even an older/manual preferences file cannot inject serialized credentials.
    (tmp_path / "preferences.json").write_text(json.dumps({"api_key": "old", "session_api_keys": {"qwen": "old"}}))
    loaded = Preferences.load()
    assert not loaded.api_key and not loaded.session_api_keys


def test_actual_citation_removed_but_sources_and_scientific_brackets_preserved():
    identity = "27cfd86b8b24972a6b8c"
    text = "Model scattering [" + identity + "]. Use [S0,S1,S2] and [0,1]."
    assert clean_answer_text(text) == "Model scattering. Use [S0,S1,S2] and [0,1]."
    raw = json.dumps({"answer_en": text, "answer_zh": "保留线偏振 [S0,S1,S2]。 [" + identity + "]",
                      "question_zh": "问题？", "citations": [identity]})
    answer = parse_reply(raw, [Passage("paper.pdf", "p.2", "evidence", identity)], strict=True)
    assert identity not in answer.answer_en and identity not in answer.answer_zh
    assert answer.citations == [identity] and answer.evidence[0]["id"] == identity
    assert "[S0,S1,S2]" in answer.answer_en


@pytest.mark.parametrize("identity", ["27cfd86b8b24972a6b8c", "cb2efbe05243cb10d734", "source-a1"])
def test_stream_cleaning_never_flashes_source_ids_or_loses_following_words(identity):
    events = []
    streamer = JsonFieldStreamer(lambda field, delta: events.append((field, delta)), [identity])
    raw = json.dumps({"answer_en": "Model scattering [" + identity + "]. Then test [S0,S1,S2].",
                      "answer_zh": "保留语义 [" + identity + "]。继续。", "question_zh": "为什么？"}, ensure_ascii=False)
    for character in raw:
        streamer.feed(character)
        en = "".join(delta for field, delta in events if field == "answer_en")
        assert identity[:4] not in en
    assert "".join(delta for field, delta in events if field == "answer_en") == "Model scattering. Then test [S0,S1,S2]."
    assert "".join(delta for field, delta in events if field == "answer_zh") == "保留语义。继续。"


def test_known_single_citation_does_not_remove_multicomponent_stokes():
    assert clean_answer_text("Use [S0,S1,S2]. Cite [S1].", citation_ids=["S1", "S2"]) == "Use [S0,S1,S2]. Cite."
    assert clean_answer_text("Paper [2024] and [a,b].") == "Paper [2024] and [a,b]."


def test_model_list_uses_read_only_get_without_free_inference_confirmation():
    observed = []
    def opener(request, timeout):
        observed.append((request.full_url, request.get_method(), request.data, request.get_header("Authorization")))
        return io.BytesIO(json.dumps({"data": [{"id": "deepseek-flash"}, {"id": "deepseek-pro"}]}).encode())
    prefs = Preferences(provider="deepseek", cloud_consent=True, api_key="test-key", free_quota_confirmed=False)
    assert [m["model"] for m in fetch_provider_models(prefs, opener=opener)] == ["deepseek-flash", "deepseek-pro"]
    assert observed == [("https://api.deepseek.com/v1/models", "GET", None, "Bearer test-key")]
    with pytest.raises(ValueError, match="免费额度"):
        OpenAICompatibleClient(prefs)


def test_qwen_models_region_and_provider_key_isolation(monkeypatch):
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    prefs = Preferences(provider="qwen", cloud_consent=True, qwen_region="ap-southeast-1",
                        api_key="test-wrong-provider", session_api_keys={"deepseek": "test-deepseek"})
    with pytest.raises(ValueError, match="API Key"):
        provider_connection(prefs, require_quota=False)
    prefs.session_api_keys["qwen"] = "test-qwen"
    observed = []
    def opener(request, timeout):
        observed.append((request.full_url, request.get_header("Authorization")))
        return io.BytesIO(json.dumps({"data": [{"id": "qwen3.8-flash"}, {"id": "qwen3.8-flash"},
                                             {"id": "qwen3.7-text-embedding"}, {"id": "qwen3.8-max"},
                                             {"id": "qwen3.8-omni-flash-realtime"}, {"id": "non-qwen"}]}).encode())
    models = fetch_provider_models(prefs, opener=opener)
    assert [m["model"] for m in models] == ["qwen3.8-flash", "qwen3.8-max"]
    assert observed == [("https://dashscope-intl.aliyuncs.com/compatible-mode/v1/models", "Bearer test-qwen")]


def test_qwen_and_deepseek_payload_disable_extra_thinking_without_real_requests(monkeypatch):
    calls = []
    from zoom_copilot import providers
    monkeypatch.setattr(providers, "stream_http", lambda request, *args, **kwargs: calls.append((request.full_url, json.loads(request.data))) or "{}")
    for provider, model in (("qwen", "qwen3.8-flash"), ("deepseek", "deepseek-flash")):
        prefs = Preferences(provider=provider, cloud_consent=True, free_quota_confirmed=True,
                            api_key="test-session-only", answer_model=model)
        OpenAICompatibleClient(prefs).complete("system", "user")
    assert calls[0][1]["enable_thinking"] is False
    assert calls[1][1]["thinking"] == {"type": "disabled"}
    assert calls[0][1]["response_format"] == {"type": "json_object"}


def test_qwen_thinking_only_models_do_not_receive_unsupported_disable_parameter():
    from zoom_copilot.providers import qwen_can_disable_thinking
    assert qwen_can_disable_thinking("qwen3.8-flash")
    assert qwen_can_disable_thinking("qwen3.8-max")
    assert not qwen_can_disable_thinking("qwen3.8-2.4t-a95b")
    assert not qwen_can_disable_thinking("qwen3.7-max-preview")
    assert not qwen_can_disable_thinking("qwen3-235b-thinking")


def test_compatible_stream_reports_service_model_without_logging_content():
    from zoom_copilot.providers import stream_http
    from urllib.request import Request
    data = b'data: {"model":"deepseek-server-snapshot","choices":[{"delta":{"content":"answer"},"finish_reason":null}]}\n\ndata: [DONE]\n'
    stats = {}
    assert stream_http(Request("https://api.deepseek.com/v1/chat/completions"), 1, "compatible", lambda text: None,
                       opener=lambda *a, **kw: io.BytesIO(data), on_stats=stats.update) == "answer"
    assert stats == {"reported_model": "deepseek-server-snapshot"}


@pytest.mark.parametrize("provider,url", [("deepseek", "https://example.org/v1"),
    ("qwen", "https://dashscope.aliyuncs.com.evil.example/v1"),
    ("qwen", "https://dashscope.aliyuncs.com/v1?key=test"),
    ("deepseek", "https://user:password@api.deepseek.com/v1")])
def test_wrong_provider_hosts_cannot_receive_session_credentials(provider, url):
    prefs = Preferences(provider=provider, api_base_url=url, api_key="test-key", cloud_consent=True)
    with pytest.raises(ValueError):
        provider_connection(prefs, require_quota=False)


def test_catalog_error_never_echoes_remote_body_or_secret():
    def opener(*args, **kwargs):
        raise HTTPError("https://api.deepseek.com/v1/models", 401, "test-body-secret", {}, None)
    with pytest.raises(RuntimeError, match="401") as error:
        fetch_provider_models(Preferences(provider="deepseek", api_key="test-key", cloud_consent=True), opener=opener)
    assert "test-body-secret" not in str(error.value)


def test_dialog_lists_all_actual_codex_models_and_filters_without_inference():
    app = QApplication.instance() or QApplication([])
    calls = []
    def loader(prefs):
        calls.append(prefs.provider)
        return catalog()
    dialog = ModelPickerDialog(Preferences(cloud_consent=True), catalog_loader=loader)
    dialog.show()
    for _ in range(20):
        QTest.qWait(15)
        if dialog.model_list.count():
            break
    assert dialog.model_list.count() == 4 and calls == ["codex"]
    assert dialog.model_input.text() == "gpt-5.6-luna"
    dialog.search_input.setText("astra")
    assert sum(not dialog.model_list.item(i).isHidden() for i in range(4)) == 1
    dialog.model_list.setCurrentRow(3)
    prefs = dialog.selected_preferences()
    assert prefs.answer_model == "gpt-6-astra" and prefs.answer_speed == "standard"
    dialog.close()


def test_dark_windows_palette_has_readable_clickable_catalog_and_saves_choice(tmp_path, monkeypatch):
    """Exercise native interaction, not only programmatic setCurrentRow()."""
    app = QApplication.instance()
    original_palette = app.palette()
    dark_palette = QPalette(original_palette)
    dark_palette.setColor(QPalette.ColorRole.Base, QColor("#181818"))
    dark_palette.setColor(QPalette.ColorRole.Text, QColor("#EEEEEE"))
    app.setPalette(dark_palette)
    # Reproduce the real parent theme: dark foreground plus the OS native base.
    parent = QWidget()
    parent.setStyleSheet("QWidget { color: #172F3A; }")
    dialog = ModelPickerDialog(Preferences(provider="codex", cloud_consent=False), parent)
    try:
        dialog._catalog_ready(dialog._generation, catalog(), "")
        dialog.show()
        QTest.qWait(30)
        assert dialog.model_list.palette().color(QPalette.ColorRole.Base).name() == "#111d2e"
        assert dialog.model_list.palette().color(QPalette.ColorRole.Text).name() == "#dde7f4"
        for index in (0, 3, 1):
            row = dialog.model_list.item(index)
            assert row.flags() & Qt.ItemFlag.ItemIsEnabled
            assert row.flags() & Qt.ItemFlag.ItemIsSelectable
            dialog.model_list.scrollToItem(row)
            QTest.qWait(10)
            QTest.mouseClick(dialog.model_list.viewport(), Qt.MouseButton.LeftButton,
                             pos=dialog.model_list.visualItemRect(row).center())
            assert dialog.model_list.currentItem() is row
            assert dialog.selected_preferences().answer_model == catalog()[index]["model"]
        dialog.model_list.setFocus()
        QTest.keyClick(dialog.model_list, Qt.Key.Key_Down)
        assert dialog.selected_preferences().answer_model == "gpt-6-luna"
        monkeypatch.setenv("PRISM_INTERVIEW_HOME", str(tmp_path))
        dialog.selected_preferences().save()
        assert Preferences.load().answer_model == "gpt-6-luna"
        # Moving focus to the action button must keep the selected row visible.
        dialog.buttons.button(dialog.buttons.StandardButton.Ok).setFocus()
        QTest.qWait(10)
        assert dialog.model_list.currentItem().isSelected()
    finally:
        dialog.close()
        dialog.deleteLater()
        parent.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
        app.setPalette(original_palette)


def test_dialog_never_carries_api_key_or_free_confirmation_to_another_provider():
    app = QApplication.instance() or QApplication([])
    dialog = ModelPickerDialog(Preferences(provider="deepseek", cloud_consent=True, api_key="test-deepseek",
                                           free_quota_confirmed=True))
    dialog.provider_select.setCurrentIndex(dialog.provider_select.findData("qwen"))
    assert dialog.key_input.text() == "" and not dialog.free_checkbox.isChecked()
    dialog.key_input.setText("test-qwen")
    dialog.provider_select.setCurrentIndex(dialog.provider_select.findData("deepseek"))
    assert dialog.key_input.text() == "test-deepseek" and dialog.free_checkbox.isChecked()
    dialog.provider_select.setCurrentIndex(dialog.provider_select.findData("qwen"))
    selected = dialog.selected_preferences()
    assert selected.api_key == "test-qwen"
    assert selected.session_api_keys == {"deepseek": "test-deepseek", "qwen": "test-qwen"}
    dialog.close()


def test_stale_catalog_does_not_replace_new_provider_selection():
    app = QApplication.instance() or QApplication([])
    waiting = threading.Event()
    started = threading.Event()
    def loader(prefs):
        started.set()
        waiting.wait(1)
        return catalog()
    dialog = ModelPickerDialog(Preferences(cloud_consent=True), catalog_loader=loader)
    dialog.show()
    QTest.qWait(20)
    assert started.is_set()
    dialog.provider_select.setCurrentIndex(dialog.provider_select.findData("qwen"))
    waiting.set()
    QTest.qWait(40)
    assert dialog.model_list.count() == 0 and dialog.provider_select.currentData() == "qwen"
    dialog.close()


def test_short_answer_requires_explicit_click_and_sends_no_personal_context(monkeypatch):
    app = QApplication.instance() or QApplication([])
    from zoom_copilot import answers
    calls = []
    class Service:
        def __init__(self, prefs):
            self.prefs = prefs
        def answer(self, question, passages, on_delta, cancel_event, history):
            calls.append((question, passages, history, self.prefs.research_profile, self.prefs.share_evidence,
                          self.prefs.candidate_profile, self.prefs.prefer_prepared_answers))
            on_delta("answer_en", "Estimate the scene [27cfd86b8b24972a6b8c].")
            on_delta("answer_zh", "估计场景。")
            return answers.BilingualAnswer("什么是逆渲染？", "Estimate the scene [27cfd86b8b24972a6b8c].", "估计场景。", [],
                                            provider="qwen", model="server-confirmed-qwen")
        def close(self):
            pass
    monkeypatch.setattr(answers, "AnswerService", Service)
    prefs = Preferences(provider="qwen", cloud_consent=True, api_key="test-qwen", answer_model="qwen3.8-flash",
                        free_quota_confirmed=True, research_profile="PRIVATE PERSONAL PROFILE",
                        candidate_profile="PRIVATE CANDIDATE CV AND PHD FACTS", prefer_prepared_answers=True)
    dialog = ModelPickerDialog(prefs)
    QTest.qWait(20)
    assert calls == []  # Merely entering a key or opening the picker is read-only.
    dialog.test_button.click()
    for _ in range(20):
        QTest.qWait(15)
        if dialog.test_button.isEnabled():
            break
    assert len(calls) == 1 and calls[0][1:3] == ([], [])
    assert "PRIVATE" not in calls[0][3] and calls[0][4] is False
    assert calls[0][5:] == ("", False)
    assert dialog.selected_preferences().candidate_profile == "PRIVATE CANDIDATE CV AND PHD FACTS"
    assert dialog.selected_preferences().prefer_prepared_answers is True
    assert prefs.candidate_profile == "PRIVATE CANDIDATE CV AND PHD FACTS"
    assert "server-confirmed-qwen" in dialog.test_metrics.text()
    assert "首段英文" in dialog.test_metrics.text()
    assert dialog.test_output.toPlainText() == "Estimate the scene.\n\n估计场景。"
    dialog.close()


def test_short_answer_blocks_api_without_free_confirmation_before_service(monkeypatch):
    app = QApplication.instance() or QApplication([])
    from zoom_copilot import answers
    calls = []
    monkeypatch.setattr(answers, "AnswerService", lambda prefs: calls.append(prefs))
    dialog = ModelPickerDialog(Preferences(provider="deepseek", cloud_consent=True, api_key="test-deepseek",
                                           answer_model="deepseek-flash", free_quota_confirmed=False))
    dialog.test_button.click()
    QTest.qWait(20)
    assert calls == [] and "免费额度" in dialog.status_label.text()
    dialog.close()


def test_short_answer_cancel_ignores_late_model_output(monkeypatch):
    app = QApplication.instance() or QApplication([])
    from zoom_copilot import answers
    started = threading.Event()
    release = threading.Event()
    cancelled = []
    class Service:
        def __init__(self, prefs):
            pass
        def answer(self, question, passages, on_delta, cancel_event, history):
            started.set()
            release.wait(1)
            cancelled.append(cancel_event.is_set())
            on_delta("answer_en", "late text")
            return answers.BilingualAnswer("问题", "late text", "迟到内容", [], provider="deepseek", model="test")
        def close(self):
            pass
    monkeypatch.setattr(answers, "AnswerService", Service)
    dialog = ModelPickerDialog(Preferences(provider="deepseek", cloud_consent=True, api_key="test-key",
                                           answer_model="deepseek-flash", free_quota_confirmed=True))
    dialog.test_button.click()
    for _ in range(20):
        QTest.qWait(10)
        if started.is_set():
            break
    assert started.is_set()
    dialog.cancel_test_button.click()
    release.set()
    QTest.qWait(40)
    assert cancelled == [True]
    assert "late" not in dialog.test_output.toPlainText()
    assert dialog.test_button.isEnabled()
    dialog.close()

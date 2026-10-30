"""Masked default-key feedback with exact provider/endpoint isolation."""
import json
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QLineEdit

from zoom_copilot import credential_store as vault
from zoom_copilot.config import Preferences
from zoom_copilot.model_picker import ModelPickerDialog


QWEN = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEEPSEEK = "https://api.deepseek.com/v1"
SINGAPORE = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


@pytest.fixture
def setup_picker(monkeypatch):
    app = QApplication.instance() or QApplication([])
    saved = {("qwen", QWEN): "test-qwen-default", ("deepseek", DEEPSEEK): "test-deepseek-default"}
    reads = []
    writes = []
    dialogs = []

    def load(provider, endpoint):
        reads.append((provider, endpoint))
        return saved.get((provider, endpoint), "")

    def store(provider, endpoint, key):
        writes.append((provider, endpoint))
        saved[(provider, endpoint)] = key

    monkeypatch.setattr(vault, "load_key", load)
    monkeypatch.setattr(vault, "store_key", store)
    monkeypatch.setattr(vault, "delete_key", lambda provider, endpoint: saved.pop((provider, endpoint), None))

    def create(**kwargs):
        values = dict(provider="qwen", api_base_url=QWEN, cloud_consent=False, answer_model="qwen3.8-flash")
        values.update(kwargs)
        dialog = ModelPickerDialog(Preferences(**values))
        dialogs.append(dialog)
        return dialog

    yield create, saved, reads, writes
    for dialog in dialogs:
        dialog.close()
        dialog.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()


def test_existing_default_is_masked_visible_and_does_not_become_session_key(setup_picker):
    create, saved, reads, writes = setup_picker
    dialog = create()
    assert dialog.key_input.echoMode() == QLineEdit.EchoMode.Password
    assert dialog.key_input.text() == saved[("qwen", QWEN)]
    assert saved[("qwen", QWEN)] not in dialog.key_input.displayText()
    assert "已自动填入" in dialog.key_status_label.text()
    assert saved[("qwen", QWEN)] not in dialog.key_status_label.text() + dialog.status_label.text()
    selected = dialog.selected_preferences()
    assert selected.api_key == "" and selected.session_api_keys == {}
    assert selected.cloud_consent is False and selected.free_quota_confirmed is False
    assert reads == [("qwen", QWEN)] and writes == []


def test_switch_provider_loads_only_its_saved_default(setup_picker):
    create, saved, reads, writes = setup_picker
    dialog = create()
    dialog.provider_select.setCurrentIndex(dialog.provider_select.findData("deepseek"))
    assert dialog.endpoint_input.text() == DEEPSEEK
    assert dialog.key_input.text() == saved[("deepseek", DEEPSEEK)]
    dialog.provider_select.setCurrentIndex(dialog.provider_select.findData("qwen"))
    assert dialog.key_input.text() == saved[("qwen", QWEN)]
    assert reads == [("qwen", QWEN), ("deepseek", DEEPSEEK), ("qwen", QWEN)]
    assert not dialog.selected_preferences().session_api_keys and writes == []


def test_endpoint_edit_immediately_clears_old_default_and_loads_only_exact_new_target(setup_picker):
    create, saved, reads, writes = setup_picker
    saved[("qwen", SINGAPORE)] = "test-singapore-default"
    dialog = create()
    dialog.endpoint_input.setText("https://other.example.com/v1")
    assert dialog.key_input.text() == ""
    assert not dialog.selected_preferences().api_key
    dialog.endpoint_input.setText(SINGAPORE)
    assert dialog.key_input.text() == "test-singapore-default"
    dialog.endpoint_input.setText(QWEN)
    assert dialog.key_input.text() == "test-qwen-default"
    assert saved[("qwen", QWEN)] == "test-qwen-default" and writes == []


def test_typed_session_key_stays_with_exact_endpoint_and_provider(setup_picker):
    create, saved, reads, writes = setup_picker
    dialog = create(api_key="test-session-qwen")
    assert dialog.key_input.text() == "test-session-qwen"
    dialog.endpoint_input.setText(SINGAPORE)
    assert dialog.key_input.text() == ""
    assert not dialog.selected_preferences().api_key
    dialog.key_input.setText("test-session-singapore")
    dialog.endpoint_input.setText(QWEN)
    assert dialog.key_input.text() == "test-session-qwen"
    dialog.provider_select.setCurrentIndex(dialog.provider_select.findData("deepseek"))
    assert dialog.key_input.text() == "test-deepseek-default"
    dialog.provider_select.setCurrentIndex(dialog.provider_select.findData("qwen"))
    dialog.endpoint_input.setText(SINGAPORE)
    assert dialog.key_input.text() == "test-session-singapore"
    assert dialog.selected_preferences().api_key == "test-session-singapore"
    assert writes == []


def test_region_switch_refreshes_masked_default_for_the_new_endpoint(setup_picker):
    create, saved, reads, writes = setup_picker
    saved[("qwen", SINGAPORE)] = "test-singapore-default"
    dialog = create()
    dialog.region_select.setCurrentIndex(dialog.region_select.findData("ap-southeast-1"))
    assert dialog.endpoint_input.text() == SINGAPORE
    assert dialog.key_input.text() == "test-singapore-default"
    assert dialog.selected_preferences().api_key == ""
    assert reads[-1] == ("qwen", SINGAPORE) and writes == []


def test_store_keeps_masked_feedback_and_delete_clears_only_current_target(setup_picker):
    create, saved, reads, writes = setup_picker
    dialog = create()
    dialog.key_input.setText("test-qwen-replacement")
    dialog.remember_key_button.click()
    assert saved[("qwen", QWEN)] == "test-qwen-replacement"
    assert dialog.key_input.text() == "test-qwen-replacement"
    assert "已自动填入" in dialog.key_status_label.text()
    assert not dialog.selected_preferences().api_key
    dialog.forget_key_button.click()
    assert ("qwen", QWEN) not in saved
    assert saved[("deepseek", DEEPSEEK)] == "test-deepseek-default"
    assert dialog.key_input.text() == ""
    assert "尚未保存" in dialog.key_status_label.text()
    assert not dialog.forget_key_button.isEnabled()
    assert writes == [("qwen", QWEN)]


def test_reopening_picker_restores_saved_key_and_normalized_address_keeps_scope(setup_picker):
    create, saved, reads, writes = setup_picker
    dialog = create()
    dialog.key_input.setText("test-new-default")
    dialog.remember_key_button.click()
    reopened = create()
    assert reopened.key_input.text() == "test-new-default"
    assert "已自动填入" in reopened.key_status_label.text()
    before = list(reads)
    reopened.endpoint_input.setText("https://DASHSCOPE.ALIYUNCS.COM:443/compatible-mode/v1/")
    assert reopened.key_input.text() == "test-new-default"
    assert reads == before
    assert not reopened.selected_preferences().api_key


def test_cleared_password_explains_saved_default_fallback_without_reclassifying_it(setup_picker):
    create, saved, reads, writes = setup_picker
    dialog = create()
    dialog.key_input.clear()
    assert "留空调用时仍会使用" in dialog.key_status_label.text()
    assert dialog.forget_key_button.isEnabled()
    assert dialog.selected_preferences().session_api_keys == {}
    assert saved[("qwen", QWEN)] == "test-qwen-default" and writes == []


@pytest.mark.parametrize("endpoint", ["", "http://example.com/v1", "https://u:p@example.com/v1",
                                    "https://example.com/v1?key=value", "https://example.com:9443/v1"])
def test_invalid_endpoint_clears_password_without_reading_vault(setup_picker, endpoint):
    create, saved, reads, writes = setup_picker
    dialog = create()
    reads.clear()
    dialog.endpoint_input.setText(endpoint)
    assert dialog.key_input.text() == "" and reads == []
    assert not dialog.remember_key_button.isEnabled() and not dialog.forget_key_button.isEnabled()


def test_loaded_default_never_enters_preferences_file(setup_picker, tmp_path, monkeypatch):
    create, saved, reads, writes = setup_picker
    monkeypatch.setenv("PRISM_INTERVIEW_HOME", str(tmp_path))
    dialog = create()
    dialog.selected_preferences().save()
    text = (tmp_path / "preferences.json").read_text(encoding="utf8")
    assert all(secret not in text for secret in saved.values())
    value = json.loads(text)
    assert "api_key" not in value and "session_api_keys" not in value


def test_vault_error_is_generic_and_previous_address_key_is_cleared(setup_picker, monkeypatch):
    create, saved, reads, writes = setup_picker
    dialog = create()
    def fail(*_):
        raise OSError("test-private-error-detail")
    monkeypatch.setattr(vault, "load_key", fail)
    dialog.endpoint_input.setText(SINGAPORE)
    assert dialog.key_input.text() == ""
    assert "未能读取" in dialog.key_status_label.text()
    assert "test-private-error-detail" not in dialog.key_status_label.text()


def test_codex_and_ollama_do_not_read_api_credential_targets(setup_picker):
    create, saved, reads, writes = setup_picker
    dialog = create(provider="codex", api_base_url="")
    assert reads == []
    dialog.provider_select.setCurrentIndex(dialog.provider_select.findData("ollama"))
    assert reads == [] and dialog.key_input.text() == ""

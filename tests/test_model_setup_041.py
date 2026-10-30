"""Public model preparation stays off the UI thread and never starts capture."""
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QCoreApplication, QEvent
from PySide6.QtWidgets import QApplication, QWidget

from zoom_copilot import model_setup, setup_models
from zoom_copilot.config import Preferences


@pytest.fixture
def parent():
    qt = QApplication.instance() or QApplication([])
    window = QWidget()
    yield qt, window
    dialog = getattr(window, "_model_setup_dialog", None)
    if dialog is not None and dialog.thread:
        dialog.thread.join(2)
    window.close()
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qt.processEvents()


def eventually(qt, predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not predicate():
        qt.processEvents()
        time.sleep(.005)
    qt.processEvents()
    assert predicate()


def test_show_prepares_only_public_speech_models_on_worker_thread(parent, tmp_path, monkeypatch):
    qt, window = parent
    main_thread = threading.get_ident()
    calls = []
    gui_threads = []
    def prepare(root, **kwargs):
        calls.append((root, kwargs, threading.get_ident()))
        kwargs["on_status"]("正在下载公开测试模型")
    monkeypatch.setattr(model_setup, "prepare", prepare)
    preferences = Preferences(model_cache_dir=str(tmp_path / "selected-models"), streaming_captions=True)
    dialog = model_setup.show_model_setup(window, preferences, auto_start=False)
    original = dialog.log.appendPlainText
    def append(message):
        gui_threads.append(threading.get_ident())
        original(message)
    monkeypatch.setattr(dialog.log, "appendPlainText", append)
    results = []
    dialog.prepared.connect(results.append)
    assert calls == []  # simply building a window does not access the network
    dialog.start_button.click()
    eventually(qt, lambda: bool(calls) and not dialog.active)
    root, options, worker_thread = calls[0]
    assert worker_thread != main_thread and gui_threads and set(gui_threads) == {main_thread}
    assert root == (tmp_path / "selected-models").resolve()
    assert options["asr"] == "large-v3-turbo" and options["translation"] is False
    assert options["streaming"] is True and options["punctuation"] is True
    assert results == [str(root)] and "准备好" in dialog.status_label.text()


def test_active_task_is_reused_and_close_only_hides_without_second_download(parent, tmp_path, monkeypatch):
    qt, window = parent
    started, release = threading.Event(), threading.Event()
    calls = []
    def prepare(root, **kwargs):
        calls.append(root)
        started.set()
        release.wait(2)
        kwargs["on_status"]("公开模型校验完成")
    monkeypatch.setattr(model_setup, "prepare", prepare)
    pref = Preferences(model_cache_dir=str(tmp_path))
    dialog = model_setup.show_model_setup(window, pref)
    eventually(qt, started.is_set)
    assert dialog.active and not dialog.start_button.isEnabled()
    dialog.close()
    assert not dialog.isVisible() and dialog.active
    assert model_setup.show_model_setup(window, pref) is dialog
    assert dialog.isVisible() and calls == [tmp_path.resolve()]
    dialog.start()
    assert len(calls) == 1
    release.set()
    eventually(qt, lambda: not dialog.active)
    assert dialog.start_button.isEnabled() and "公开模型校验完成" in dialog.log.toPlainText()


def test_error_message_does_not_echo_exception_bodies_and_retry_is_possible(parent, tmp_path, monkeypatch):
    qt, window = parent
    calls = []
    def prepare(root, **kwargs):
        calls.append(root)
        if len(calls) == 1:
            raise RuntimeError("private-example-response-body")
        kwargs["on_status"]("公开模型准备完成")
    monkeypatch.setattr(model_setup, "prepare", prepare)
    dialog = model_setup.show_model_setup(window, Preferences(model_cache_dir=str(tmp_path)))
    eventually(qt, lambda: len(calls) == 1 and not dialog.active)
    assert "准备失败" in dialog.status_label.text() and dialog.start_button.text() == "重试"
    assert "private-example-response-body" not in dialog.log.toPlainText() + dialog.status_label.text()
    first_generation = dialog._generation
    dialog.start_button.click()
    eventually(qt, lambda: len(calls) == 2 and not dialog.active)
    text = dialog.log.toPlainText()
    dialog._progress(first_generation, "stale status")
    dialog._complete(first_generation, False)
    assert dialog.log.toPlainText() == text and "准备好" in dialog.status_label.text()


def test_destroyed_parent_does_not_receive_worker_widget_calls(parent, tmp_path, monkeypatch):
    qt, window = parent
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    def prepare(root, **kwargs):
        started.set()
        release.wait(2)
        kwargs["on_status"]("公开模型校验结束")
        finished.set()
    monkeypatch.setattr(model_setup, "prepare", prepare)
    dialog = model_setup.show_model_setup(window, Preferences(model_cache_dir=str(tmp_path)))
    eventually(qt, started.is_set)
    window.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qt.processEvents()
    release.set()
    eventually(qt, finished.is_set)
    dialog.thread.join(2)
    assert not dialog.thread.is_alive()
    # Fixture teardown must not call methods on the now-deleted widget.
    monkeypatch.setattr(window, "close", lambda: None)
    monkeypatch.setattr(window, "deleteLater", lambda: None)


def test_preparation_does_not_read_cloud_keys_and_resolves_default_model_cache(parent, tmp_path, monkeypatch):
    qt, window = parent
    class Choices:
        model_cache_dir = ""
        whisper_size = "medium.en"
        streaming_captions = False
        @property
        def api_key(self): pytest.fail("Model preparation must not read credentials")
        @property
        def cloud_consent(self): pytest.fail("Model downloads do not use a cloud answer")
    monkeypatch.setenv("PRISM_INTERVIEW_MODELS", str(tmp_path / "default-models"))
    calls = []
    monkeypatch.setattr(model_setup, "prepare", lambda root, **kwargs: calls.append((root, kwargs)))
    dialog = model_setup.show_model_setup(window, Choices())
    eventually(qt, lambda: bool(calls) and not dialog.active)
    assert calls[0][0] == (tmp_path / "default-models").resolve()
    assert calls[0][1]["asr"] == "medium.en" and calls[0][1]["streaming"] is False
    assert calls[0][1]["translation"] is False


def test_cached_asr_files_do_not_require_network_or_credentials(tmp_path, monkeypatch):
    target = tmp_path / "asr/large-v3-turbo"
    target.mkdir(parents=True)
    for name in ("model.bin", "config.json", "tokenizer.json"):
        (target / name).write_bytes(b"cached fixture")
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda *args, **kwargs: pytest.fail("Cached models must stay offline"))
    monkeypatch.setattr(setup_models, "_preserve_whisper_notice", lambda *args: pytest.fail("No extra license network on cached files"))
    statuses = []
    result = setup_models.prepare(tmp_path, translation=False, on_status=statuses.append)
    assert result["asr"] == "large-v3-turbo" and any("已找到" in text for text in statuses)


def test_new_asr_download_includes_licenses_and_never_uses_account_auth(tmp_path, monkeypatch):
    import huggingface_hub
    calls = []
    def snapshot(repository, *, local_dir, token, allow_patterns, max_workers):
        calls.append((repository, token, allow_patterns, max_workers))
        target = Path(local_dir); target.mkdir(parents=True)
        for name in ("model.bin", "config.json", "tokenizer.json"):
            (target / name).write_bytes(b"public fixture")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    notices = []
    monkeypatch.setattr(setup_models, "_preserve_whisper_notice", lambda target, repo, on_status: notices.append(repo))
    monkeypatch.setattr(setup_models.shutil, "disk_usage", lambda _: SimpleNamespace(free=8*1024**3))
    setup_models.prepare(tmp_path, translation=False, on_status=lambda _: None)
    assert len(calls) == 1 and calls[0][1] is False
    assert "LICENSE.*" in calls[0][2] and "README.md" in calls[0][2] and notices == [calls[0][0]]


def test_primary_whisper_license_is_preserved_without_replacing_existing_notice(tmp_path, monkeypatch):
    from urllib import request
    license_text = "MIT License\nPermission is hereby granted, free of charge, to any person.\n"
    urls = []
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, limit): return license_text.encode("utf8")
    def open_url(url, timeout):
        urls.append(url)
        assert timeout == 20
        return Response()
    monkeypatch.setattr(request, "urlopen", open_url)
    setup_models._preserve_whisper_notice(tmp_path, "public/example-model", lambda _: None)
    assert (tmp_path / "WHISPER_LICENSE.txt").read_text(encoding="utf8") == license_text
    assert "https://huggingface.co/public/example-model" in (tmp_path / "MODEL_SOURCE.txt").read_text()
    setup_models._preserve_whisper_notice(tmp_path, "public/example-model", lambda _: None)
    assert urls == ["https://raw.githubusercontent.com/openai/whisper/main/LICENSE"]


def test_license_network_error_is_bounded_and_does_not_expose_response_body(tmp_path, monkeypatch):
    from urllib import request
    def failed(*args, **kwargs): raise RuntimeError("private-example-response-body")
    monkeypatch.setattr(request, "urlopen", failed)
    statuses = []
    setup_models._preserve_whisper_notice(tmp_path, "public/example-model", statuses.append)
    assert (tmp_path / "MODEL_SOURCE.txt").is_file()
    assert not (tmp_path / "WHISPER_LICENSE.txt").exists()
    assert statuses and "许可证" in statuses[-1] and "private-example-response-body" not in statuses[-1]


def test_windowed_model_download_disables_real_nested_hf_tqdm_without_changing_environment(tmp_path, monkeypatch):
    import sys
    import huggingface_hub
    from huggingface_hub.utils import tqdm as hf_tqdm, are_progress_bars_disabled, enable_progress_bars, disable_progress_bars
    previous_disabled = are_progress_bars_disabled()
    environment_before = dict(os.environ)
    bars = []
    def snapshot(repository, *, local_dir, token, allow_patterns, max_workers, tqdm_class):
        assert sys.stderr is None and token is False and issubclass(tqdm_class, hf_tqdm)
        # Exercise real Hub/tqdm objects for the outer snapshot counter and
        # nested file progress. No remote API or model file is accessed.
        with tqdm_class(total=3, desc="outer public fixture") as outer:
            with hf_tqdm(total=8, desc="inner public fixture") as inner:
                outer.update(3)
                inner.update(8)
                assert outer.disable and inner.disable
                bars.extend([outer, inner])
        target = Path(local_dir); target.mkdir(parents=True)
        for name in ("model.bin", "config.json", "tokenizer.json"):
            (target / name).write_bytes(b"public fixture")
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot)
    monkeypatch.setattr(setup_models, "_preserve_whisper_notice", lambda *args: None)
    monkeypatch.setattr(setup_models.shutil, "disk_usage", lambda _: SimpleNamespace(free=8*1024**3))
    try:
        with monkeypatch.context() as patch:
            patch.setattr(sys, "stderr", None)
            setup_models.prepare(tmp_path, translation=False, on_status=lambda _: None)
        assert len(bars) == 2 and dict(os.environ) == environment_before
    finally:
        if previous_disabled:
            disable_progress_bars()
        else:
            enable_progress_bars()

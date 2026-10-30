"""NLLB integration invariants. Real translation quality is assessed separately."""
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from zoom_copilot import translation_nllb as nllb


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        return SimpleNamespace(tokens=text.split())

    def token_to_id(self, token):
        return {"中文": 42}.get(token)

    def decode(self, ids, skip_special_tokens=True):
        assert ids == [42] and skip_special_tokens
        return "中文"


def loaded_translator(tmp_path):
    translator = nllb.NllbTranslator(tmp_path)
    translator._tokenizer = FakeTokenizer()

    class Engine:
        def translate_batch(self, sources, **kwargs):
            assert sources == [[nllb.SOURCE_LANGUAGE, "Hello", "world?", "</s>"]]
            assert kwargs["target_prefix"] == [[nllb.TARGET_LANGUAGE]]
            return [SimpleNamespace(hypotheses=[[nllb.TARGET_LANGUAGE, "中文", "</s>"]])]

    translator._translator = Engine()
    return translator


def test_explicit_english_prefix_simplified_chinese_target_and_decode(tmp_path):
    translator = loaded_translator(tmp_path)
    assert translator.translate("Hello world?") == "中文"
    assert translator.last_metrics["source_tokens"] == 4
    assert translator.last_metrics["device"] == "cpu"
    assert translator.last_metrics["cold_load_included"] is False


def test_empty_input_does_not_load_or_download(tmp_path):
    translator = nllb.NllbTranslator(tmp_path)
    assert translator.translate("  ") == ""
    assert not tmp_path.joinpath(nllb.MANIFEST).exists()


def test_missing_model_is_explicit_failure(tmp_path):
    translator = nllb.NllbTranslator(tmp_path)
    assert not translator.ready
    with pytest.raises(RuntimeError, match="尚未安装"):
        translator.translate("What is polarization?")
    assert "尚未安装" in translator.last_error


def test_never_silently_truncates_a_long_sentence(tmp_path):
    translator = nllb.NllbTranslator(tmp_path)
    translator._tokenizer = FakeTokenizer()
    with pytest.raises(ValueError, match="未静默截断"):
        translator._source_tokens("term " * 501)


def test_ready_requires_complete_pinned_install(tmp_path, monkeypatch):
    monkeypatch.setattr(nllb, "FILES", {"model.bin": {"size": 3}, "tokenizer.json": {"size": 2}})
    translator = nllb.NllbTranslator(tmp_path)
    (tmp_path / "model.bin").write_bytes(b"abc")
    (tmp_path / "tokenizer.json").write_bytes(b"{}")
    assert not translator.ready
    (tmp_path / nllb.MANIFEST).write_text(json.dumps({"model_id": nllb.MODEL_ID, "revision": "wrong"}), encoding="utf-8")
    assert not translator.ready
    (tmp_path / nllb.MANIFEST).write_text(json.dumps({"model_id": nllb.MODEL_ID, "revision": nllb.REVISION}), encoding="utf-8")
    assert translator.ready
    (tmp_path / "model.bin").write_bytes(b"x")
    assert not translator.ready


def test_resume_restarts_when_server_ignores_range(tmp_path, monkeypatch):
    data = b"complete"
    translator = nllb.NllbTranslator(tmp_path)
    (tmp_path / "test.bin.download").write_bytes(b"com")

    class Response(io.BytesIO):
        status = 200

    def open_request(request, timeout):
        assert nllb.REVISION in request.full_url
        assert request.get_header("Range") == "bytes=3-"
        return Response(data)

    monkeypatch.setattr(nllb, "urlopen", open_request)
    result = translator._download("test.bin", {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}, lambda _: None)
    assert (tmp_path / "test.bin").read_bytes() == data
    assert result["sha256"] == hashlib.sha256(data).hexdigest()


def test_corrupt_download_is_not_installed(tmp_path, monkeypatch):
    translator = nllb.NllbTranslator(tmp_path)

    class Response(io.BytesIO):
        status = 200

    monkeypatch.setattr(nllb, "urlopen", lambda *args, **kwargs: Response(b"bad"))
    with pytest.raises(RuntimeError, match="校验失败"):
        translator._download("test.bin", {"size": 3, "sha256": hashlib.sha256(b"yes").hexdigest()}, lambda _: None)
    assert not (tmp_path / "test.bin").exists()
    assert not translator.ready


def test_install_manifest_and_license_only_after_files_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(nllb, "FILES", {"model.bin": {"size": 3}})
    translator = nllb.NllbTranslator(tmp_path)

    def download(name, meta, status):
        (tmp_path / name).write_bytes(b"abc")
        return {"size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}

    monkeypatch.setattr(translator, "_download", download)
    messages = []
    assert "已安装" in translator.install(messages.append)
    assert translator.ready
    assert any("非商业" in message for message in messages)
    assert "CC-BY-NC-4.0" in (tmp_path / "MODEL_LICENSE_NOTICE.txt").read_text(encoding="utf-8")
    assert json.loads((tmp_path / nllb.MANIFEST).read_text(encoding="utf-8"))["revision"] == nllb.REVISION


def test_installer_failure_leaves_no_ready_manifest(tmp_path, monkeypatch):
    translator = nllb.NllbTranslator(tmp_path)

    def failed(*args):
        raise OSError("network unavailable")

    monkeypatch.setattr(translator, "_download", failed)
    with pytest.raises(OSError):
        translator.install()
    assert not translator.ready
    assert not (tmp_path / nllb.MANIFEST).exists()


def test_invalid_tokenizer_output_is_explicit_failure(tmp_path):
    translator = loaded_translator(tmp_path)
    translator._translator = SimpleNamespace(translate_batch=lambda *args, **kwargs: [
        SimpleNamespace(hypotheses=[[nllb.TARGET_LANGUAGE, "not-in-vocabulary"]])])
    with pytest.raises(RuntimeError, match="未知标记"):
        translator.translate("Hello world?")


def test_invalid_manifest_type_is_not_ready(tmp_path):
    (tmp_path / nllb.MANIFEST).write_text("[]", encoding="utf-8")
    assert not nllb.NllbTranslator(tmp_path).ready


def test_output_at_generation_limit_is_not_silently_accepted(tmp_path):
    translator = loaded_translator(tmp_path)
    translator._translator = SimpleNamespace(translate_batch=lambda *args, **kwargs: [
        SimpleNamespace(hypotheses=[[nllb.TARGET_LANGUAGE] + ["中文"] * 511])])
    with pytest.raises(RuntimeError, match="可能不完整"):
        translator.translate("Hello world?")

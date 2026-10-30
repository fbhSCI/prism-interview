"""Exercise the exporter with tiny structural fixtures, never model inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


pytestmark = pytest.mark.skipif(sys.platform != "win32" or not shutil.which("powershell.exe"),
                                reason="Windows PowerShell exporter")
SCRIPT = Path(__file__).resolve().parents[1] / "export_runtime.ps1"


@pytest.fixture
def export_fixture(tmp_path):
    models = tmp_path / "models"
    cuda = tmp_path / "cuda"
    target = tmp_path / "portable with spaces"
    def put(root, name, content=b"structural-test-fixture"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    for name in ("model.bin", "tokenizer.json"):
        put(models, "asr/large-v3-turbo/" + name)
    put(models, "asr/large-v3-turbo/config.json", b'{"model_type":"whisper"}')
    argos = "translation/translate-en_zh-test/"
    put(models, argos + "metadata.json", b'{"from_code":"en","to_code":"zh"}')
    for name in ("sentencepiece.model", "model/model.bin", "model/config.json"):
        put(models, argos + name)
    for lib in ("cublas", "cudnn", "cuda_nvrtc"):
        put(cuda, f"nvidia/{lib}/bin/{lib}-test.dll")
    for root in (models, cuda):
        for name in ("auth.json", "knowledge.json", "my-paper.pdf", "transcript.txt"):
            put(root, name, b"must never be exported")
    put(models, "asr/large-v3-turbo/auth.json", b"must never be exported")
    put(models, argos + "auth.json", b"must never be exported")
    put(models, "translation-nllb/model.bin", b"not selected by default")
    streaming = "streaming-asr/fastconformer-ctc-en-480ms/"
    for name in ("model.int8.onnx", "tokens.txt"):
        put(models, streaming + name)
    put(models, streaming + "model.json", json.dumps({
        "repository": "csukuangfj/sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-480ms-int8",
        "revision": "66e9b61a9fe334eba579700599e391c6ee093633", "license": "CC-BY-4.0"
    }).encode())
    punctuation = "punctuation/edge-punct-en/"
    for name in ("model.int8.onnx", "bpe.vocab", "LICENSE"):
        put(models, punctuation + name)
    put(models, punctuation + "model.json", json.dumps({
        "archive_sha256": "9f5e5a72c7d2829635bd074fce92b6bbd5b78da8a52e7ad8ed1be933f366b99d",
        "license": "Apache-2.0"
    }).encode())
    for folder in (streaming, punctuation):
        put(models, folder + "auth.json", b"must never be exported")
        put(models, folder + "notes.txt", b"private notes must never be exported")
    return models, cuda, target


def run_export(models, cuda, target, *flags):
    return subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass",
                           "-File", str(SCRIPT), "-ModelDirectory", str(models),
                           "-CudaDirectory", str(cuda), "-DestinationDirectory", str(target), *flags],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)


def test_export_copies_only_whitelist_and_verifies_manifest(export_fixture):
    models, cuda, target = export_fixture
    result = run_export(models, cuda, target)
    assert result.returncode == 0, result.stdout + result.stderr
    config = json.loads((target / "runtime.json").read_text(encoding="utf-8"))
    assert config == {"data_dir": "data", "model_dir": "runtime/models", "cuda_dir": "runtime/gpu"}
    manifest = json.loads((target / "runtime/manifest.json").read_text(encoding="utf-8"))
    assert manifest["nllb_included"] is False
    assert manifest["streaming_model_included"] is False
    assert manifest["punctuation_model_included"] is False
    assert not (target / "data").exists()
    for record in manifest["files"]:
        output = target / record["path"]
        assert record["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
        assert record["bytes"] == output.stat().st_size
    names = {p.name for p in target.rglob("*") if p.is_file()}
    assert not names & {"auth.json", "knowledge.json", "my-paper.pdf", "transcript.txt"}
    assert not (target / "runtime/models/translation-nllb").exists()
    assert not (target / "runtime/models/streaming-asr").exists()
    assert not (target / "runtime/models/punctuation").exists()


@pytest.mark.parametrize("streaming,punctuation", [(True, False), (False, True), (True, True)])
def test_optional_caption_models_export_only_selected_public_files(export_fixture, streaming, punctuation):
    models, cuda, target = export_fixture
    flags = []
    if streaming:
        flags.append("-IncludeStreamingModel")
    if punctuation:
        flags.append("-IncludePunctuationModel")
    result = run_export(models, cuda, target, *flags)
    assert result.returncode == 0, result.stdout + result.stderr
    manifest = json.loads((target / "runtime/manifest.json").read_text(encoding="utf-8"))
    assert manifest["streaming_model_included"] is streaming
    assert manifest["punctuation_model_included"] is punctuation
    streaming_target = target / "runtime/models/streaming-asr/fastconformer-ctc-en-480ms"
    punctuation_target = target / "runtime/models/punctuation/edge-punct-en"
    assert streaming_target.is_dir() is streaming
    assert punctuation_target.is_dir() is punctuation
    if streaming:
        assert (streaming_target / "model.int8.onnx").is_file()
        assert (streaming_target / "tokens.txt").is_file()
    if punctuation:
        assert (punctuation_target / "model.int8.onnx").is_file()
        assert (punctuation_target / "bpe.vocab").is_file()
        assert (punctuation_target / "LICENSE").is_file()
    for record in manifest["files"]:
        output = target / record["path"]
        assert hashlib.sha256(output.read_bytes()).hexdigest() == record["sha256"]
    names = {file.name for file in target.rglob("*") if file.is_file()}
    assert not names & {"auth.json", "notes.txt"}


def test_optional_streaming_model_rejects_unknown_manifest_before_copy(export_fixture):
    models, cuda, target = export_fixture
    manifest = models / "streaming-asr/fastconformer-ctc-en-480ms/model.json"
    metadata = json.loads(manifest.read_text())
    metadata["revision"] = "unknown-unreviewed-model"
    manifest.write_text(json.dumps(metadata))
    result = run_export(models, cuda, target, "-IncludeStreamingModel")
    assert result.returncode != 0
    assert not target.exists()


def test_optional_punctuation_model_requires_license_before_copy(export_fixture):
    models, cuda, target = export_fixture
    (models / "punctuation/edge-punct-en/LICENSE").unlink()
    result = run_export(models, cuda, target, "-IncludePunctuationModel")
    assert result.returncode != 0
    assert not target.exists()


def test_whatif_never_creates_destination(export_fixture):
    models, cuda, target = export_fixture
    result = run_export(models, cuda, target, "-WhatIf")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not target.exists()


def test_nested_destination_is_rejected(export_fixture):
    models, cuda, _ = export_fixture
    target = models / "accidentally-nested"
    result = run_export(models, cuda, target)
    assert result.returncode != 0
    assert not target.exists()


def test_existing_runtime_needs_explicit_replacement(export_fixture):
    models, cuda, target = export_fixture
    target.mkdir()
    config = target / "runtime.json"
    config.write_text('{"keep":"my config"}', encoding="utf-8")
    result = run_export(models, cuda, target)
    assert result.returncode != 0
    assert config.read_text(encoding="utf-8") == '{"keep":"my config"}'
    assert not (target / "runtime").exists()


def test_different_existing_weight_is_never_overwritten(export_fixture):
    models, cuda, target = export_fixture
    assert run_export(models, cuda, target).returncode == 0
    damaged = target / "runtime/models/asr/large-v3-turbo/model.bin"
    damaged.write_bytes(b"existing different contents")
    result = run_export(models, cuda, target, "-ReplaceRuntimeConfig")
    assert result.returncode != 0
    assert damaged.read_bytes() == b"existing different contents"

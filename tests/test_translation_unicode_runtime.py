"""Opt-in real Argos regression for portable Windows Unicode paths.

Set PRISM_TEST_REAL_MODEL_ROOT to an already installed model directory. This
test copies only the existing translation model; it never downloads anything.
"""
import os
from pathlib import Path
import re
import shutil

import pytest

from zoom_copilot.translation import ArgosTranslator


@pytest.fixture
def actual_argos_unicode_dir(tmp_path):
    root = os.environ.get("PRISM_TEST_REAL_MODEL_ROOT")
    if not root:
        pytest.skip("Set PRISM_TEST_REAL_MODEL_ROOT for offline real-model portability regression")
    original = Path(root) / "translation"
    if not original.is_dir():
        pytest.fail("Configured real translation model directory does not exist")
    destination = tmp_path / "迁移 译文 中文路径" / "translation"
    shutil.copytree(original, destination)
    return destination


def test_actual_argos_translates_from_chinese_and_space_path(actual_argos_unicode_dir):
    translator = ArgosTranslator(actual_argos_unicode_dir)
    assert translator.ready
    result = translator.translate("The system uses 12 images to reconstruct the scene.")
    assert re.search(r"[\u4e00-\u9fff]", result)
    assert "12" in result
    assert not translator.last_error
    assert translator.last_metrics["sentences"] == 1


def test_actual_argos_can_retry_after_invalid_tokenizer_is_repaired(actual_argos_unicode_dir):
    translator = ArgosTranslator(actual_argos_unicode_dir)
    model_file = translator._model_path / "sentencepiece.model"
    serialized = model_file.read_bytes()
    model_file.write_bytes(b"deliberately invalid sentencepiece model for failure recovery test")
    with pytest.raises(RuntimeError, match="本地翻译失败"):
        translator.translate("The camera captures an image.")
    assert translator._translator is None and translator._tokenizer is None
    model_file.write_bytes(serialized)
    result = translator.translate("The camera captures an image.")
    assert re.search(r"[\u4e00-\u9fff]", result)
    assert not translator.last_error

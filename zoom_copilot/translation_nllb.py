"""Optional offline EN->ZH translation using OpenNMT's NLLB 1.3B INT8.

Runtime needs CTranslate2 and tokenizers, not torch/transformers. Model files are
downloaded only by an explicit install call from a pinned OpenNMT repository.
Meta NLLB weights are CC-BY-NC-4.0: personal non-commercial research evaluation,
not a license to sell/distribute this model in a commercial product. The model
card describes a research model trained on general-domain, sentence-level text;
larger size does not establish specialist translation accuracy.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
from urllib.request import Request, urlopen


MODEL_ID = "OpenNMT/nllb-200-distilled-1.3B-ct2-int8"
REVISION = "70f572adafa4794890ce7826156a4209717855af"
SOURCE_LANGUAGE, TARGET_LANGUAGE = "eng_Latn", "zho_Hans"
MODEL_LICENSE = "CC-BY-NC-4.0"
MODEL_PAGE = f"https://huggingface.co/{MODEL_ID}/tree/{REVISION}"
UPSTREAM_PAGE = "https://huggingface.co/facebook/nllb-200-distilled-1.3B"
FILES = {
    "model.bin": {"size": 1377755738, "sha256": "89dc4b9eb7f4dcd3ff36023c9b1c55fcecb47ddc0df97ab31b068ef9cf194940"},
    "tokenizer.json": {"size": 17331176, "sha256": "e316b82de11d0f951f370943b3c438311629547285129b0b81dadabd01bca665"},
    "shared_vocabulary.json": {"size": 5921176},
    "config.json": {"size": 1071},
    "tokenizer_config.json": {"size": 564},
    "special_tokens_map.json": {"size": 3548},
    "README.md": {"size": 8872},
}
MANIFEST = "nllb-install.json"
LICENSE_NOTICE = (
    "Optional translation model: Meta NLLB-200 distilled 1.3B, converted by OpenNMT.\n"
    f"Pinned source: {MODEL_PAGE}\nOriginal model: {UPSTREAM_PAGE}\n"
    "Model license: CC-BY-NC-4.0 (attribution; non-commercial use).\n"
    "License text: https://creativecommons.org/licenses/by-nc/4.0/\n"
    "Intended here for personal, non-commercial research evaluation.\n"
    "Commercial-looking application UI does not change the model's non-commercial license.\n"
    "This general-domain research model is not a certified specialist translator.\n"
    "CPU INT8 inference uses CTranslate2 + tokenizers; no torch or paid API is required.\n"
)


class NllbTranslator:
    def __init__(self, model_dir: str | Path, *, intra_threads: int = 4, beam_size: int = 3):
        self.model_dir = Path(model_dir)
        self.intra_threads = min(16, max(1, int(intra_threads)))
        self.beam_size = min(5, max(1, int(beam_size)))
        self._translator = None
        self._tokenizer = None
        self._lock = threading.RLock()
        self._install_lock = threading.Lock()
        self.last_error = ""
        self.last_metrics: dict = {}

    @property
    def ready(self) -> bool:
        try:
            manifest = json.loads((self.model_dir / MANIFEST).read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                return False
            return (manifest.get("model_id") == MODEL_ID and manifest.get("revision") == REVISION
                    and all((self.model_dir / name).is_file() and (self.model_dir / name).stat().st_size == meta["size"]
                            for name, meta in FILES.items()))
        except (OSError, ValueError, TypeError):
            return False

    def refresh(self) -> bool:
        return self.ready

    @staticmethod
    def _sha256(path: Path) -> str:
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    def _valid(self, path: Path, meta: dict) -> bool:
        if not path.is_file() or path.stat().st_size != meta["size"]:
            return False
        return not meta.get("sha256") or self._sha256(path) == meta["sha256"]

    def _download(self, name: str, meta: dict, status) -> dict:
        final = self.model_dir / name
        if self._valid(final, meta):
            status(f"已校验并复用 NLLB 文件：{name}")
            return {"size": meta["size"], "sha256": self._sha256(final)}
        partial = self.model_dir / (name + ".download")
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}?download=true"
        for attempt in range(3):
            try:
                offset = partial.stat().st_size if partial.is_file() else 0
                if offset >= meta["size"]:
                    if self._valid(partial, meta):
                        os.replace(partial, final)
                        return {"size": meta["size"], "sha256": self._sha256(final)}
                    partial.unlink()
                    offset = 0
                headers = {"User-Agent": "PrismInterview-local-model-installer/0.4"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                status(f"下载 NLLB {name}：{offset / 1024**2:.1f} / {meta['size'] / 1024**2:.1f} MiB")
                request = Request(url, headers=headers)
                with urlopen(request, timeout=30) as response:
                    if offset and response.status != 206:
                        offset = 0  # Server ignored Range: restart instead of appending a whole file.
                    total, last_status = offset, time.monotonic()
                    with partial.open("ab" if offset else "wb") as dest:
                        while chunk := response.read(2 * 1024 * 1024):
                            total += len(chunk)
                            if total > meta["size"]:
                                raise RuntimeError(f"{name} 超出固定版本的预期大小，已中止")
                            dest.write(chunk)
                            if time.monotonic() - last_status >= 4:
                                status(f"下载 NLLB {name}：{100 * total / meta['size']:.1f}% ({total / 1024**2:.1f} MiB)")
                                last_status = time.monotonic()
                        dest.flush()
                        os.fsync(dest.fileno())
                if not self._valid(partial, meta):
                    raise RuntimeError(f"{name} 大小或 SHA256 校验失败，未安装")
                os.replace(partial, final)
                return {"size": meta["size"], "sha256": self._sha256(final)}
            except Exception as exc:
                if attempt == 2:
                    raise RuntimeError(f"下载 {name} 失败：{exc}") from exc
                status(f"下载重试 {attempt + 1}/2：{name}（{exc}）")
        raise RuntimeError("下载未完成")

    def install(self, on_status=None) -> str:
        status = on_status or (lambda message: None)
        with self._install_lock:
            self.model_dir.mkdir(parents=True, exist_ok=True)
            status("准备可选 NLLB 实验模式：约 1.40 GB；CC-BY-NC-4.0，仅限非商业用途。")
            hashes = {name: self._download(name, meta, status) for name, meta in FILES.items()}
            manifest = {"version": 1, "model_id": MODEL_ID, "revision": REVISION,
                        "license": MODEL_LICENSE, "source_language": SOURCE_LANGUAGE,
                        "target_language": TARGET_LANGUAGE, "files": hashes}
            partial = self.model_dir / (MANIFEST + ".tmp")
            partial.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            os.replace(partial, self.model_dir / MANIFEST)
            (self.model_dir / "MODEL_LICENSE_NOTICE.txt").write_text(LICENSE_NOTICE, encoding="utf-8")
            with self._lock:
                self._translator, self._tokenizer = None, None
                self.last_error = ""
            if not self.ready:
                raise RuntimeError("NLLB 下载后完整性检查未通过")
            status("NLLB 1.3B INT8 已安装并校验；首次翻译需要载入 CPU 内存。")
            return "可选 NLLB 离线英译中模型已安装。"

    def install_english_chinese(self, on_status=None) -> str:
        return self.install(on_status)

    def _load(self) -> None:
        if self._translator is not None:
            return
        if not self.ready:
            raise RuntimeError("可选 NLLB 模型尚未安装，请先准备模型或切回 Argos 快速模式。")
        import ctranslate2
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        if any(tokenizer.token_to_id(t) is None for t in (SOURCE_LANGUAGE, TARGET_LANGUAGE, "</s>")):
            raise RuntimeError("NLLB tokenizer 缺少英/简中语言标记或 EOS")
        translator = ctranslate2.Translator(str(self.model_dir), device="cpu", compute_type="int8",
                                             inter_threads=1, intra_threads=self.intra_threads)
        self._tokenizer, self._translator = tokenizer, translator

    def _source_tokens(self, text: str) -> list[str]:
        # Explicit modern NLLB source template avoids a legacy tokenizer.json
        # adding its own language suffix. The raw subword IDs are unchanged.
        tokens = self._tokenizer.encode(text, add_special_tokens=False).tokens
        if len(tokens) > 500:
            raise ValueError("单句超过 NLLB 训练长度范围；请分成较短句子，未静默截断。")
        return [SOURCE_LANGUAGE, *tokens, "</s>"]

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        with self._lock:
            started = time.perf_counter()
            was_loaded = self._translator is not None
            try:
                self._load()
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
                sources = [self._source_tokens(s) for s in sentences]
                results = self._translator.translate_batch(
                    sources, target_prefix=[[TARGET_LANGUAGE] for _ in sources], beam_size=self.beam_size,
                    max_decoding_length=512, max_batch_size=8, batch_type="examples",
                    repetition_penalty=1.0, no_repeat_ngram_size=0)
                output = []
                for result in results:
                    tokens = result.hypotheses[0]
                    if len(tokens) >= 512:
                        raise RuntimeError("NLLB 输出达到最大长度，可能不完整；请缩短输入后重试。")
                    if tokens and tokens[0] == TARGET_LANGUAGE:
                        tokens = tokens[1:]
                    identities = [self._tokenizer.token_to_id(t) for t in tokens
                                  if t not in (SOURCE_LANGUAGE, TARGET_LANGUAGE, "</s>", "<s>", "<pad>")]
                    if any(i is None for i in identities):
                        raise RuntimeError("NLLB 输出包含 tokenizer 词表外的未知标记")
                    output.append(self._tokenizer.decode(identities, skip_special_tokens=True).strip())
                translated = " ".join(output).strip()
                if not translated:
                    raise RuntimeError("NLLB 未返回译文")
                self.last_error = ""
                self.last_metrics = {"translation_seconds": round(time.perf_counter() - started, 4),
                                     "cold_load_included": not was_loaded, "sentences": len(sentences),
                                     "source_tokens": sum(len(s) for s in sources), "device": "cpu",
                                     "compute_type": "int8", "beam_size": self.beam_size}
                return translated
            except Exception as exc:
                self.last_error = str(exc)
                raise RuntimeError("NLLB 本地翻译失败：" + str(exc)) from exc

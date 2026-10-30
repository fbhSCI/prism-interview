"""Local incremental English draft captions; Whisper remains the final decoder.

FastConformer (or optional Zipformer) consumes each 100-ms block once with
persistent encoder state. Drafts are replaceable hypotheses and must never be
used as a confirmed question.
"""
from __future__ import annotations

from collections import deque
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.request

import numpy as np

RATE = 16000
MODEL_REPOSITORY = "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26"
MODEL_REVISION = "672fbf1b30579d6585301139bb363f42a0ad4a24"
MODEL_SUBDIR = "zipformer-en-2023-06-26"
MODEL_FILES = {
    "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx":
        "563fde436d16cf7607cf408cd6b30909819d03162652ef389c2450ced3f45ac1",
    "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx":
        "98da299f471e38bb4e1a8df579b8cc9122d6039576a77e357b3c60f17dd83b02",
    "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx":
        "d944208d660d67c8d72cd2acaeac971fa5ceb8c80e76c1968148846fedd6e297",
}
# Small ordinary Git files are checked against the official repository Git IDs.
MODEL_GIT_FILES = {"tokens.txt": "b4d1bf82e68de1354f2a3f5fd51094d61d65c77c",
                   "README.md": "cb864962f371aa02e5716610cf4afc480b166a16"}
FAST_REPOSITORY = "csukuangfj/sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-480ms-int8"
FAST_REVISION = "66e9b61a9fe334eba579700599e391c6ee093633"
FAST_FILES = {"model.int8.onnx": "927a40a30e683e1dce4137f0e8ccae96e728ae8475f4f9798ce2aca8714272ad"}
FAST_GIT_FILES = {"tokens.txt": "3228fbf04d889e59138c56a1b19fd5fe795c3dbf"}


def model_directory(model_root, backend="fastconformer"):
    name = "fastconformer-ctc-en-480ms" if backend == "fastconformer" else MODEL_SUBDIR
    return Path(model_root).expanduser().resolve() / "streaming-asr" / name


def _valid(path, digest, git=False):
    if not path.is_file():
        return False
    value = hashlib.sha1() if git else hashlib.sha256()
    if git:
        value.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest() == digest


def prepare_streaming_model(model_root, on_status=lambda text: None, backend="fastconformer"):
    """Download pinned public weights; no user audio leaves the PC.

    FastConformer is derived from NVIDIA's CC-BY-4.0 English streaming model;
    the alternative Zipformer is Apache-2.0. Both are CPU-only draft decoders.
    """
    if backend not in ("fastconformer", "zipformer"):
        raise ValueError("Unsupported streaming ASR backend: " + str(backend))
    fast = backend == "fastconformer"
    repository, revision = (FAST_REPOSITORY, FAST_REVISION) if fast else (MODEL_REPOSITORY, MODEL_REVISION)
    files, git_files = (FAST_FILES, FAST_GIT_FILES) if fast else (MODEL_FILES, MODEL_GIT_FILES)
    destination = model_directory(model_root, backend)
    destination.mkdir(parents=True, exist_ok=True)
    for name, digest in {**files, **git_files}.items():
        target = destination / name
        git = name in git_files
        if _valid(target, digest, git):
            continue
        on_status("下载本地实时英文草稿模型：" + name)
        temporary = target.with_suffix(target.suffix + ".part")
        url = f"https://huggingface.co/{repository}/resolve/{revision}/{name}?download=true"
        last_error = None
        for attempt in range(3):
            try:
                offset = temporary.stat().st_size if temporary.is_file() else 0
                request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
                with urllib.request.urlopen(request, timeout=30) as response:
                    mode = "ab" if offset and response.status == 206 else "wb"
                    with temporary.open(mode) as output:
                        for block in iter(lambda: response.read(1024 * 1024), b""):
                            output.write(block)
                if _valid(temporary, digest, git):
                    break
                temporary.unlink(missing_ok=True)
                raise RuntimeError("模型下载内容未通过校验")
            except Exception as exc:
                last_error = exc
                if attempt == 2:
                    raise RuntimeError("实时英文模型下载失败：" + name) from last_error
        if not _valid(temporary, digest, git):
            temporary.unlink(missing_ok=True)
            raise RuntimeError("实时英文模型校验失败：" + name)
        temporary.replace(target)
    manifest = {"repository": repository, "revision": revision,
                "license": "CC-BY-4.0" if fast else "Apache-2.0",
                "upstream": "https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi" if fast
                    else "https://huggingface.co/" + MODEL_REPOSITORY,
                "sha256": files, "git_blobs": git_files}
    (destination / "model.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return destination


def create_recognizer(directory):
    import sherpa_onnx
    path = Path(directory)
    if (path / "model.int8.onnx").is_file():
        return sherpa_onnx.OnlineRecognizer.from_nemo_ctc(
            tokens=str(path / "tokens.txt"), model=str(path / "model.int8.onnx"),
            provider="cpu", num_threads=2, sample_rate=RATE,
            enable_endpoint_detection=False, decoding_method="greedy_search")
    return sherpa_onnx.OnlineRecognizer.from_transducer(
        tokens=str(path / "tokens.txt"),
        encoder=str(path / "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
        decoder=str(path / "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
        joiner=str(path / "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx"),
        provider="cpu", num_threads=2, model_type="zipformer2", sample_rate=RATE,
        enable_endpoint_detection=False, decoding_method="greedy_search")


class StreamingDraftWorker:
    """Independent CPU decoder so CUDA finalization cannot stall word updates.

    Source-clock metadata permits an accurate Whisper final to retire its draft
    prefix, while retaining words spoken after that final's commit boundary.
    """
    def __init__(self, model_root, on_text, on_status=lambda text: None,
                 on_metrics=lambda metrics: None, recognizer_factory=None, backend="fastconformer"):
        self.model_root = model_root
        self.on_text, self.on_status, self.on_metrics = on_text, on_status, on_metrics
        self.recognizer_factory = recognizer_factory
        self.backend = backend
        # sherpa-onnx 1.13.8 ConvertCtc uses 4*10ms for CTC timestamps;
        # this pinned NVIDIA model declares subsampling_factor=8. Correct the
        # source clock before merging with Whisper's true 16-kHz timestamps.
        self.timestamp_scale = 2.0 if backend == "fastconformer" and recognizer_factory is None else 1.0
        self.ready_event = threading.Event()
        self.failed_event = threading.Event()
        self.finished = threading.Event()
        self.failure = ""
        self.thread = None
        self._condition = threading.Condition()
        self._blocks = deque()
        self._queued_samples = 0
        self._committed = {}
        self._refinements = {}
        self._commit_generation = {}
        self._drain = True

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.finished.clear()
        self.thread = threading.Thread(target=self._run, name="PrismStreamingASR", daemon=True)
        self.thread.start()

    def submit(self, audio, metadata):
        if self.finished.is_set() or self.failed_event.is_set():
            return False
        with self._condition:
            # Keep final Whisper transcription; don't display delayed drafts if
            # a machine cannot sustain the streaming model's real-time pace.
            if self._queued_samples + len(audio) > 5 * RATE:
                self.failure = "实时英文草稿积压超过5秒，已停用草稿；Whisper最终识别继续。"
                self.failed_event.set()
                self._blocks.clear()
                self._queued_samples = 0
                self._condition.notify_all()
                self.on_status(self.failure)
                return False
            self._blocks.append((np.asarray(audio, dtype=np.float32).reshape(-1).copy(), dict(metadata), time.perf_counter()))
            self._queued_samples += len(audio)
            self._condition.notify_all()
        return True

    def commit(self, metadata):
        key = (metadata.get("stream_id"), metadata.get("capture_context"))
        boundary = metadata.get("commit_end_sample")
        if boundary is not None:
            with self._condition:
                self._committed[key] = max(int(boundary), self._committed.get(key, -1))
                self._commit_generation[key] = self._commit_generation.get(key, 0) + 1

    def refine(self, metadata, words):
        """Replace only the acoustically covered draft prefix, keep newer tail.

        Untimed Whisper output cannot safely splice into a rolling hypothesis.
        It is deliberately left to the final-caption path in that case.
        """
        if not words:
            return
        key = (metadata.get("stream_id"), metadata.get("capture_context"))
        end = metadata.get("commit_end_sample")
        if end is None:
            return
        with self._condition:
            previous = self._refinements.get(key)
            if end > self._committed.get(key, -1) and (not previous or end >= previous[0]):
                self._refinements[key] = (int(end), [dict(word) for word in words])

    def _hypothesis(self, result, origin, key):
        with self._condition:
            lower = max(origin, self._committed.get(key, origin))
            refinement = self._refinements.get(key)
        if refinement and refinement[0] > lower:
            boundary, words = refinement
            prefix = "".join(word["text"] for word in words
                             if (word["start"] + word["end"]) / 2 >= lower).strip()
            prefix_words = [dict(word) for word in words if (word["start"] + word["end"]) / 2 >= lower]
            tail_words = self._words_after(result, origin, boundary, self.timestamp_scale)
            tail = self._text_after(result, origin, boundary, self.timestamp_scale)
            return " ".join(part for part in (prefix, tail) if part), lower, "whisper_revision", prefix_words + tail_words
        return (self._text_after(result, origin, lower, self.timestamp_scale), lower, self.backend,
                self._words_after(result, origin, lower, self.timestamp_scale))

    def stop(self, drain=True):
        self.finished.set()
        with self._condition:
            self._drain = bool(drain)
            if not drain:
                self._blocks.clear()
                self._queued_samples = 0
            self._condition.notify_all()

    @staticmethod
    def _words_after(result, origin, lower, timestamp_scale=1.0):
        tokens = list(getattr(result, "tokens", []) or [])
        timestamps = list(getattr(result, "timestamps", []) or [])
        if len(tokens) != len(timestamps) or not tokens:
            return []
        selected, word, word_start, word_end = [], [], origin, origin
        for token, stamp in zip(tokens, timestamps):
            if token.startswith((" ", "▁")) and word:
                if word_start >= lower:
                    selected.append({"text": "".join(word).replace("▁", " "), "start": word_start, "end": word_end})
                word = []
            if not word:
                word_start = origin + round(float(stamp) * timestamp_scale * RATE)
            word_end = origin + round((float(stamp) + .04) * timestamp_scale * RATE)
            word.append(token)
        if word and word_start >= lower:
            selected.append({"text": "".join(word).replace("▁", " "), "start": word_start, "end": word_end})
        return selected

    @staticmethod
    def _text_after(result, origin, lower, timestamp_scale=1.0):
        words = StreamingDraftWorker._words_after(result, origin, lower, timestamp_scale)
        if words:
            text = "".join(word["text"] for word in words).strip()
        elif not getattr(result, "tokens", None) and lower <= origin:
            text = str(getattr(result, "text", "")).strip()
        else:
            # Untimed prefixes cannot safely replace a newer confirmed caption.
            return ""
        # Published English Zipformer emits upper-case BPE tokens.
        return text[:1].upper() + text[1:].lower() if text.isupper() else text

    def _run(self):
        try:
            if self.recognizer_factory:
                recognizer = self.recognizer_factory()
            else:
                self.on_status("准备本地流式英文草稿模型（CPU）…")
                directory = prepare_streaming_model(self.model_root, self.on_status, self.backend)
                recognizer = create_recognizer(directory)
                warm = recognizer.create_stream()
                warm.accept_waveform(RATE, np.zeros(RATE, dtype=np.float32))
                while recognizer.is_ready(warm):
                    recognizer.decode_stream(warm)
            self.ready_event.set()
            self.on_status("实时英文草稿已就绪 · 本地 " + self.backend + " CPU；Whisper持续校正")
        except Exception as exc:
            self.failure = str(exc)
            self.failed_event.set()
            self.on_status("实时英文草稿未启用；Whisper最终识别继续：" + str(exc))
            return
        stream = None
        identity = None
        origin = 0
        latest = ""
        until = 0
        lookback = np.empty(0, dtype=np.float32)
        last_voice = None
        latest_generation = -1
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._blocks or self.finished.is_set() or self.failed_event.is_set(), .3)
                if self.failed_event.is_set() or (self.finished.is_set() and (not self._drain or not self._blocks)):
                    break
                if not self._blocks:
                    continue
                audio, metadata, submitted = self._blocks.popleft()
                self._queued_samples -= len(audio)
            key = (metadata.get("stream_id"), metadata.get("capture_context"))
            start = int(metadata.get("start_sample", until))
            speech_active = metadata.get("speech_active")
            new_voice_after_silence = bool(speech_active and last_voice is not None and start - last_voice > 2.4 * RATE)
            # Phase/device boundaries create fresh independent online state.
            if stream is None or key != identity or start != until:
                stream = recognizer.create_stream()
                identity, origin, latest = key, start, ""
                lookback = np.empty(0, dtype=np.float32)
                last_voice = None
                latest_generation = -1
            elif start - origin >= 60 * RATE or new_voice_after_silence:
                # Bound decoder history during multi-hour calls. Replaying
                # only 1.6s of acoustic context does not change final commits.
                stream = recognizer.create_stream()
                origin = start - lookback.size
                stream.accept_waveform(RATE, lookback)
                latest = ""
            if speech_active:
                last_voice = start + len(audio)
            until = start + len(audio)
            started = time.perf_counter()
            try:
                stream.accept_waveform(RATE, audio)
                lookback = np.concatenate((lookback, audio))[-int(1.6 * RATE):]
                while recognizer.is_ready(stream):
                    recognizer.decode_stream(stream)
                result = recognizer.recognizer.get_result(stream)
                with self._condition:
                    text, lower, backend, draft_words = self._hypothesis(result, origin, key)
                    generation = self._commit_generation.get(key, 0)
                display_allowed = speech_active is None or (last_voice is not None and until - last_voice <= 1.1 * RATE)
                force_commit = generation != latest_generation and latest_generation >= 0
                if force_commit and not display_allowed:
                    # A delayed final often arrives after VAD trailing silence.
                    # Retire its old draft even then; don't expose silence noise.
                    text, draft_words = "", []
                changed = text != latest or generation != latest_generation
                if (changed and (text or latest_generation >= 0) and (display_allowed or force_commit)
                        and not self.finished.is_set()):
                    latest = text
                    latest_generation = generation
                    context = {"phase_epoch": metadata.get("capture_context"),
                               "draft_backend": backend, "word_timed": True,
                               "stream_id": metadata.get("stream_id"),
                               "start_sample": lower, "end_sample": until,
                               "text_end_sample": max((word["end"] for word in draft_words), default=lower),
                               "draft_words": draft_words, "commit_generation": generation}
                    self.on_text("partial", text, context)
                    self.on_metrics({"draft_backend": self.backend + "_cpu", "draft_audio_end_seconds": round(until / RATE, 3),
                                     "draft_queue_seconds": round(started - submitted, 4),
                                     "draft_decode_seconds": round(time.perf_counter() - started, 4)})
            except Exception as exc:
                self.failure = str(exc)
                self.failed_event.set()
                self.on_status("实时英文草稿停止；Whisper最终识别继续：" + str(exc))
                break

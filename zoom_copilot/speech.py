"""Local ASR with explicit readiness, final-segment priority and GPU support."""
from __future__ import annotations
from collections import deque
from pathlib import Path
import math
import re
import threading
import time
from typing import Callable
import numpy as np
from .runtime import configure_cuda_dlls

# Kept as a backwards-compatible explicit caller hook; supply only your own terms.
RESEARCH_GLOSSARY = ""
# A vocabulary prompt is a language-model prior, not an acoustic constraint.
# In ordinary speech/music it previously induced unrelated research captions.
# Scientific hints remain an explicit caller opt-in only.
DEFAULT_GLOSSARY = ""


def _segment_separator(previous: str, incoming: str) -> str:
    """Restore only a missing separator between two decoded ASR segments.

    Word objects within one segment can be pieces of a contraction, decimal,
    abbreviation or hyphenated term. Their spelling and internal spacing must
    remain untouched. Segment starts, however, need not carry Whisper's usual
    leading space (``life.`` + ``Scheduling`` must not become one token).
    Existing spaces and punctuation-led continuations remain as supplied.
    """
    if not previous or not incoming or previous[-1].isspace() or incoming[0].isspace():
        return ""
    if not incoming[0].isalnum() or previous[-1] in "([{/'’\"“‘-=+":
        return ""
    # A decimal or an unspaced dotted abbreviation can itself cross segments.
    # Do not turn 0. + 95 into two sentences or e. + g. into separate initials.
    if (re.search(r"\d\.$", previous) and incoming[0].isdigit()
            or re.search(r"(?<!\w)(?:[A-Za-z]\.)+$", previous)
            and re.match(r"[A-Za-z]\.(?=\W|$)", incoming)):
        return ""
    return " "


class SpeechWorker:
    def __init__(self, model_size: str, translator=None,
                 on_text: Callable[[str, str, str | dict], None] | None = None,
                 on_status: Callable[[str], None] | None = None, *,
                 device: str = "auto", compute_type: str | None = None,
                 download_root: str | Path | None = None,
                 on_ready: Callable[[], None] | None = None,
                 on_metrics: Callable[[dict], None] | None = None,
                 glossary: str = DEFAULT_GLOSSARY, model_factory=None, diarizer=None,
                 streaming_model_root=None, streaming_factory=None, streaming_backend="fastconformer"):
        self.model_size = model_size
        self.on_text = on_text or (lambda *args: None)
        self.on_status = on_status or (lambda *args: None)
        self.on_ready = on_ready or (lambda: None)
        self.on_metrics = on_metrics or (lambda metrics: None)
        self.device, self.compute_type = device, compute_type
        self.download_root = str(download_root) if download_root else None
        self.glossary = glossary
        self.model_factory = model_factory
        self.diarizer = diarizer
        self.streaming_model_root = streaming_model_root
        self.streaming_factory = streaming_factory
        self.streaming_backend = streaming_backend
        self.streaming = None
        self._decoded_words = []
        self._acoustic_words = []
        self._speaker_failed = False
        self.finished = threading.Event()
        self.ready_event = threading.Event()
        self.failed_event = threading.Event()
        self.failure = ""
        self.thread: threading.Thread | None = None
        self._condition = threading.Condition()
        self._finals = deque()
        self._partials = {}
        self._working = False
        self._working_source = None
        self._queued_samples = 0
        self._drain = True
        self._revision = 0
        self._source_revisions = {}
        self._committed_until: dict[str, int] = {}
        self.actual_device = ""

    @property
    def pending(self) -> int:
        with self._condition:
            return len(self._finals) + len(self._partials) + int(self._working)

    @staticmethod
    def _source(metadata):
        return metadata.get("input_source") or "loopback"

    def pending_for_source(self, source: str) -> int:
        """Own microphone work must not hold a remote question's endpoint open."""
        with self._condition:
            return (sum(self._source(item[2]) == source for item in self._finals)
                    + int(source in self._partials)
                    + int(self._working and self._working_source == source))

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.finished.clear()
        self.ready_event.clear()
        self.failed_event.clear()
        self.thread = threading.Thread(target=self._run, name="PrismASR", daemon=True)
        self.thread.start()

    def submit(self, kind: str, audio: np.ndarray) -> bool:
        if self.finished.is_set() or self.failed_event.is_set():
            self.on_status("识别器已停止接收；本段音频未入队，请先完成采集尾段再停止识别。")
            return False
        metadata = {key: getattr(audio, key) for key in
                    ("start_sample", "commit_start_sample", "commit_end_sample", "stream_id", "hard_cut", "capture_context", "speech_active", "input_source")
                    if getattr(audio, key, None) is not None}
        if kind == "stream":
            return bool(self.streaming and self.streaming.submit(audio, metadata))
        audio = np.asarray(audio, dtype=np.float32).reshape(-1).copy()
        if not audio.size:
            return False
        with self._condition:
            source = self._source(metadata)
            if kind == "partial":
                if not any(self._source(item[2]) == source for item in self._finals):
                    self._partials[source] = (kind, audio, metadata,
                        self._source_revisions.get(source, 0), time.perf_counter())
            else:
                if self._queued_samples + audio.size > 120 * 16000:
                    self.failure = "识别积压超过120秒，已停止采集；请更换较快模型后重新开始。"
                    self.failed_event.set()
                    self.on_status("Whisper 处理失败：" + self.failure)
                    self._condition.notify_all()
                    return False
                self._partials.pop(source, None)
                self._revision += 1
                self._source_revisions[source] = self._source_revisions.get(source, 0) + 1
                self._finals.append(("final", audio, metadata,
                    self._source_revisions[source], time.perf_counter()))
                self._queued_samples += audio.size
            self._condition.notify_all()
        return True

    def stop(self, drain: bool = True):
        """Stop accepting; by default finish every already accepted final.

        Caller must stop/join AudioCapture first so its flush is submitted before
        this method. A pending hypothesis is not a committed final and is dropped.
        """
        self.finished.set()
        if self.streaming is not None:
            self.streaming.stop(drain=False)  # Drafts never own the final tail.
        with self._condition:
            self._drain = bool(drain)
            self._partials.clear()
            self._revision += 1
            if not drain and self._finals:
                self.on_status(f"已明确取消 {len(self._finals)} 段排队的最终识别。")
                self._finals.clear()
                self._queued_samples = 0
            self._condition.notify_all()

    @staticmethod
    def _rejection(part):
        logprob = getattr(part, "avg_logprob", None)
        no_speech = getattr(part, "no_speech_prob", None)
        compression = getattr(part, "compression_ratio", None)
        if logprob is not None and logprob < -1.0:
            return "low_log_probability"
        if no_speech is not None and no_speech >= .60 and (logprob is None or logprob < -.35):
            return "high_no_speech_probability"
        if compression is not None and compression > 2.7:
            return "repetitive_low_information"
        return ""

    def _decode_text(self, parts, kind, metadata):
        self._decoded_words = []
        self._acoustic_words = []
        rejected, accepted = [], []
        for part in parts:
            reason = self._rejection(part)
            if reason:
                rejected.append({"reason": reason,
                                 "avg_logprob": getattr(part, "avg_logprob", None),
                                 "no_speech_prob": getattr(part, "no_speech_prob", None)})
            else:
                accepted.append(part)
        bounds = ("stream_id", "start_sample", "commit_start_sample", "commit_end_sample")
        if not all(key in metadata for key in bounds):
            return " ".join(p.text.strip() for p in accepted).strip(), rejected, False
        stream = metadata["stream_id"]
        # Adjacent Whisper windows can timestamp the same boundary word on
        # opposite sides of a fixed cutoff. Continue from the last *emitted word
        # end*, not that mechanical cutoff, so an omitted boundary word can be
        # recovered in the next overlapping window without a lexical blacklist.
        lower = max(metadata["start_sample"], self._committed_until.get(stream, metadata["commit_start_sample"]))
        upper = metadata["commit_end_sample"]
        start = metadata["start_sample"]
        pieces = []
        last_word_end = None
        untimed = False
        for part in accepted:
            timed_words = getattr(part, "words", None)
            complete = bool(timed_words)
            if complete:
                try:
                    complete = all(
                        math.isfinite(float(word.start)) and math.isfinite(float(word.end))
                        and 0 <= float(word.start) <= float(word.end)
                        and isinstance(word.word, str) and bool(word.word.strip())
                        for word in timed_words)
                    # A partial alignment must not silently drop words that
                    # still exist in the segment text.
                    complete = complete and " ".join("".join(word.word for word in timed_words).split()) == " ".join(part.text.split())
                except (AttributeError, TypeError, ValueError):
                    complete = False
            if complete:
                part_emitted = False
                for word in timed_words:
                    middle = start + ((float(word.start) + float(word.end)) / 2) * 16000
                    acoustic_word = {"text": word.word,
                        "start": int(start + float(word.start) * 16000),
                        "end": int(start + float(word.end) * 16000)}
                    self._acoustic_words.append(acoustic_word)
                    if lower <= middle < upper:
                        separator = (_segment_separator("".join(pieces), word.word)
                                     if not part_emitted else "")
                        if separator:
                            # Keep the public text and timed word metadata in
                            # agreement for speaker grouping/streaming reuse.
                            # Do not mutate the model's original word object.
                            acoustic_word["text"] = separator + word.word
                        pieces.append(acoustic_word["text"])
                        self._decoded_words.append(acoustic_word)
                        part_emitted = True
                        last_word_end = max(last_word_end or 0, int(start + float(word.end) * 16000))
            else:
                # Model test doubles/exceptional alignments may omit words.
                # Preserve the text visibly rather than silently discarding it.
                pieces.append(" " + part.text.strip())
                untimed = True
                if metadata.get("hard_cut"):
                    self.on_status("本段缺少逐字时间，保留原始识别；硬切边界可能需要核对重复。")
        if kind == "final":
            committed = upper if untimed or not accepted else (last_word_end or self._committed_until.get(stream, lower))
            self._committed_until[stream] = max(committed, self._committed_until.get(stream, -1))
        if untimed:
            self._decoded_words = []
            self._acoustic_words = []
        return "".join(pieces).strip(), rejected, bool(accepted) and not untimed

    def _load(self):
        configure_cuda_dlls()
        if self.model_factory:
            factory = self.model_factory
            device = "cpu" if self.device == "auto" else self.device
        else:
            import ctranslate2
            from faster_whisper import WhisperModel
            factory = WhisperModel
            device = self.device
            if device == "auto":
                device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
        compute = self.compute_type or ("int8_float16" if device == "cuda" else "int8")
        if device == "cpu" and compute in ("float16", "int8_float16"):
            compute = "int8"
        kwargs = dict(device=device, compute_type=compute, cpu_threads=4)
        if self.download_root:
            kwargs["download_root"] = self.download_root
        try:
            model = factory(self.model_size, **kwargs)
            if not self.model_factory:
                warm_segments, _ = model.transcribe(np.zeros(16000,dtype=np.float32), language="en", beam_size=1)
                list(warm_segments)  # Force lazy CUDA DLL/inference failures before capture.
        except (RuntimeError, OSError) as exc:
            if self.device != "auto" or device != "cuda":
                raise
            self.on_status("GPU加载失败，明确降级到CPU；可在设置中检查CUDA依赖。" + str(exc)[:100])
            kwargs.update(device="cpu", compute_type="int8")
            model = factory(self.model_size, **kwargs)
            device = "cpu"
        self.actual_device = device
        return model

    def _run(self):
        try:
            self.on_status(f"加载本地 Whisper {self.model_size}；模型准备完成后才开始采集…")
            model = self._load()
            if self.streaming_model_root is not None:
                from .streaming_asr import StreamingDraftWorker
                self.streaming = StreamingDraftWorker(
                    self.streaming_model_root, self.on_text, self.on_status, self.on_metrics,
                    recognizer_factory=self.streaming_factory, backend=self.streaming_backend)
                self.streaming.start()
                while not self.streaming.ready_event.wait(.1):
                    if self.streaming.failed_event.is_set() or self.finished.is_set():
                        if self.finished.is_set():
                            self.streaming.stop(drain=False)
                        break
            if self.diarizer is not None:
                try:
                    if self.diarizer.prepare():
                        self.on_status("本地说话人分组已就绪；短句、混合或不确定声音会标为未确认。")
                    else:
                        self.on_status(self.diarizer.failure)
                except Exception as exc:
                    self._speaker_failed = True
                    self.on_status("说话人模型未就绪；继续转写并标为未确认。" + str(exc)[:100])
            if self.finished.is_set() and (not self._drain or not self._finals):
                return
            self.ready_event.set()
            self.on_status(f"Whisper 已就绪 · {self.actual_device.upper()}")
            if not self.finished.is_set():
                self.on_ready()
        except Exception as exc:
            self.failure = str(exc)
            self.failed_event.set()
            self.on_status(f"Whisper 加载失败：{exc}")
            return
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: bool(self._finals) or bool(self._partials) or self.finished.is_set() or self.failed_event.is_set(),
                    timeout=0.3)
                stopping = self.finished.is_set() or self.failed_event.is_set()
                if stopping and (not self._drain or not self._finals):
                    break
                if self._finals:
                    kind, audio, metadata, revision, submitted = self._finals.popleft()
                    self._queued_samples -= audio.size
                elif self._partials:
                    key = next(iter(self._partials))
                    kind, audio, metadata, revision, submitted = self._partials.pop(key)
                else:
                    continue
                self._working = True
                self._working_source = self._source(metadata)
            started = time.perf_counter()
            try:
                segments, _ = model.transcribe(
                    audio, language="en", task="transcribe", beam_size=1 if kind == "partial" else 3,
                    condition_on_previous_text=False, initial_prompt=self.glossary.strip() or None,
                    vad_filter=True, vad_parameters={"threshold": .5, "min_speech_duration_ms": 120,
                                                    "min_silence_duration_ms": 450, "speech_pad_ms": 180},
                    no_speech_threshold=.55, log_prob_threshold=-1.0,
                    compression_ratio_threshold=2.4, hallucination_silence_threshold=1.0,
                    temperature=0.0, word_timestamps=True)
                transcript, rejected, word_timed = self._decode_text(list(segments), kind, metadata)
                if kind == "final" and self.streaming is not None:
                    self.streaming.commit(metadata)
                asr_elapsed = time.perf_counter() - started
                context = {"phase_epoch": metadata.get("capture_context"), "word_timed": word_timed,
                           "stream_id": metadata.get("stream_id"), "start_sample": metadata.get("commit_start_sample"),
                           "end_sample": metadata.get("commit_end_sample"),
                           "hard_cut": bool(metadata.get("hard_cut")),
                           "words": list(self._decoded_words)} if metadata else ""
                if isinstance(context, dict) and metadata.get("input_source"):
                    context["input_source"] = metadata["input_source"]
                if kind == "final" and (rejected or isinstance(context, dict)):
                    # A rejected segment inside an otherwise accepted chunk
                    # must not silently bridge a missing question condition.
                    # Keep the accepted text visible and let the controller
                    # gate automatic answers, including on unclocked inputs.
                    if not isinstance(context, dict):
                        context = {}
                    context["recognition_gap"] = bool(rejected)
                stale = kind == "partial" and (revision != self._source_revisions.get(self._source(metadata), 0)
                                                or self.finished.is_set())
                if transcript and not stale:
                    if (kind == "partial" and self.streaming is not None
                            and self.streaming.ready_event.is_set() and not self.streaming.failed_event.is_set()):
                        # The CPU streaming thread splices a revised Whisper
                        # prefix into its newer live tail using source clocks.
                        self.streaming.refine(metadata, self._decoded_words)
                    elif (kind == "final" and metadata.get("input_source") != "microphone"
                          and self.diarizer is not None and not self._speaker_failed):
                        try:
                            # Word emission stays inside non-overlapping commit
                            # bounds. Speaker embeddings may also use decoded
                            # overlap acoustics, so a boundary word is not forced
                            # to identify a voice from a fraction of a second.
                            speaker_metadata = dict(metadata, acoustic_words=self._acoustic_words)
                            grouped = self.diarizer.annotate(audio, self._decoded_words, speaker_metadata, transcript)
                            grouped_text = " ".join(" ".join(group["text"].split()) for group in grouped)
                            if not grouped or grouped_text != " ".join(transcript.split()):
                                raise RuntimeError("说话人分组未完整保留已确认文字，已回退原始转写。")
                            for group in grouped:
                                speaker_context = dict(context) if isinstance(context, dict) else {}
                                speaker_context.update({key: value for key, value in group.items() if key != "text"})
                                self.on_text(kind, group["text"], speaker_context)
                        except Exception as exc:
                            # A speaker-labeling failure must never swallow the
                            # already accepted final words or stop ASR draining.
                            self._speaker_failed = True
                            self.on_status("说话人分组失败；继续保留转写文字。" + str(exc)[:100])
                            fallback = dict(context) if isinstance(context, dict) else {}
                            fallback.update(speaker_id=None, speaker_label="说话人未确认")
                            self.on_text(kind, transcript, fallback)
                    else:
                        if kind == "partial" and isinstance(context, dict) and self.streaming is not None:
                            context.update(draft_backend="whisper_fallback", draft_words=list(self._decoded_words),
                                           text_end_sample=max((word["end"] for word in self._decoded_words),
                                                               default=context.get("start_sample") or 0))
                        self.on_text(kind, transcript, context)
                elif kind == "final":
                    if word_timed and not rejected:
                        # A flush can decode only the already committed overlap.
                        # No new lexical words is not a recognition failure, and
                        # must not invalidate the complete question just emitted.
                        if isinstance(context,dict):
                            context['empty_commit']=True
                        self.on_text("final", "", context)
                    else:
                        # A genuine failed segment still blocks automatic
                        # answers rather than silently bridging missing speech.
                        self.on_text("rejected", "", context)
                self.on_metrics({"asr_seconds": round(asr_elapsed, 3),
                                 "asr_total_seconds": round(time.perf_counter()-started, 3),
                                 "audio_seconds": round(audio.size / 16000, 3),
                                 "asr_device": self.actual_device, "asr_backlog": self.pending,
                                 "asr_queue_seconds": round(started - submitted, 3),
                                 "asr_rejected_segments": rejected, "asr_stale_partial": stale,
                                 "asr_result_kind": "stale" if stale else kind if transcript else "empty_commit" if word_timed and not rejected else "rejected"})
                if kind == "final" and metadata.get("input_source") != "microphone" and self.diarizer is not None:
                    self.on_metrics(dict(self.diarizer.last_metrics))
            except Exception as exc:
                self.on_status(f"转写出错：{exc}")
            finally:
                with self._condition:
                    self._working = False
                    self._working_source = None
        if self.diarizer is not None and not self._speaker_failed:
            finalize = getattr(self.diarizer, "finalize_pending", None)
            if finalize is not None:
                try:
                    revisions = finalize()
                    if revisions:
                        self.on_text("speaker_revisions", "", {"speaker_revisions": revisions})
                except Exception as exc:
                    self.on_status("结束收音后的说话人复核失败，已保留全部转写文字。" + str(exc)[:100])

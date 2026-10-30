"""Selected Windows playback or microphone capture and speech segmentation.

Playback loopback can contain every app using that device. Microphone mode
captures only the explicitly selected physical recording device. No Zoom API,
credentials or meeting bot is involved.
"""
from __future__ import annotations

from collections import deque
import math
import threading
from typing import Callable
import uuid

import numpy as np


RATE = 16_000
CAPTURE_RATE = 48_000


class AudioChunk(np.ndarray):
    """An ndarray with source-clock commit bounds; public (kind, audio) API stays.

    A hard-cut chunk includes lookahead and subsequent chunks retain lookback.
    SpeechWorker commits only words inside the non-overlapping commit interval.
    """
    def __new__(cls, audio, *, start_sample, commit_start_sample, commit_end_sample,
                stream_id, hard_cut=False, capture_context=None, speech_active=None,
                input_source=None):
        result = np.asarray(audio, dtype=np.float32).view(cls)
        result.start_sample = int(start_sample)
        result.commit_start_sample = int(commit_start_sample)
        result.commit_end_sample = int(commit_end_sample)
        result.stream_id = stream_id
        result.hard_cut = bool(hard_cut)
        result.capture_context = capture_context
        result.speech_active = speech_active
        result.input_source = input_source
        return result

    def __array_finalize__(self, source):
        for name in ("start_sample", "commit_start_sample", "commit_end_sample", "stream_id", "hard_cut", "capture_context", "speech_active", "input_source"):
            setattr(self, name, getattr(source, name, None))


class StreamingVoiceDetector:
    """Stateful Silero ONNX, bundled with faster-whisper; no model download.

    Its recurrent state and 64-sample context continue across 100-ms blocks.
    Energy alone cannot distinguish music from speech. Loading failures are
    explicit; production never silently falls back to energy-only gating.
    """
    def __init__(self, threshold=.55, negative_threshold=.35):
        from faster_whisper.vad import get_vad_model
        self._session = get_vad_model().session
        self.threshold, self.negative_threshold = threshold, negative_threshold
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros_like(self._h)
        self._context = np.zeros(64, dtype=np.float32)
        self._pending = np.empty(0, dtype=np.float32)
        self.probability, self.voiced = 0.0, False

    def __call__(self, audio):
        self._pending = np.concatenate((self._pending, audio))
        probabilities = []
        while self._pending.size >= 512:
            frame, self._pending = self._pending[:512], self._pending[512:]
            inp = np.concatenate((self._context, frame))[None, :]
            out, self._h, self._c = self._session.run(None, {"input": inp, "h": self._h, "c": self._c})
            probabilities.append(float(np.asarray(out).reshape(-1)[0]))
            self._context = frame[-64:].copy()
        if probabilities:
            self.probability = max(probabilities)
            if self.probability >= self.threshold:
                self.voiced = True
            elif self.probability < self.negative_threshold:
                self.voiced = False
        return self.voiced


class Segmenter:
    """Real VAD + rolling context, with faster partials and bounded hard cuts.

    ``vad=False`` is an explicit test/diagnostic opt-out, not production fallback.
    Complete audio commit intervals tile a long utterance without discarding its
    final tail. Partial captions remain hypotheses and do not commit words.
    """

    def __init__(self, threshold: float = 0.005, silence_sec: float = 1.1,
                 maximum_sec: float = 12.0, *, partial_sec: float = .9,
                 vad=True, overlap_sec: float = 1.6, holdback_sec: float = .6,
                 commit_interval_sec: float | None = None,
                 initial_maximum_sec: float | None = None):
        self.threshold = threshold
        self.silence_samples = int(RATE * silence_sec)
        self.max_samples = int(RATE * maximum_sec)
        self.preview_samples = int(RATE * partial_sec)
        self.overlap_samples = min(int(RATE * overlap_sec), self.max_samples // 3)
        if commit_interval_sec is not None:
            interval = float(commit_interval_sec)
            if not math.isfinite(interval) or not 0 < interval < float(maximum_sec):
                raise ValueError("commit_interval_sec must be positive and shorter than maximum_sec")
            interval_samples = int(RATE * interval)
            if interval_samples <= 0 or interval_samples >= self.max_samples:
                raise ValueError("commit_interval_sec is outside the audio sample range")
            # Explicit cadence keeps the full acoustic context instead of
            # shortening the decoder window. Old callers retain the old clamp.
            self.overlap_samples = self.max_samples - interval_samples
        self.holdback_samples = min(int(RATE * holdback_sec), self.overlap_samples // 2)
        self._initial_max_samples = None
        if initial_maximum_sec is not None:
            initial = float(initial_maximum_sec)
            if not math.isfinite(initial) or not 0 < initial <= float(maximum_sec):
                raise ValueError("initial_maximum_sec must be positive and no longer than maximum_sec")
            initial_samples = int(RATE * initial)
            if initial_samples <= self.holdback_samples:
                raise ValueError("initial_maximum_sec must exceed the held audio tail")
            self._initial_max_samples = initial_samples
        # This budget belongs to the capture session, not to an utterance or
        # phase. reset() must never reinstate it after a pause or source flush.
        self._initial_window_pending = self._initial_max_samples is not None
        self._vad = StreamingVoiceDetector() if vad is True else (vad if callable(vad) else None)
        self.vad_backend = "silero-onnx" if vad is True else "injected" if callable(vad) else "energy-explicit-optout"
        self.is_speech = False
        self.stream_id = uuid.uuid4().hex
        self._cursor = 0
        self._active_start = 0
        self._commit_start = 0
        self.pre = deque(maxlen=6)  # ~600 ms at the recorder's 100-ms block size.
        self.active: list[np.ndarray] = []
        self.count = 0
        self.speech_samples = 0
        self.silent_count = 0
        self.last_preview_at = 0

    def reset(self):
        self.active = []
        self.count = 0
        self.speech_samples = 0
        self.silent_count = 0
        self.last_preview_at = 0

    def _chunk(self, complete, *, hard_cut=False):
        end = self._active_start + complete.size
        commit_end = end - self.holdback_samples if hard_cut else end
        return AudioChunk(complete, start_sample=self._active_start,
                          commit_start_sample=self._commit_start, commit_end_sample=commit_end,
                          stream_id=self.stream_id, hard_cut=hard_cut)

    def push(self, audio: np.ndarray) -> list[tuple[str, np.ndarray]]:
        if audio.size == 0:
            return []
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        block_start = self._cursor
        self._cursor += audio.size
        rms = float(np.sqrt(np.mean(audio * audio)))
        detected = self._vad(audio) if self._vad is not None else True
        voiced = bool(detected and rms >= self.threshold)
        self.is_speech = voiced
        events: list[tuple[str, np.ndarray]] = []

        if not self.active:
            if not voiced:
                self.pre.append(audio.copy())
                return events
            self.active = list(self.pre)
            self.pre.clear()
            self.count = sum(a.size for a in self.active)
            self._active_start = block_start - self.count
            self._commit_start = self._active_start
            self.speech_samples = 0

        self.active.append(audio.copy())
        self.count += audio.size
        if voiced:
            self.silent_count = 0
            self.speech_samples += audio.size
        else:
            self.silent_count += audio.size

        if self.count >= self.preview_samples and self.count - self.last_preview_at >= self.preview_samples:
            self.last_preview_at = self.count
            events.append(("partial", self._chunk(np.concatenate(self.active))))

        maximum = (self._initial_max_samples if self._initial_window_pending else self.max_samples)
        ending = self.silent_count >= self.silence_samples or self.count >= maximum
        if ending:
            if self.speech_samples >= int(0.18 * RATE):
                complete = np.concatenate(self.active)
                # With an audible pause, end cleanly. With continuous speech,
                # defer the tail and keep overlapping context for the next ASR.
                hard = self.count >= maximum and self.silent_count < self.silence_samples
                chunk = self._chunk(complete, hard_cut=hard)
                events.append(("final", chunk))
                self._initial_window_pending = False
                if hard:
                    keep = min(self.overlap_samples, complete.size)
                    self.active = [complete[-keep:].copy()]
                    self._active_start = self._cursor - keep
                    self._commit_start = chunk.commit_end_sample
                    self.count = keep
                    self.speech_samples = keep
                    self.silent_count = 0
                    self.last_preview_at = keep
                    return events
            self.reset()
        return events

    def flush(self) -> np.ndarray | None:
        if self.active and self.speech_samples >= int(0.18 * RATE):
            result = self._chunk(np.concatenate(self.active))
            self._initial_window_pending = False
            self.reset()
            return result
        self.reset()
        return None


INPUT_SOURCES = ("loopback", "microphone")


def _input_devices(input_source: str):
    """Enumerate by source class; never turn a mic choice into system capture."""
    if input_source not in INPUT_SOURCES:
        raise ValueError("Unsupported audio input source")
    import soundcard as sc
    loopback = input_source == "loopback"
    return [device for device in sc.all_microphones(include_loopback=True)
            if bool(getattr(device, "isloopback", False)) == loopback]


def input_devices(input_source: str = "loopback") -> list[tuple[str, str]]:
    """Return stable (id, label) choices for exactly one source class."""
    return [(str(device.id), str(device.name)) for device in _input_devices(input_source)]


def speaker_loopbacks() -> list[tuple[str, str]]:
    """Playback devices, retained for existing callers and diagnostics."""
    return input_devices("loopback")


def microphone_inputs() -> list[tuple[str, str]]:
    """Physical microphones/headsets; playback loopbacks are always excluded."""
    return input_devices("microphone")


def _selected_input(device_id: str, input_source: str):
    # SoundCard's get_microphone accepts fuzzy names. Selecting the exact ID
    # prevents a disconnected headset from silently capturing another device.
    matches = [device for device in _input_devices(input_source)
               if str(device.id) == str(device_id)]
    if len(matches) != 1:
        if input_source == "microphone":
            raise RuntimeError("未找到所选麦克风／耳麦。请连接设备、刷新列表并重新选择麦克风。")
        raise RuntimeError("未找到所选会议声音设备。请刷新列表，并选择 Zoom 使用的耳机或扬声器。")
    return matches[0]


class AudioCapture:
    def __init__(self, speaker_id: str, threshold: float,
                 on_audio: Callable[[str, np.ndarray], None],
                 on_level: Callable[[float], None],
                 on_status: Callable[[str], None],
                 on_activity: Callable[[bool], None] | None = None, *, context=0, maximum_sec=12.0,
                 streaming=False, on_ready: Callable[[], None] | None = None,
                 commit_interval_sec: float | None = None,
                 initial_maximum_sec: float | None = None,
                 input_source: str = "loopback", _start_gate: threading.Event | None = None):
        if input_source not in INPUT_SOURCES:
            raise ValueError("Unsupported audio input source")
        self.input_source = input_source
        self._start_gate = _start_gate
        self.speaker_id = speaker_id
        self.threshold = threshold
        self.on_audio = on_audio
        self.on_level = on_level
        self.on_status = on_status
        self.on_activity = on_activity or (lambda active: None)
        self.on_ready = on_ready or (lambda: None)
        self._context = context
        # A 6 s floor used to override the shorter caption preference and made
        # even a warm recognizer wait before it could confirm any translation.
        # Keep >=3 s for overlap and conservative acoustic speaker grouping.
        self.maximum_sec = min(12.0, max(3.0, float(maximum_sec)))
        self.commit_interval_sec = commit_interval_sec
        self.initial_maximum_sec = initial_maximum_sec
        self.streaming = bool(streaming)
        self._context_lock = threading.Lock()
        self.stop_requested = threading.Event()
        self.ready_event = threading.Event()
        self.error_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_requested.clear()
        self.ready_event.clear()
        self.error_event.clear()
        thread_name = "PrismMicrophoneAudio" if self.input_source == "microphone" else "ZoomSystemAudio"
        self.thread = threading.Thread(target=self._run, name=thread_name, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_requested.set()

    def set_context(self, value):
        """Assign the next capture-block epoch without relabeling queued audio.

        A phase boundary is applied by the recorder thread, which finalizes its
        old segment first. The UI never mutates a chunk already handed to ASR.
        """
        with self._context_lock:
            self._context = value

    def _get_context(self):
        with self._context_lock:
            return self._context

    def _emit_audio(self, kind, chunk, context):
        chunk.capture_context = context
        chunk.input_source = self.input_source
        self.on_audio(kind, chunk)

    def _run(self):
        try:
            from scipy.signal import resample_poly
            device = _selected_input(self.speaker_id, self.input_source)
            seg = Segmenter(threshold=self.threshold, maximum_sec=self.maximum_sec,
                            partial_sec=1.8 if self.streaming else .9,
                            commit_interval_sec=self.commit_interval_sec,
                            initial_maximum_sec=self.initial_maximum_sec)
            capture_context = self._get_context()
            with device.recorder(samplerate=CAPTURE_RATE, blocksize=4096) as recorder:
                self.ready_event.set()
                label = "麦克风" if self.input_source == "microphone" else "会议声音"
                self.on_status(f"监听中（{label}）：{device.name}")
                if not self.stop_requested.is_set():
                    self.on_ready()
                # A dual-source group opens both recorders before it starts
                # reading either one. A failed device never becomes an
                # unnoticed one-sided recording session.
                if self._start_gate is not None:
                    while not self.stop_requested.is_set() and not self._start_gate.wait(.05):
                        pass
                while not self.stop_requested.is_set():
                    raw = recorder.record(numframes=4800)  # 0.1 s
                    if raw is None or len(raw) == 0:
                        continue
                    next_context = self._get_context()
                    if next_context != capture_context:
                        last = seg.flush()
                        if last is not None:
                            self._emit_audio("final", last, capture_context)
                        # Silence pre-roll belongs to the previous phase too.
                        # Never prepend presentation audio to a new QA chunk.
                        seg.pre.clear()
                        capture_context = next_context
                    raw = np.asarray(raw, dtype=np.float32)
                    if raw.ndim not in (1, 2) or (raw.ndim == 2 and not raw.shape[1]):
                        raise RuntimeError("音频设备返回了不支持的声道格式。请重新选择输入设备。")
                    mono = raw.mean(axis=1) if raw.ndim > 1 else raw
                    level = float(np.sqrt(np.mean(mono * mono)))
                    self.on_level(min(level * 20.0, 1.0))
                    at_16k = resample_poly(mono, 1, 3).astype(np.float32, copy=False)
                    block_start = seg._cursor
                    events = seg.push(at_16k)
                    if self.streaming:
                        # Nonoverlapping blocks carry the same source clock as
                        # final Whisper windows, including silence for acoustics.
                        draft = AudioChunk(at_16k, start_sample=block_start,
                            commit_start_sample=block_start,
                            commit_end_sample=block_start + at_16k.size,
                            stream_id=seg.stream_id, speech_active=seg.is_speech)
                        self._emit_audio("stream", draft, capture_context)
                    self.on_activity(seg.is_speech)
                    for kind, chunk in events:
                        self._emit_audio(kind, chunk, capture_context)
            last = seg.flush()
            if last is not None:
                self._emit_audio("final", last, capture_context)
            self.on_status("音频监听已停止")
        except Exception as exc:
            self.error_event.set()
            self.on_status(f"音频设备错误：{exc}")


class DualAudioCapture:
    """Two separate source streams with one supervised capture lifecycle.

    No waveform mixing or inferred identity is used. Chunks retain their own
    stream clocks and explicit loopback/microphone origin. The supervisor does
    not finish until both recorder threads have closed and flushed their tails.
    """
    def __init__(self, speaker_id: str, microphone_id: str, threshold: float,
                 on_audio: Callable[[str, np.ndarray], None],
                 on_level: Callable[[float], None], on_status: Callable[[str], None],
                 on_activity: Callable[[bool], None] | None = None, *, context=0,
                 maximum_sec=12.0, streaming=False, on_ready: Callable[[], None] | None = None,
                 commit_interval_sec: float | None = None,
                 initial_maximum_sec: float | None = None):
        self.input_source = "both"
        self.speaker_id, self.microphone_id = speaker_id, microphone_id
        self.on_audio, self.on_level, self.on_status = on_audio, on_level, on_status
        self.on_activity = on_activity or (lambda active: None)
        self.on_ready = on_ready or (lambda: None)
        self._context = context
        self._context_lock = threading.Lock()
        self._level_lock = threading.Lock()
        self._levels = {source: 0.0 for source in INPUT_SOURCES}
        self.stop_requested = threading.Event()
        self.ready_event = threading.Event()
        self.error_event = threading.Event()
        self._start_gate = threading.Event()
        self.thread: threading.Thread | None = None
        self.captures: dict[str, AudioCapture] = {}
        self._options = dict(threshold=threshold, maximum_sec=maximum_sec, streaming=streaming,
                             commit_interval_sec=commit_interval_sec,
                             initial_maximum_sec=initial_maximum_sec)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_requested.clear()
        self.ready_event.clear()
        self.error_event.clear()
        self._start_gate.clear()
        self._levels = {source: 0.0 for source in INPUT_SOURCES}
        with self._context_lock:
            context = self._context
            self.captures = {
                source: AudioCapture(identity, on_audio=self.on_audio,
                    on_level=lambda value, source=source: self._level(source, value),
                    on_status=lambda message, source=source: self._status(source, message),
                    # Candidate speech must not prolong the interviewer's
                    # automatic question endpoint. Both still reach ASR.
                    on_activity=self.on_activity if source == "loopback" else None,
                    context=context, input_source=source, _start_gate=self._start_gate,
                    **self._options)
                for source, identity in (("loopback", self.speaker_id), ("microphone", self.microphone_id))
            }
        self.thread = threading.Thread(target=self._run, name="PrismDualAudio", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_requested.set()
        self._start_gate.set()
        for capture in tuple(self.captures.values()):
            capture.stop()

    def set_context(self, value):
        with self._context_lock:
            self._context = value
            for capture in self.captures.values():
                capture.set_context(value)

    def _level(self, source, value):
        with self._level_lock:
            self._levels[source] = value
            level = max(self._levels.values())
        self.on_level(level)

    def _status(self, source, message):
        label = "我的麦克风" if source == "microphone" else "会议声音"
        if message.startswith("监听中") and not self.ready_event.is_set():
            self.on_status(f"正在连接双方输入：{label}已就绪")
        else:
            self.on_status(f"{label} · {message}")

    def _run(self):
        try:
            for capture in self.captures.values():
                if self.stop_requested.is_set():
                    break
                capture.start()
            while not self.stop_requested.is_set():
                if any(capture.error_event.is_set() for capture in self.captures.values()):
                    raise RuntimeError("一侧音频设备无法使用，双方监听已停止；请检查会议输出和麦克风后重试。")
                if not self.ready_event.is_set() and all(capture.ready_event.is_set() for capture in self.captures.values()):
                    self.ready_event.set()
                    self._start_gate.set()
                    self.on_status("监听中：会议声音 + 我的麦克风")
                    if not self.stop_requested.is_set():
                        self.on_ready()
                if any(capture.thread is not None and not capture.thread.is_alive()
                       for capture in self.captures.values()):
                    raise RuntimeError("一侧音频线程已退出，双方监听已停止；请重新检查设备。")
                self.stop_requested.wait(.02)
        except Exception as exc:
            self.error_event.set()
            self.on_status(f"音频设备错误：{exc}")
        finally:
            self.stop()
            for capture in self.captures.values():
                if capture.thread:
                    capture.thread.join()

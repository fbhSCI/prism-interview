"""Regression tests for acoustic gating, bounded context and final completeness."""
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

from zoom_copilot.audio import AudioChunk, RATE, Segmenter, StreamingVoiceDetector
from zoom_copilot.speech import DEFAULT_GLOSSARY, RESEARCH_GLOSSARY, SpeechWorker


def eventually(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.005)
    assert predicate()


def segment(audio, instance):
    events = []
    for index in range(0, len(audio), 1600):
        events.extend(instance.push(audio[index:index + 1600]))
    return events


def test_default_prompt_is_empty_research_is_only_opt_in():
    assert DEFAULT_GLOSSARY == ""
    assert RESEARCH_GLOSSARY == ""
    assert SpeechWorker("test", glossary="User vocabulary").glossary == "User vocabulary"
    assert SpeechWorker("test").glossary == ""


def test_short_pause_retains_context_and_partial_is_fast():
    instance = Segmenter(vad=False)
    events = segment(np.concatenate([np.ones(RATE), np.zeros(RATE // 2), np.ones(RATE)]).astype(np.float32), instance)
    assert not [e for e in events if e[0] == "final"]
    partials = [x for kind, x in events if kind == "partial"]
    assert partials and partials[0].size <= RATE
    events.extend(segment(np.zeros(int(1.2 * RATE), np.float32), instance))
    finals = [x for kind, x in events if kind == "final"]
    assert len(finals) == 1 and finals[0].size >= 3 * RATE


def test_long_continuous_audio_has_contiguous_commits_and_flush_tail():
    instance = Segmenter(vad=False)
    events = segment(np.ones(36 * RATE, np.float32), instance)
    tail = instance.flush()
    assert tail is not None
    finals = [x for kind, x in events if kind == "final"] + [tail]
    assert len(finals) >= 3
    assert finals[0].commit_start_sample == 0
    assert finals[-1].commit_end_sample == 36 * RATE
    assert all(right.commit_start_sample == left.commit_end_sample for left, right in zip(finals, finals[1:]))
    assert all(x.size <= 12.1 * RATE for x in finals)
    assert any(x.hard_cut for x in finals)
    assert instance.flush() is None


def test_real_silero_rejects_silence_and_pure_tone_not_just_energy():
    pytest.importorskip("onnxruntime")
    pytest.importorskip("faster_whisper")
    instance = Segmenter(threshold=.005)
    silence = np.zeros(RATE, np.float32)
    tone = (.1 * np.sin(2 * np.pi * 220 * np.arange(2 * RATE) / RATE)).astype(np.float32)
    assert not segment(np.concatenate([silence, tone, silence]), instance)
    assert instance.flush() is None
    assert instance.vad_backend == "silero-onnx"


def test_word_commit_intervals_keep_boundary_word_only_in_next_chunk():
    worker = SpeechWorker("test")
    word = lambda text, start, end: SimpleNamespace(word=text, start=start, end=end)
    first = {"stream_id": "stream", "start_sample": 0, "commit_start_sample": 0,
             "commit_end_sample": int(11.4 * RATE), "hard_cut": True}
    part = SimpleNamespace(text="alpha old boundary", words=[word(" alpha", .1, .3), word(" old", 10.4, 10.8), word(" boundary", 11.3, 11.7)])
    text, _, timed = worker._decode_text([part], "final", first)
    assert text == "alpha old"
    assert timed is True
    second = {"stream_id": "stream", "start_sample": int(10.4 * RATE),
              "commit_start_sample": int(11.4 * RATE), "commit_end_sample": 24 * RATE, "hard_cut": False}
    part = SimpleNamespace(text="old boundary omega", words=[word(" old", .1, .3), word(" boundary", 1.0, 1.2), word(" omega", 2, 3)])
    text, _, timed = worker._decode_text([part], "final", second)
    assert text == "boundary omega"
    assert timed is True


def test_confidence_filter_has_no_domain_word_blacklist_and_rejects_bad_final():
    output = []
    class Model:
        def __init__(self, *args, **kwargs): pass
        def transcribe(self, audio, **kwargs):
            assert kwargs["initial_prompt"] is None and kwargs["vad_filter"] is True
            bad = audio[0] == 0
            return iter([SimpleNamespace(text="Mamba.", avg_logprob=-.8 if bad else -.1,
                                         no_speech_prob=.95 if bad else .01, compression_ratio=1.0)]), None
    worker = SpeechWorker("test", on_text=lambda *args: output.append(args), model_factory=Model)
    worker.start()
    assert worker.ready_event.wait(2)
    worker.submit("final", np.zeros(16000, np.float32))
    worker.submit("final", np.ones(16000, np.float32))
    eventually(lambda: len(output) == 2)
    worker.stop()
    worker.thread.join(2)
    assert output == [("rejected", "", {"recognition_gap": True}), ("final", "Mamba.", "")]


def test_stop_drains_all_already_accepted_finals_even_during_model_load():
    release = threading.Event()
    output = []
    class Model:
        def __init__(self, *args, **kwargs):
            assert release.wait(2)
        def transcribe(self, audio, **kwargs):
            return iter([SimpleNamespace(text=f"Final {int(audio[0])}")]), None
    worker = SpeechWorker("test", on_text=lambda *args: output.append(args), model_factory=Model)
    worker.start()
    for i in range(8):
        assert worker.submit("final", np.full(1000, i, np.float32))
    worker.stop(drain=True)
    release.set()
    worker.thread.join(3)
    assert not worker.thread.is_alive()
    assert len(output) == 8 and output[-1][1] == "Final 7"
    assert worker.pending == 0


def test_pending_includes_queued_partial_and_stop_clears_it():
    worker = SpeechWorker("test", model_factory=lambda *args, **kwargs: None)
    assert worker.submit("partial", np.ones(1600, np.float32))
    assert worker.pending == 1
    worker.stop()
    assert worker.pending == 0


def test_inflight_partial_is_suppressed_when_final_arrives():
    started, release = threading.Event(), threading.Event()
    output, metrics = [], []
    class Model:
        def __init__(self, *args, **kwargs): pass
        def transcribe(self, audio, **kwargs):
            if audio[0] == 1:
                started.set()
                assert release.wait(2)
            return iter([SimpleNamespace(text=f"Text {int(audio[0])}")]), None
    worker = SpeechWorker("test", on_text=lambda *args: output.append(args), on_metrics=metrics.append, model_factory=Model)
    worker.start()
    assert worker.ready_event.wait(2)
    worker.submit("partial", np.ones(1600, np.float32))
    assert started.wait(2)
    worker.submit("final", np.full(1600, 2, np.float32))
    release.set()
    eventually(lambda: any(x[0] == "final" for x in output))
    worker.stop()
    worker.thread.join(2)
    assert output == [("final", "Text 2", "")]
    assert metrics[0]["asr_stale_partial"] is True

from types import SimpleNamespace
import threading
import time

import numpy as np

from zoom_copilot.audio import AudioChunk, AudioCapture
from zoom_copilot.speech import SpeechWorker
from zoom_copilot.streaming_asr import StreamingDraftWorker, _valid


def until(predicate, seconds=3):
    deadline = time.perf_counter() + seconds
    while not predicate() and time.perf_counter() < deadline:
        time.sleep(.005)
    assert predicate()


class Stream:
    def __init__(self):
        self.accepted = 0
        self.decoded = 0
    def accept_waveform(self, rate, audio):
        assert rate == 16000
        self.accepted += len(audio)


class Recognizer:
    def __init__(self):
        self.recognizer = self
        self.streams = []
    def create_stream(self):
        stream = Stream()
        self.streams.append(stream)
        return stream
    def is_ready(self, stream):
        return stream.decoded < stream.accepted
    def decode_stream(self, stream):
        stream.decoded = stream.accepted
    def get_result(self, stream):
        count = stream.decoded // 1600
        return SimpleNamespace(tokens=[" ONE", " TWO", " THREE"][:count],
                               timestamps=[.0, .1, .2][:count], text="unused")


def test_committing_final_retires_whole_bpe_words_without_fragmenting_them():
    result = SimpleNamespace(tokens=[" AN", "A", "LY", "ZER", " IS", " GOOD"],
                             timestamps=[.0, .1, .2, .3, .4, .5])
    assert StreamingDraftWorker._text_after(result, 0, 2000) == "Is good"
    assert StreamingDraftWorker._text_after(result, 16000, 16000) == "Analyzer is good"
    assert StreamingDraftWorker._text_after(result, 0, 16000) == ""


def test_phase_boundary_resets_online_encoder_and_keeps_drafts_uncommitted():
    recognizer = Recognizer()
    texts = []
    worker = StreamingDraftWorker(None, lambda *args: texts.append(args), recognizer_factory=lambda: recognizer)
    worker.start()
    assert worker.ready_event.wait(2)
    worker.submit(np.ones(1600), {"stream_id": "a", "capture_context": 1, "start_sample": 0})
    until(lambda: len(texts) == 1)
    worker.submit(np.ones(1600), {"stream_id": "a", "capture_context": 2, "start_sample": 1600})
    until(lambda: len(texts) == 2)
    worker.stop(False)
    worker.thread.join(2)
    assert len(recognizer.streams) == 2
    assert [item[1] for item in texts] == ["One", "One"]
    assert [item[2]["phase_epoch"] for item in texts] == [1, 2]
    assert all(item[0] == "partial" for item in texts)


def test_final_commit_retains_only_later_draft_words():
    texts = []
    worker = StreamingDraftWorker(None, lambda *args: texts.append(args), recognizer_factory=Recognizer)
    worker.start()
    assert worker.ready_event.wait(2)
    worker.submit(np.ones(1600), {"stream_id": "a", "capture_context": 1, "start_sample": 0})
    until(lambda: bool(texts))
    worker.commit({"stream_id": "a", "capture_context": 1, "commit_end_sample": 1600})
    worker.submit(np.ones(1600), {"stream_id": "a", "capture_context": 1, "start_sample": 1600})
    until(lambda: len(texts) == 2)
    worker.stop(False)
    worker.thread.join(2)
    assert [item[1] for item in texts] == ["One", "Two"]


def test_delayed_whisper_revision_preserves_newer_online_tail_and_commit_boundary():
    worker = StreamingDraftWorker(None, lambda *args: None, recognizer_factory=Recognizer)
    key = ("a", 1)
    result = SimpleNamespace(tokens=[" GOOD", " WORD", " LATE", " TAIL"], timestamps=[0, .1, .2, .3])
    metadata = {"stream_id": "a", "capture_context": 1, "commit_end_sample": 3000}
    worker.refine(metadata, [{"text": " Correct", "start": 0, "end": 1000},
                             {"text": " prefix", "start": 1700, "end": 2900}])
    text, start, backend, words = worker._hypothesis(result, 0, key)
    assert (text, start, backend) == ("Correct prefix Late tail", 0, "whisper_revision")
    assert [word["text"].strip() for word in words] == ["Correct", "prefix", "LATE", "TAIL"]
    # A slower older revision cannot overwrite the new correction.
    worker.refine(dict(metadata, commit_end_sample=2000), [{"text": " obsolete", "start": 0, "end": 1900}])
    assert worker._hypothesis(result, 0, key)[0] == "Correct prefix Late tail"
    worker.commit(dict(metadata, commit_end_sample=1600))
    assert worker._hypothesis(result, 0, key)[0] == "prefix Late tail"
    worker.commit(dict(metadata, commit_end_sample=3000))
    assert worker._hypothesis(result, 0, key)[0] == "Late tail"


def test_pinned_nemo_eightfold_subsampling_uses_correct_source_clock():
    result = SimpleNamespace(tokens=[" earlier", " later"], timestamps=[.25, .5])
    words = StreamingDraftWorker._words_after(result, 0, 9000, timestamp_scale=2.0)
    assert words == [{"text": " later", "start": 16000, "end": 17280}]


def test_commit_republishes_even_same_text_and_uses_lexical_end_clock():
    texts = []
    worker = StreamingDraftWorker(None, lambda *args: texts.append(args), recognizer_factory=Recognizer)
    worker.start()
    assert worker.ready_event.wait(2)
    metadata = {"stream_id": "a", "capture_context": 1, "start_sample": 0}
    worker.submit(np.ones(1600), metadata)
    until(lambda: len(texts) == 1)
    worker.commit(dict(metadata, commit_end_sample=0))
    worker.submit(np.ones(100), dict(metadata, start_sample=1600))
    until(lambda: len(texts) == 2)
    worker.stop(False)
    worker.thread.join(2)
    assert [item[1] for item in texts] == ["One", "One"]
    assert texts[1][2]["commit_generation"] == 1
    assert texts[1][2]["text_end_sample"] == 640
    assert texts[1][2]["end_sample"] == 1700


def test_silence_gates_draft_display_but_still_decodes_acoustic_context():
    texts = []
    worker = StreamingDraftWorker(None, lambda *args: texts.append(args), recognizer_factory=Recognizer)
    worker.start()
    assert worker.ready_event.wait(2)
    worker.submit(np.ones(1600), {"stream_id": "a", "capture_context": 1, "start_sample": 0, "speech_active": False})
    until(lambda: worker._queued_samples == 0)
    assert not texts
    worker.submit(np.ones(1600), {"stream_id": "a", "capture_context": 1, "start_sample": 1600, "speech_active": True})
    until(lambda: bool(texts))
    worker.stop(False)
    worker.thread.join(2)
    assert texts[0][1] == "One two"


def test_final_commit_clears_stale_draft_after_vad_trailing_silence():
    texts = []
    worker = StreamingDraftWorker(None, lambda *args: texts.append(args), recognizer_factory=Recognizer)
    worker.start()
    assert worker.ready_event.wait(2)
    metadata = {"stream_id": "a", "capture_context": 1, "start_sample": 0, "speech_active": True}
    worker.submit(np.ones(1600), metadata)
    until(lambda: len(texts) == 1)
    worker.commit(dict(metadata, commit_end_sample=1600))
    worker.submit(np.zeros(32000), dict(metadata, start_sample=1600, speech_active=False))
    until(lambda: len(texts) == 2)
    worker.stop(False)
    worker.thread.join(2)
    assert texts[1][1] == "" and texts[1][2]["draft_words"] == []
    assert texts[1][2]["commit_generation"] == 1


def test_draft_failure_does_not_lose_final_whisper_words():
    texts = []
    class FinalModel:
        def __init__(self, *args, **kwargs):
            pass
        def transcribe(self, *args, **kwargs):
            return iter([SimpleNamespace(text="Why does it work?")]), None
    def broken():
        raise RuntimeError("bad draft model")
    worker = SpeechWorker("fake", on_text=lambda *args: texts.append(args), model_factory=FinalModel,
                          streaming_model_root="unused", streaming_factory=broken)
    worker.start()
    assert worker.ready_event.wait(2)
    assert worker.streaming.failed_event.is_set()
    worker.submit("final", np.ones(1600))
    worker.stop(True)
    worker.thread.join(2)
    assert [(item[0], item[1]) for item in texts] == [("final", "Why does it work?")]


def test_stream_blocks_are_never_added_to_final_queue():
    recognizer = Recognizer()
    worker = SpeechWorker("unused")
    worker.streaming = StreamingDraftWorker(None, lambda *args: None, recognizer_factory=lambda: recognizer)
    chunk = AudioChunk(np.ones(1600), start_sample=0, commit_start_sample=0,
                       commit_end_sample=1600, stream_id="a", capture_context=3)
    assert worker.submit("stream", chunk)
    assert worker.pending == 0
    assert worker.streaming._blocks[0][1]["capture_context"] == 3


def test_draft_overflow_is_explicit_and_does_not_silently_evict_audio():
    worker = StreamingDraftWorker(None, lambda *args: None, recognizer_factory=Recognizer)
    assert not worker.submit(np.ones(6 * 16000), {})
    assert worker.failed_event.is_set()
    assert "5" in worker.failure


def test_public_model_integrity_detects_corrupted_download(tmp_path):
    import hashlib
    path = tmp_path / "model"
    path.write_bytes(b"test weights")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert _valid(path, digest)
    path.write_bytes(b"changed weights")
    assert not _valid(path, digest)


def test_capture_ready_only_after_recorder_context_is_open(monkeypatch):
    import sys
    from zoom_copilot import audio as audio_module
    events = []
    capture = AudioCapture("fake", .005, lambda *args: None, lambda *args: None,
                           lambda text: events.append("status:" + text),
                           on_ready=lambda: events.append("ready"))
    class Recorder:
        def __enter__(self):
            assert not capture.ready_event.is_set()
            events.append("opened")
            return self
        def __exit__(self, *args):
            events.append("closed")
        def record(self, numframes):
            assert capture.ready_event.is_set() and "ready" in events
            capture.stop()
            return np.zeros((numframes, 1), dtype=np.float32)
    device = SimpleNamespace(id="fake", isloopback=True, name="fake", recorder=lambda **kwargs: Recorder())
    monkeypatch.setitem(sys.modules, "soundcard", SimpleNamespace(all_microphones=lambda **kwargs: [device]))
    real_segmenter = audio_module.Segmenter
    monkeypatch.setattr(audio_module, "Segmenter", lambda **kwargs: real_segmenter(vad=False, **kwargs))
    capture._run()
    assert events.index("opened") < events.index("ready") < events.index("closed")
    assert not capture.error_event.is_set()


def test_capture_open_error_never_signals_ready(monkeypatch):
    import sys
    ready = []
    capture = AudioCapture("missing", .005, lambda *args: None, lambda *args: None,
                           lambda *args: None, on_ready=lambda: ready.append(True))
    monkeypatch.setitem(sys.modules, "soundcard", SimpleNamespace(all_microphones=lambda **kwargs: []))
    capture._run()
    assert capture.error_event.is_set() and not capture.ready_event.is_set() and not ready

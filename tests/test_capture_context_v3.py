"""Capture epochs survive queueing, decoding and rejected caption callbacks."""
from types import SimpleNamespace
import sys
import time

import numpy as np
import pytest

from zoom_copilot import audio as audio_module
from zoom_copilot.audio import AudioCapture, AudioChunk, RATE, Segmenter
from zoom_copilot.speech import SpeechWorker


def eventually(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not predicate():
        time.sleep(.005)
    assert predicate()


def chunk(value, epoch):
    return AudioChunk(np.full(RATE, value, np.float32), start_sample=0,
                      commit_start_sample=0, commit_end_sample=RATE,
                      stream_id=f"stream-{epoch}", capture_context=epoch)


def test_audio_chunk_view_keeps_immutable_capture_epoch_value():
    original = chunk(1, 5)
    copied = original.copy()
    sliced = original[:800]
    original.capture_context = 6
    assert copied.capture_context == sliced.capture_context == 5


@pytest.mark.parametrize("old_value", [0.0, 1.0])
def test_capture_phase_change_flushes_old_audio_and_discards_old_preroll(monkeypatch, old_value):
    # Deterministic fake WASAPI blocks exercise the production capture loop.
    # Segmentation is real; VAD and resampling are isolated from this epoch test.
    emitted, statuses = [], []
    capture = AudioCapture("test", .005, lambda kind, data: emitted.append((kind, data)),
                           lambda value: None, statuses.append, context=41)
    class Recorder:
        index = 0
        def __enter__(self): return self
        def __exit__(self, *args): return False
        def record(self, numframes):
            self.index += 1
            if self.index == 5:
                capture.set_context(42)
            if self.index == 7:
                capture.stop()
            return np.full((numframes, 1), old_value if self.index < 5 else 2.0, np.float32)
    device = SimpleNamespace(id="test", isloopback=True, name="test", recorder=lambda **kwargs: Recorder())
    monkeypatch.setitem(sys.modules, "soundcard", SimpleNamespace(all_microphones=lambda **kwargs: [device]))
    monkeypatch.setattr(audio_module, "Segmenter", lambda **kwargs: Segmenter(vad=False, **kwargs))
    import scipy.signal
    monkeypatch.setattr(scipy.signal, "resample_poly", lambda values, up, down: values[::down])
    capture._run()
    assert statuses[-1] == "音频监听已停止"
    finals = [data for kind, data in emitted if kind == "final"]
    if old_value:
        assert len(finals) == 2
        assert finals[0].capture_context == 41
        assert finals[0].size == 4 * 1600 and np.all(finals[0] == 1)
    else:
        assert len(finals) == 1
    assert finals[-1].capture_context == 42
    assert finals[-1].size == 3 * 1600 and np.all(finals[-1] == 2)
    assert finals[-1].start_sample == 4 * 1600


def test_submit_snapshots_epoch_and_callbacks_mark_timed_untimed_and_rejected():
    emitted, metrics = [], []
    class Model:
        def __init__(self, *args, **kwargs): pass
        def transcribe(self, values, **kwargs):
            value = int(values[0])
            words = [SimpleNamespace(word=" test", start=.1, end=.3)] if value == 1 else None
            return iter([SimpleNamespace(text="test", words=words, avg_logprob=-2 if value == 0 else -.1)]), None
    worker = SpeechWorker("test", on_text=lambda *args: emitted.append(args),
                          on_metrics=metrics.append, model_factory=Model)
    # Queued metadata must be snapshotted even if the caller later mutates its ndarray.
    first = chunk(1, 51)
    assert worker.submit("partial", first)
    first.capture_context = 999
    worker.start()
    assert worker.ready_event.wait(2)
    eventually(lambda: len(emitted) == 1)
    assert worker.submit("final", chunk(2, 52))
    eventually(lambda: len(emitted) == 2)
    assert worker.submit("final", chunk(0, 53))
    eventually(lambda: len(emitted) == 3)
    worker.stop()
    worker.thread.join(2)
    assert [(kind,text,{key:ctx[key] for key in ('phase_epoch','word_timed')}) for kind,text,ctx in emitted] == [
        ("partial", "test", {"phase_epoch": 51, "word_timed": True}),
        ("final", "test", {"phase_epoch": 52, "word_timed": False}),
        ("rejected", "", {"phase_epoch": 53, "word_timed": False}),
    ]
    assert emitted[0][2]['stream_id']=='stream-51'
    assert emitted[0][2]['end_sample']==16000
    assert emitted[0][2]['words']==[{'text':' test','start':1600,'end':4800}]
    # Generic diagnostics never contain recognized content.
    assert "test" not in str(metrics)


@pytest.mark.parametrize("words", [
    None,
    [SimpleNamespace(word=" alpha", start=None, end=.2)],
    [SimpleNamespace(word=" alpha", start=.2, end=.1)],
    [SimpleNamespace(word=" alpha", start=float("nan"), end=.2)],
    [SimpleNamespace(word=" alpha", start=.1, end=.2)],  # missing beta
])
def test_incomplete_word_alignment_falls_back_to_whole_text(words):
    metadata = dict(stream_id="stream", start_sample=0, commit_start_sample=0,
                    commit_end_sample=RATE, capture_context=7)
    worker = SpeechWorker("test")
    text, rejected, timed = worker._decode_text([SimpleNamespace(text="alpha beta", words=words)], "final", metadata)
    assert text == "alpha beta" and rejected == [] and timed is False


def test_context_without_interval_metadata_retains_text_and_is_not_word_timed():
    text, rejected, timed = SpeechWorker("test")._decode_text(
        [SimpleNamespace(text="Why?")], "final", {"capture_context": 7})
    assert (text, rejected, timed) == ("Why?", [], False)


def test_accepted_overlap_only_flush_is_empty_commit_not_rejected():
    emitted=[]
    class Model:
        def __init__(self,*args,**kwargs):pass
        def transcribe(self,values,**kwargs):
            word=SimpleNamespace(word=' question',start=.1,end=.3)
            return iter([SimpleNamespace(text='question',words=[word],avg_logprob=-.1)]),None
    worker=SpeechWorker('test',on_text=lambda *args:emitted.append(args),model_factory=Model)
    first=AudioChunk(np.ones(RATE,dtype=np.float32),start_sample=0,commit_start_sample=0,
                     commit_end_sample=RATE,stream_id='overlap',capture_context=1)
    overlap=AudioChunk(np.ones(RATE,dtype=np.float32),start_sample=0,commit_start_sample=RATE,
                       commit_end_sample=RATE*2,stream_id='overlap',capture_context=1)
    worker.submit('final',first);worker.submit('final',overlap);worker.start()
    eventually(lambda:len(emitted)==2)
    worker.stop();worker.thread.join(2)
    assert emitted[0][:2]==('final','question')
    assert emitted[1][:2]==('final','')
    assert emitted[1][2]['empty_commit'] is True

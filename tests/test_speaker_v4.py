from types import SimpleNamespace

import numpy as np

from zoom_copilot.audio import AudioChunk
from zoom_copilot.speaker import OnlineSpeakerDiarizer
from zoom_copilot.speech import SpeechWorker


def test_acoustic_identity_repeats_and_ambiguity_does_not_force_a_person(tmp_path):
    diarizer = OnlineSpeakerDiarizer(tmp_path, extractor=lambda a: [1, 0, 0])
    assert diarizer._identify(np.array([1., 0, 0]))[0] == "speaker_1"
    assert diarizer._identify(np.array([0., 1, 0]))[0] == "speaker_2"
    assert diarizer._identify(np.array([1., 0, 0]))[0] == "speaker_1"
    assert diarizer._identify(np.array([.707, .707, 0]))[0] is None
    assert len(diarizer.profiles) == 2


def test_short_audio_cannot_invent_a_person(tmp_path):
    diarizer = OnlineSpeakerDiarizer(tmp_path, extractor=lambda a: [1, 0])
    assert diarizer._identify(np.array([1., 0]), enough_audio=False)[0] is None
    assert not diarizer.profiles


def test_missing_model_is_explicit_and_keeps_all_text(tmp_path):
    diarizer = OnlineSpeakerDiarizer(tmp_path)
    assert not diarizer.prepare()
    result = diarizer.annotate(np.zeros(16000), [], {}, "Every original word stays.")
    assert result[0]["text"] == "Every original word stays."
    assert result[0]["speaker_id"] is None
    assert result[0]["speaker_label"] == "说话人未确认"


def test_pause_retains_turn_but_return_after_another_voice_starts_new_turn(tmp_path):
    diarizer = OnlineSpeakerDiarizer(tmp_path, extractor=lambda a: [1, 0])
    a = diarizer._context("speaker_1", .9, 0, 16000, "stream")
    b = diarizer._context("speaker_1", .9, 6*16000, 8*16000, "stream")
    c = diarizer._context("speaker_2", .9, 8*16000, 9*16000, "stream")
    d = diarizer._context("speaker_1", .9, 10*16000, 11*16000, "stream")
    assert a["turn_id"] == b["turn_id"]
    assert len({b["turn_id"], c["turn_id"], d["turn_id"]}) == 3


def test_word_grouping_is_acoustic_and_lossless(tmp_path):
    diarizer = OnlineSpeakerDiarizer(tmp_path, extractor=lambda a: [1, 0])
    # Deterministic acoustic vectors model two clean non-overlapping windows.
    diarizer.window = 2 * 16000
    diarizer.hop = 2 * 16000
    diarizer.extractor = lambda a: [1, 0] if a.mean() < .15 else [0, 1]
    audio = np.r_[np.full(32000, .1), np.full(32000, .2)].astype(np.float32)
    words = [{"text": " Alpha", "start": 0, "end": 15000},
             {"text": " beta.", "start": 16000, "end": 31000},
             {"text": " Gamma", "start": 32000, "end": 47000},
             {"text": " delta.", "start": 48000, "end": 64000}]
    result = diarizer.annotate(audio, words,
        {"start_sample": 0, "commit_start_sample": 0, "commit_end_sample": 64000},
        "Alpha beta. Gamma delta.")
    assert [p["speaker_id"] for p in result] == ["speaker_1", "speaker_2"]
    assert " ".join(p["text"] for p in result) == "Alpha beta. Gamma delta."
    assert len({p["turn_id"] for p in result}) == 2


def test_bad_embeddings_are_unconfirmed(tmp_path):
    diarizer = OnlineSpeakerDiarizer(tmp_path, extractor=lambda a: [float("nan"), 0])
    words = [{"text": " still here", "start": 0, "end": 32000}]
    result = diarizer.annotate(np.full(32000, .1), words, {}, "still here")
    assert result[0]["speaker_id"] is None
    assert result[0]["text"] == "still here"


def test_speaker_failure_cannot_drop_final_or_capture_epoch(tmp_path):
    class Model:
        def __init__(self, *args, **kwargs): pass
        def transcribe(self, *args, **kwargs):
            words = [SimpleNamespace(word=" Original", start=.1, end=.8),
                     SimpleNamespace(word=" words.", start=.8, end=1.8)]
            return iter([SimpleNamespace(text=" Original words.", words=words)]), None

    class BrokenSpeaker:
        last_metrics = {}
        def prepare(self): return True
        def annotate(self, *args): raise RuntimeError("speaker test failure")

    emitted = []
    worker = SpeechWorker("test", model_factory=Model, diarizer=BrokenSpeaker(),
                          on_text=lambda *args: emitted.append(args))
    worker.start()
    assert worker.ready_event.wait(3)
    chunk = AudioChunk(np.full(32000, .1), start_sample=0, commit_start_sample=0,
                       commit_end_sample=32000, stream_id="s", capture_context=42)
    assert worker.submit("final", chunk)
    worker.stop(drain=True)
    worker.thread.join(3)
    assert not worker.thread.is_alive()
    assert emitted[0][0:2] == ("final", "Original words.")
    assert emitted[0][2]["phase_epoch"] == 42
    assert emitted[0][2]["speaker_id"] is None


def test_many_queued_finals_are_drained_with_speaker_metadata(tmp_path):
    class Model:
        def __init__(self, *args, **kwargs): pass
        def transcribe(self, *args, **kwargs):
            return iter([SimpleNamespace(text=" Retained.", words=None)]), None

    diarizer = OnlineSpeakerDiarizer(tmp_path, extractor=lambda a: [1, 0])
    emitted = []
    worker = SpeechWorker("test", model_factory=Model, diarizer=diarizer,
                          on_text=lambda *args: emitted.append(args))
    worker.start()
    assert worker.ready_event.wait(3)
    for _ in range(8):
        assert worker.submit("final", np.full(32000, .1))
    worker.stop(drain=True)
    worker.thread.join(3)
    assert len(emitted) == 8
    assert all(e[1] == "Retained." and e[2]["speaker_id"] is None for e in emitted)


def test_incomplete_speaker_result_falls_back_before_any_text_is_emitted(tmp_path):
    class Model:
        def __init__(self, *args, **kwargs): pass
        def transcribe(self, *args, **kwargs):
            return iter([SimpleNamespace(text=" All original words.", words=None)]), None

    class IncompleteSpeaker:
        last_metrics = {}
        def prepare(self): return True
        def annotate(self, *args): return [{"text": "Only some words."}]

    emitted = []
    worker = SpeechWorker("test", model_factory=Model, diarizer=IncompleteSpeaker(),
                          on_text=lambda *args: emitted.append(args))
    worker.start()
    assert worker.ready_event.wait(3)
    worker.submit("final", np.full(32000, .1))
    worker.stop(drain=True)
    worker.thread.join(3)
    assert len(emitted) == 1
    assert emitted[0][1] == "All original words."
    assert emitted[0][2]["speaker_id"] is None

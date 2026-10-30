"""Bound latency without dropping overlap, revisions, formulas or final tails."""
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from zoom_copilot.audio import AudioCapture, Segmenter, RATE
from zoom_copilot.config import Preferences
from zoom_copilot.translation import TranslationWorker
from zoom_copilot.speaker import OnlineSpeakerDiarizer


def test_confirmed_caption_window_is_shorter_without_ignoring_requested_value():
    assert Preferences().caption_window_seconds == 3.6
    for requested, actual in ((3.2, 3.2), (.5, 3.0), (6, 6.0), (30, 12.0)):
        capture = AudioCapture("test", .005, lambda *a: None, lambda *a: None,
                               lambda *a: None, maximum_sec=requested)
        assert capture.maximum_sec == actual


def test_short_confirmed_windows_tile_continuous_audio_with_context_and_tail():
    segmenter = Segmenter(vad=False, maximum_sec=3.2)
    finals = []
    for _ in range(179):
        finals.extend(audio for kind, audio in segmenter.push(np.ones(1600, np.float32))
                      if kind == "final")
    finals.append(segmenter.flush())
    assert finals[0].size == int(3.2 * RATE)
    assert finals[0].commit_end_sample >= int(2.6 * RATE)
    assert finals[-1].commit_end_sample == 179 * 1600
    assert all(a.commit_end_sample == b.commit_start_sample for a, b in zip(finals, finals[1:]))
    assert all(b.start_sample < b.commit_start_sample for b in finals[1:])
    assert all(len(audio) <= int(3.3 * RATE) for audio in finals)


def test_growing_card_reuses_coherent_prefix_and_revises_changed_tail():
    calls, metrics, results = [], [], []
    complete = threading.Event()
    def translate(text):
        calls.append(text)
        return "中:" + text
    def done(*args):
        results.append(args)
        complete.set()
    worker = TranslationWorker(SimpleNamespace(translate=translate), done, on_metrics=metrics.append)
    prefix = ('We evaluate the reconstruction under controlled changes in illumination, scattering, '
              'and object motion, using held out measurements and independent reference images '
              'to assess the limitations of each method.')
    tail = ('A partial phrase is now complete with the corresponding measurement assumptions '
            'and reconstruction conditions explicitly described, so that readers can verify '
            'exactly which evidence supports this conclusion.')
    worker.start()
    try:
        for source in (prefix + ' A partial.', prefix + ' ' + tail,
                       prefix + ' ' + tail + ' What next?'):
            complete.clear()
            worker.submit(("speaker-card", source), source)
            assert complete.wait(3)
        assert calls == [prefix, 'A partial.', tail, 'What next?']
        assert results[-1][1] == '中:' + prefix + ' 中:' + tail + ' 中:What next?'
        assert metrics[-1]["translation_cached_sentences"] == 2
        assert metrics[-1]["translation_model_calls"] == 1
        assert "first sentence" not in str(metrics)
    finally:
        worker.stop()
        worker.thread.join(3)


def test_sentence_cache_is_card_scoped_bounded_and_does_not_cache_failed_formula():
    calls = []
    worker = TranslationWorker(SimpleNamespace(translate=lambda text: calls.append(text) or text), lambda *a: None)
    assert worker._translate_confirmed(("a", "x"), "Dr. Smith uses 0.95. Why?")[0] == "Dr. Smith uses 0.95. Why?"
    assert len(calls) == 1
    worker._translate_confirmed(("b", "x"), "Dr. Smith uses 0.95. Why?")
    assert len(calls) == 2
    worker._maximum_cached_sentences = 2
    worker._translate_confirmed(("b", "y"), " ".join(
        ' '.join([label] * 25) + '.' for label in ('One', 'Two', 'Three')))
    assert len(worker._sentence_cache) == 2
    worker.translator = SimpleNamespace(translate=lambda text: "incorrect formula")
    with pytest.raises(ValueError, match="formula_placeholder"):
        worker._translate_confirmed(("formula", "x"), "s = D i.")
    assert ("formula", "s = D i.") not in worker._sentence_cache
    worker.stop()
    assert not worker._sentence_cache


def test_context_sensitive_translation_keeps_full_paragraph_for_disambiguation():
    calls = []
    translator = SimpleNamespace(context_sensitive=True,
        translate=lambda text: calls.append(text) or "完整上下文译文")
    worker = TranslationWorker(translator, lambda *a: None)
    source = "There are many AI models. Let us name one model each."
    assert worker._translate_confirmed(("a", source), source) == ("完整上下文译文", 0, 1)
    assert calls == [source]
    assert not worker._sentence_cache


def test_boundary_word_can_use_continuous_overlap_audio_without_reemitting_words(tmp_path):
    # Short isolated crops are deliberately ambiguous; enough real lookback
    # matches the established voice. No identity is assigned from the text.
    diarizer = OnlineSpeakerDiarizer(tmp_path,
        extractor=lambda audio: [1., 0.] if len(audio) >= 2*RATE else [0., 1.])
    diarizer._identify(np.array([1., 0.]))
    audio = np.full(3*RATE, .1, np.float32)
    words = [{"text": " detection.", "start": 2*RATE, "end": int(2.6*RATE)}]
    previous = [{"text": " good", "start": int(.3*RATE), "end": int(.9*RATE)},
                {"text": " at", "start": RATE, "end": int(1.3*RATE)},
                {"text": " object", "start": int(1.4*RATE), "end": int(1.9*RATE)}]
    result = diarizer.annotate(audio, words,
        {"start_sample": 0, "commit_start_sample": 2*RATE, "commit_end_sample": 3*RATE,
         "hard_cut": True, "acoustic_words": previous + words}, "detection.")
    assert len(result) == 1
    assert result[0]["text"] == "detection."
    assert result[0]["speaker_id"] == "speaker_1"


@pytest.mark.parametrize("sentence_end,word_end", [(" object.", 1.9), (" object", 1.4)])
def test_overlap_crop_never_crosses_sentence_end_or_audible_gap(tmp_path, sentence_end, word_end):
    diarizer = OnlineSpeakerDiarizer(tmp_path,
        extractor=lambda audio: [1., 0.] if len(audio) >= 2*RATE else [0., 1.])
    diarizer._identify(np.array([1., 0.]))
    audio = np.full(3*RATE, .1, np.float32)
    words = [{"text": " Got it.", "start": 2*RATE, "end": int(2.7*RATE)}]
    previous = [{"text": sentence_end, "start": int(.2*RATE), "end": int(word_end*RATE)}]
    result = diarizer.annotate(audio, words,
        {"start_sample": 0, "commit_start_sample": 2*RATE, "commit_end_sample": 3*RATE,
         "hard_cut": True, "acoustic_words": previous + words}, "Got it.")
    assert result[0]["speaker_id"] is None
    assert result[0]["text"] == "Got it."

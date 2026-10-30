"""Timed ASR segment seams preserve words, metadata and final commit bounds.

The worker uses an in-memory model double. No microphone/loopback, speech model,
network request or real recognition-accuracy claim is involved.
"""
from types import SimpleNamespace
import time

import numpy as np
import pytest

from zoom_copilot.audio import AudioChunk, RATE
from zoom_copilot.speech import SpeechWorker


def segment(tokens, offset=0.0):
    words = [SimpleNamespace(word=token, start=offset + index / 8,
                             end=offset + (index + 1) / 8)
             for index, token in enumerate(tokens)]
    return SimpleNamespace(text=''.join(tokens), words=words, avg_logprob=-.1)


def metadata(*, start=0, lower=0, upper=10, stream='segment-spacing'):
    return dict(stream_id=stream, start_sample=int(start * RATE),
                commit_start_sample=int(lower * RATE), commit_end_sample=int(upper * RATE),
                hard_cut=True)


@pytest.mark.parametrize('groups, expected', [
    ([[' life.'], ['Scheduling', ' is', ' difficult.']], 'life. Scheduling is difficult.'),
    ([['The', ' material.'], ['Physical', ' property', ' of', ' all.']],
     'The material. Physical property of all.'),
    ([[' calibration'], ['matrix']], 'calibration matrix'),
    ([[' life', '.'], ['Scheduling']], 'life. Scheduling'),
    ([[' Previous.'], [' Already', ' separated.']], 'Previous. Already separated.'),
    ([[' Previous. '], ['Already', ' separated.']], 'Previous. Already separated.'),
    ([[' It', ' can', "'t", ' recover', ' S', '3', ' or', ' -', '0', '.', '09', ' units.']],
     "It can't recover S3 or -0.09 units."),
    ([[' We', ' can'], ["'t", ' measure', ' S3.']], "We can't measure S3."),
    ([[' wavelength', '-', 'dependent', ' scattering.'], ['Noise', ' remains.']],
     'wavelength-dependent scattering. Noise remains.'),
    ([[' wavelength-'], ['dependent', ' scattering.']], 'wavelength-dependent scattering.'),
    ([[' Confidence', ' is', ' 0.'], ['95', ' in', ' this', ' example.']],
     'Confidence is 0.95 in this example.'),
    ([[' Confidence', ' is', ' 0'], ['.95', ' in', ' this', ' example.']],
     'Confidence is 0.95 in this example.'),
    ([[' Use', ' e.'], ['g.', ' a', ' calibrated', ' BRDF.']],
     'Use e.g. a calibrated BRDF.'),
    ([['The', ' U.'], ['S.', ' team', ' measured', ' S0.']],
     'The U.S. team measured S0.'),
    ([['Dr.'], ['Chen', ' measured', ' S0.']], 'Dr. Chen measured S0.'),
    ([['Use', ' Fig.', ' 2', ' and', ' Eq.', ' 3.'], ['The', ' BRDF', ' is', ' unchanged.']],
     'Use Fig. 2 and Eq. 3. The BRDF is unchanged.'),
    ([[' symbols', ':', ' I', '=', 'A', 'x', '+', 'b', ';', ' -1.95', ' μm.']],
     'symbols: I=Ax+b; -1.95 μm.'),
])
def test_missing_segment_space_is_restored_without_retokenizing_words(groups, expected):
    parts = [segment(tokens, index * 2) for index, tokens in enumerate(groups)]
    original_tokens = [[word.word for word in part.words] for part in parts]
    worker = SpeechWorker('not-loaded')
    text, rejected, timed = worker._decode_text(parts, 'final', metadata())
    assert text == expected and rejected == [] and timed
    # Speaker grouping and draft reuse concatenate this public metadata. They
    # must see the same separator, without shifted clocks or mutated ASR words.
    assert ''.join(word['text'] for word in worker._decoded_words).strip() == expected
    assert ''.join(word['text'] for word in worker._acoustic_words).strip() == expected
    assert [[word.word for word in part.words] for part in parts] == original_tokens
    assert [(word['start'], word['end']) for word in worker._decoded_words] == [
        (int(word.start * RATE), int(word.end * RATE)) for part in parts for word in part.words]


def timed_segment(*words):
    items = [SimpleNamespace(word=text, start=start, end=end) for text, start, end in words]
    return SimpleNamespace(text=''.join(word.word for word in items), words=items, avg_logprob=-.1)


def overlapping_frames():
    # The middle of Scheduling lies after the first commit. The word must
    # arrive exactly once in the next window, followed by another new segment.
    return [
        (metadata(upper=3), [
            timed_segment((' alpha', .1, .3), (' life.', 2.7, 2.9)),
            timed_segment(('Scheduling', 2.95, 3.25)),
        ]),
        (metadata(start=2, lower=3, upper=5), [
            timed_segment((' life.', .7, .9)),
            timed_segment(('Scheduling', .95, 1.25)),
            timed_segment(('continues', 1.3, 1.6), (' tail.', 2.8, 3.2)),
        ]),
        (metadata(start=4, lower=5, upper=6), [
            timed_segment((' tail.', .8, 1.2)),
        ]),
        (metadata(start=4, lower=6, upper=7), [
            timed_segment((' tail.', .8, 1.2)),
        ]),
    ]


def test_overlap_filter_and_empty_flush_do_not_duplicate_or_drop_boundary_words():
    worker = SpeechWorker('not-loaded')
    output = []
    for context, parts in overlapping_frames():
        text, rejected, timed = worker._decode_text(parts, 'final', context)
        assert not rejected and timed
        assert ''.join(word['text'] for word in worker._decoded_words).strip() == text
        output.append(text)
    assert output == ['alpha life.', 'Scheduling continues', 'tail.', '']
    assert worker._committed_until['segment-spacing'] == int(5.2 * RATE)


def test_actual_worker_callbacks_keep_spaced_text_word_metadata_and_flush_semantics():
    frames = overlapping_frames()
    emitted, metrics, statuses = [], [], []

    class Model:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, values, **kwargs):
            assert kwargs['word_timestamps'] is True
            assert kwargs['condition_on_previous_text'] is False
            assert kwargs['initial_prompt'] is None
            return iter(frames[int(values[0])][1]), None

    worker = SpeechWorker('not-loaded', model_factory=Model,
                          on_text=lambda *args: emitted.append(args),
                          on_metrics=metrics.append, on_status=statuses.append)
    for index, (context, _) in enumerate(frames):
        chunk = AudioChunk(np.full(4 * RATE, index, dtype=np.float32),
                           capture_context=17, **context)
        assert worker.submit('final', chunk)
    worker.start()
    try:
        assert worker.ready_event.wait(2)
        deadline = time.monotonic() + 4
        while len(emitted) < len(frames) and time.monotonic() < deadline:
            time.sleep(.005)
    finally:
        worker.stop(drain=True)
        worker.thread.join(2)
    assert not worker.thread.is_alive()
    assert not any('出错' in status for status in statuses)
    assert [(kind, text) for kind, text, _ in emitted] == [
        ('final', 'alpha life.'), ('final', 'Scheduling continues'),
        ('final', 'tail.'), ('final', ''),
    ]
    for _, text, context in emitted:
        assert context['word_timed'] is True and context['phase_epoch'] == 17
        assert ''.join(word['text'] for word in context['words']).strip() == text
    assert emitted[-1][2]['empty_commit'] is True
    assert all(context['recognition_gap'] is False for _, _, context in emitted)
    assert [row['asr_result_kind'] for row in metrics] == ['final', 'final', 'final', 'empty_commit']


def test_untimed_fallback_and_missing_interval_path_keep_original_behavior():
    parts = [SimpleNamespace(text='alpha.', words=None), SimpleNamespace(text='Scheduling', words=None)]
    worker = SpeechWorker('not-loaded')
    assert worker._decode_text(parts, 'final', metadata()) == ('alpha. Scheduling', [], False)
    assert worker._decoded_words == [] and worker._acoustic_words == []
    assert worker._decode_text(parts, 'final', {}) == ('alpha. Scheduling', [], False)


@pytest.mark.parametrize('clocked', [True, False])
def test_mixed_accepted_and_rejected_segments_preserve_visible_text_and_signal_gap(clocked):
    parts = [segment([' Could', ' you', ' explain'], 0),
             segment([' without', ' noise'], 1),
             segment(['the', ' calibration?'], 2)]
    parts[1].avg_logprob = -2.0
    emitted, metrics = [], []

    class Model:
        def __init__(self, *args, **kwargs):
            pass

        def transcribe(self, values, **kwargs):
            return iter(parts), None

    worker = SpeechWorker('not-loaded', model_factory=Model,
                          on_text=lambda *args: emitted.append(args), on_metrics=metrics.append)
    values = np.ones(4 * RATE, dtype=np.float32)
    audio = AudioChunk(values, capture_context=19, **metadata()) if clocked else values
    assert worker.submit('final', audio)
    worker.start()
    try:
        assert worker.ready_event.wait(2)
        deadline = time.monotonic() + 4
        while not emitted and time.monotonic() < deadline:
            time.sleep(.005)
    finally:
        worker.stop(drain=True)
        worker.thread.join(2)
    assert not worker.thread.is_alive()
    assert len(emitted) == 1
    kind, text, context = emitted[0]
    assert kind == 'final' and text == 'Could you explain the calibration?'
    assert context['recognition_gap'] is True
    if clocked:
        assert context['word_timed'] is True
        assert ''.join(word['text'] for word in context['words']).strip() == text
    else:
        assert context == {'recognition_gap': True}
    assert metrics[0]['asr_rejected_segments'][0]['reason'] == 'low_log_probability'
    assert 'calibration' not in str(metrics) and 'without' not in str(metrics)

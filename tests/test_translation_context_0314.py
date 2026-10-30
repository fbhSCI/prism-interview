"""Confirmed speech coherence, bounded context, exact caches and source safety.

All models and HTTP calls are fakes. No credentials, papers or audio are read.
"""
import json
from types import SimpleNamespace

import pytest

from zoom_copilot import translation_qwen as qwen
from zoom_copilot.config import Preferences
from zoom_copilot.translation import (
    TranslationWorker, _bounded_preceding_context, _bounded_translation_blocks,
    _coherent_translation_units,
)
from zoom_copilot.translation_protection import protect_translation_source


PREFIX = ('We evaluate the reconstruction under controlled changes in illumination, scattering, '
          'and object motion, using held out measurements and independent reference images '
          'to assess the limitations of each method.')
ALTERNATIVE = PREFIX.replace('controlled changes', 'independent changes')


def preferences():
    return Preferences(provider='qwen', translation_cloud_consent=True,
        api_base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        qwen_paid_usage_authorized=True, api_key='isolated-unit-test',
        candidate_profile='PRIVATE-CV', research_profile='PRIVATE-PAPERS')


class ContextTranslator:
    confirmed_only = True

    def __init__(self):
        self.calls = []

    def translate_batch_with_context(self, targets, contexts):
        self.calls.append((list(targets), list(contexts)))
        return ['中文：' + target for target in targets]


@pytest.mark.parametrize('source', [
    ('Basically, all of the weights in the motion synthesis network and the encoder '
     'and decoder are learned and are. Completely back-propagated.'),
    'I Mean. Or convey. The material. Physical property of all materials is discussed.',
    'We do not use 95% of the data. We use 5 and 5 measurements.',
])
def test_related_short_fragments_share_one_target_without_english_rewriting(source):
    assert _coherent_translation_units(source) == [source]


def test_dangling_auxiliary_reopens_only_tail_and_keeps_stable_prefix():
    translator = ContextTranslator()
    worker = TranslationWorker(translator, lambda *a: None)
    fragment = ('Basically, all of the weights in the motion synthesis network and the encoder '
                'and decoder are learned and are.')
    complete = fragment + ' Completely back-propagated.'
    worker._translate_confirmed(('speaker', 1), PREFIX + ' ' + fragment)
    result, hits, calls = worker._translate_confirmed(('speaker', 2), PREFIX + ' ' + complete)
    assert translator.calls == [([PREFIX, fragment], ['', PREFIX]), ([complete], [PREFIX])]
    assert hits == 1 and calls == 1
    assert result == '中文：' + PREFIX + ' 中文：' + complete


def test_context_is_available_even_when_preceding_unit_is_already_cached():
    translator = ContextTranslator()
    worker = TranslationWorker(translator, lambda *a: None)
    worker._translate_confirmed(('speaker', 1), PREFIX)
    target = 'It is not identifiable at 532 nm.'
    _, hits, calls = worker._translate_confirmed(('speaker', 2), PREFIX + ' ' + target)
    assert translator.calls[-1] == ([target], [PREFIX])
    assert hits == 1 and calls == 1
    _, hits, calls = worker._translate_confirmed(('speaker', 3), PREFIX + ' ' + target)
    assert hits == 2 and calls == 0


def test_relevant_context_changes_invalidate_target_cache_and_speakers_never_share():
    translator = ContextTranslator()
    worker = TranslationWorker(translator, lambda *a: None)
    target = 'Is that identifiable?'
    worker._translate_confirmed(('speaker', 1), PREFIX + ' ' + target)
    _, hits, _ = worker._translate_confirmed(('speaker', 2), ALTERNATIVE + ' ' + target)
    assert hits == 0
    assert translator.calls[-1] == ([ALTERNATIVE, target], ['', ALTERNATIVE])
    worker._translate_confirmed(('another-speaker', 1), target)
    assert translator.calls[-1] == ([target], [''])
    worker._translate_confirmed(('another-speaker', 2), PREFIX + ' ' + target)
    assert translator.calls[-1] == ([PREFIX, target], ['', PREFIX])


def test_irrelevant_far_past_changes_do_not_retranslate_an_identical_target():
    translator = ContextTranslator()
    worker = TranslationWorker(translator, lambda *a: None)
    recent = ' '.join(['recentobservation'] * 40) + '.'
    target = 'What does that mean?'
    worker._translate_confirmed(('speaker', 1), PREFIX + ' ' + recent + ' ' + target)
    worker._translate_confirmed(('speaker', 2), ALTERNATIVE + ' ' + recent + ' ' + target)
    assert target not in translator.calls[-1][0]
    assert translator.calls[0][1][-1] == _bounded_preceding_context(recent)


def test_context_suffix_never_starts_inside_a_formula_or_exceeds_bound():
    suffix = ' tail' * 117
    source = 'earlier ' * 100 + 's equals D i not M i' + suffix
    context = _bounded_preceding_context(source)
    assert context == suffix.strip()
    assert len(context) <= 600
    assert _bounded_preceding_context('x' * 700) == ''


def test_long_unit_boundaries_keep_complete_formulas_terms_numbers_and_negation():
    formula = 's equals D i not M i'
    source = ' '.join(['prefix'] * 98 + [formula + '.', 'not', 'material', 'reflectance',
                                      'at', '532', 'nm'] + ['tail'] * 125)
    parts = _bounded_translation_blocks(source, max_words=100)
    assert ' '.join(parts) == source
    assert any(formula in part for part in parts)
    assert any('not material reflectance' in part for part in parts)
    assert any('532 nm' in part for part in parts)
    assert all(len(part.split()) <= 100 and len(part) <= 1000 for part in parts)


def test_contextual_cloud_batches_are_bounded_and_keep_every_source_word():
    translator = ContextTranslator()
    worker = TranslationWorker(translator, lambda *a: None)
    paragraphs = ['Sentence ' + str(i) + ' ' + 'observation ' * 95 + '.' for i in range(12)]
    source = ' '.join(paragraphs)
    worker._translate_confirmed(('speaker', 1), source)
    targets = [target for batch, _ in translator.calls for target in batch]
    assert ' '.join(targets) == source
    for targets, contexts in translator.calls:
        assert len(targets) <= 8
        assert sum(map(len, targets)) + sum(map(len, contexts)) <= 2000
        assert all(len(target) <= 1000 and len(target.split()) <= 100 for target in targets)
        assert all(len(context) <= 600 for context in contexts)


def test_qwen_context_is_read_only_and_never_adds_field_terms_or_private_material(monkeypatch):
    seen = []
    def http(request, *args, **kwargs):
        seen.append(json.loads(request.data))
        return json.dumps({'translations': ['或者传达多孔信息，使用5次测量。']}, ensure_ascii=False)
    monkeypatch.setattr(qwen, 'stream_http', http)
    translator = qwen.QwenCaptionTranslator(preferences())
    target = 'Or convey porous information with 5 measurements.'
    context = 'We study polarization and inverse rendering with 95 measurements.'
    assert translator.translate_batch_with_context([target], [context]) == [
        '或者传达多孔信息，使用5次测量。']
    data = json.loads(seen[0]['messages'][1]['content'])
    assert data == {'segments': [target], 'terminology': {}, 'preceding_contexts': [context]}
    system = seen[0]['messages'][0]['content']
    assert 'read-only context' in system and 'never guess' in system
    assert 'PRIVATE' not in json.dumps(seen)
    assert not any('polarization' in value for value in data['segments'])


def test_qwen_validates_target_numbers_not_context_numbers(monkeypatch):
    monkeypatch.setattr(qwen, 'stream_http', lambda *a, **kw: json.dumps(
        {'translations': ['使用5次测量和95个通道。']}, ensure_ascii=False))
    translator = qwen.QwenCaptionTranslator(preferences())
    with pytest.raises(ValueError, match='数值'):
        translator.translate_batch_with_context(['Use 5 measurements.'], ['There are 95 channels.'])


@pytest.mark.parametrize('case', ['missing', 'repeated', 'invented'])
def test_qwen_rejects_missing_repeated_or_invented_formula_markers(case):
    protected = protect_translation_source('We use s equals D i not M i.')
    marker = protected.slots[0][0]
    output = {'missing': '我们使用该式。', 'repeated': '我们使用' + marker + marker,
              'invented': '我们使用' + marker + '和ZXQFORMULADEADBEEF0QXZ'}[case]
    with pytest.raises(ValueError, match='公式标记'):
        qwen._check_translation(protected.text, output)
    qwen._check_translation(protected.text, '我们使用' + marker + '。')


def test_formula_restore_is_atomic_for_a_contextual_batch():
    translator = ContextTranslator()
    translator.translate_batch_with_context = lambda targets, contexts: ['正确中文。', '缺失公式。']
    worker = TranslationWorker(translator, lambda *a: None)
    with pytest.raises(ValueError, match='formula_placeholder'):
        worker._translate_confirmed(('speaker', 1), PREFIX + ' We use s equals D i not M i.')
    assert not worker._sentence_cache


def test_preview_and_misaligned_context_never_reach_http(monkeypatch):
    seen = []
    monkeypatch.setattr(qwen, 'stream_http', lambda *a, **kw: seen.append(a))
    translator = qwen.QwenCaptionTranslator(preferences())
    worker = TranslationWorker(translator, lambda *a: None)
    assert worker.submit_preview('speaker', 'unconfirmed draft') is False
    with pytest.raises(ValueError, match='上下文'):
        translator.translate_batch_with_context(['confirmed target'], [])
    with pytest.raises(ValueError, match='上下文'):
        translator.translate_batch_with_context(['confirmed target'], [None])
    assert seen == []
    worker.stop()
    assert translator._cancel.is_set()

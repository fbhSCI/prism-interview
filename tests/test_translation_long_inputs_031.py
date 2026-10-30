"""Bound local MT inference without changing caption sources or losing words."""
from types import SimpleNamespace

import pytest

from zoom_copilot.translation import ArgosTranslator, _bounded_translation_blocks


class IdentityTokenizer:
    def encode(self, text, out_type=str):
        return [text]

    def decode(self, tokens):
        return ''.join(tokens)


def translator_stub(tmp_path, response=None):
    translator = ArgosTranslator(tmp_path)
    translator._tokenizer = IdentityTokenizer()
    calls = []

    class Engine:
        def translate_batch(self, batches, **kwargs):
            assert kwargs['max_decoding_length'] == 384
            output = []
            for batch in batches:
                calls.append(batch[0])
                hypothesis = response(batch[0]) if response else [batch[0]]
                output.append(SimpleNamespace(hypotheses=[hypothesis]))
            return output

    translator._translator = Engine()
    return translator, calls


def test_long_unpunctuated_input_sends_every_word_in_order_with_bounded_inputs(tmp_path):
    source = ' '.join(f'observation{number}' for number in range(450))
    translator, calls = translator_stub(tmp_path)
    result = translator.translate(source)
    assert len(calls) > 1
    assert ' '.join(calls) == source == result
    assert all(len(part.split()) <= 120 and len(part) <= 1000 for part in calls)
    assert translator.last_metrics['sentences'] == 1
    assert translator.last_metrics['long_sentences_split'] == 1
    assert translator.last_metrics['translation_blocks'] == len(calls)


def test_word_budget_and_character_budget_are_both_enforced_without_punctuation_edits():
    source = ' '.join(f'longobservationname{number:04d}' for number in range(160))
    blocks = _bounded_translation_blocks(source)
    assert ' '.join(blocks) == source
    assert len(blocks) > 1
    assert all(len(block) <= 1000 and len(block.split()) <= 120 for block in blocks)
    assert not any(mark in ''.join(blocks) for mark in '.!?')


@pytest.mark.parametrize('mark', [',', ';', ':'])
def test_existing_clause_punctuation_is_preferred_over_a_budget_boundary(mark):
    words = [f'word{number}' for number in range(170)]
    words[69] += mark
    source = ' '.join(words)
    blocks = _bounded_translation_blocks(source)
    assert blocks[0] == ' '.join(words[:70])
    assert ' '.join(blocks) == source


def test_terms_negation_and_numeric_units_remain_together_at_a_budget_boundary():
    source = ' '.join(['x'] * 118 + ['not', 'material', 'reflectance', 'at', '532', 'nm'] + ['y'] * 130)
    blocks = _bounded_translation_blocks(source)
    assert ' '.join(blocks) == source
    assert any('not material reflectance' in block for block in blocks)
    assert any('532 nm' in block for block in blocks)
    assert not any(block.endswith('not') for block in blocks)


def test_marker_failure_retranslates_whole_bounded_original_without_word_loss(tmp_path):
    source = ' '.join(['x'] * 118 + ['not', 'material', 'reflectance', 'at', '-0.95', 'with', '1,000', 'samples']
                      + ['y'] * 130)
    original_blocks = _bounded_translation_blocks(source)
    translator, calls = translator_stub(tmp_path,
        response=lambda text: ['标记丢失'] if '[TERM' in text else [text])
    result = translator.translate(source)
    assert result == source
    protected_block = next(part for part in original_blocks if 'material reflectance' in part)
    assert calls[-1] == protected_block
    assert len(calls) == len(original_blocks) + 1
    assert len(translator.last_warnings) == 1
    assert translator.last_metrics['term_fallback_sentences'] == 1
    assert '-0.95' in result and '1,000' in result and 'not material reflectance' in result


def test_short_connected_sentences_share_one_input_without_source_changes(tmp_path):
    translator, calls = translator_stub(tmp_path)
    source = 'We do not measure S3. We use 0.95 and 1,000 samples.'
    assert translator.translate(source) == source
    assert calls == [source]
    assert translator.last_metrics['long_sentences_split'] == 0


def test_single_indivisible_word_is_not_sliced_or_lost():
    word = 'x' * 1100
    source = 'prefix ' + word + ' suffix'
    assert _bounded_translation_blocks(source) == ['prefix', word, 'suffix']


def test_output_at_decoding_limit_is_reported_as_incomplete_not_returned(tmp_path):
    translator, _ = translator_stub(tmp_path, response=lambda text: ['字'] * 384)
    with pytest.raises(RuntimeError, match='长度上限'):
        translator.translate('We do not measure S3.')
    assert '长度上限' in translator.last_error


def test_marker_fallback_also_rejects_output_at_decoding_limit(tmp_path):
    translator, calls = translator_stub(tmp_path,
        response=lambda text: ['marker lost'] if '[TERM' in text else ['字'] * 384)
    with pytest.raises(RuntimeError, match='长度上限'):
        translator.translate('We do not estimate material reflectance.')
    assert len(calls) == 2
    assert calls[-1] == 'We do not estimate material reflectance.'

"""Predicate cuts seen in the rendering Q&A must not become tiny sentences."""
import re

import pytest

from zoom_copilot.transcript_text import (
    TranscriptRestorer, repair_fragment_boundaries, restore_utterance, sentence_continues,
    caption_tail_incomplete,
)


def lexical_tokens(value):
    # Punctuation-only repairs must not hide errors by dropping/reordering words.
    return re.findall(r"[\w]+(?:[-'’][\w]+)*%?", value.casefold())


RENDERING = (
    "I, Mean, it. Basically, all of the weights in the motion synthesis network "
    "and the encoder and decoder are learned and are. Completely, back-propagated."
)
RENDERING_REPAIRED = (
    "I mean, it—basically, all of the weights in the motion synthesis network "
    "and the encoder and decoder are learned and are completely back-propagated."
)


def test_actual_rendering_cut_keeps_every_spoken_word_and_raw():
    restored = restore_utterance(RENDERING)
    assert restored.raw == RENDERING
    assert restored.clean == RENDERING_REPAIRED
    assert restored.sentences == (RENDERING_REPAIRED,)
    assert lexical_tokens(restored.clean) == lexical_tokens(RENDERING)


def test_learned_punctuation_cannot_recreate_known_predicate_fragments():
    raw = RENDERING.replace(',', '').replace('.', '')
    result = TranscriptRestorer(punctuator=lambda _: RENDERING).restore(raw)
    assert result.clean == RENDERING_REPAIRED
    assert result.raw == raw
    assert result.engine == 'edge-punct-int8-cpu'
    assert lexical_tokens(result.clean) == lexical_tokens(raw)


@pytest.mark.parametrize('source,expected', [
    ('The channels are. Independently, calibrated.',
     'The channels are independently calibrated.'),
    ('We are. Not measuring S3.', 'We are not measuring S3.'),
    ('We have. Already calibrated the camera.', 'We have already calibrated the camera.'),
    ('The bias has. Never been measured.', 'The bias has never been measured.'),
    ('The weights were not. Jointly, optimized.', 'The weights were not jointly optimized.'),
    ('The operators are. Learned end-to-end.', 'The operators are learned end-to-end.'),
    ('The result is. Still being checked.', 'The result is still being checked.'),
    ('The method is. Known to require calibration.', 'The method is known to require calibration.'),
    ('The field is. Completely, Back-propagated.', 'The field is completely back-propagated.'),
    ('The field is. Completely, back propagated.', 'The field is completely back propagated.'),
    ('The parameters are completely, back-propagated.', 'The parameters are completely back-propagated.'),
    ('The parameters are completely, back propagated.', 'The parameters are completely back propagated.'),
    ('We use a particular mathematical. Operator.', 'We use a particular mathematical operator.'),
    ('This is an optical. System with 0.95 confidence.', 'This is an optical system with 0.95 confidence.'),
    ('We fit another differentiable spectral. Function.', 'We fit another differentiable spectral function.'),
    ('We use neural. Networks for motion.', 'We use neural networks for motion.'),
])
def test_syntactic_continuations_generalize_to_unseen_terms(source, expected):
    result = repair_fragment_boundaries(source)
    assert result == expected
    assert lexical_tokens(result) == lexical_tokens(source)
    assert repair_fragment_boundaries(result) == result


@pytest.mark.parametrize('source', [
    'Are they calibrated? Completely, without exception.',
    'They are. Clearly, we need more data.',
    'We have. Learning is difficult.',
    'They are. Learning helps everyone.',
    'We have. Learned behavior is a different topic.',
    'I mean it. Basically, all weights are learned.',
    'Yes. No. Okay. I am. You are.',
    'The model is mathematical. We should validate it.',
    'The inverse. Operator theory is another subject.',
    'We consider a mathematical. What does the expert mean?',
    'The result is. GOOD. The acronym is GOOD.',
    'Prof. Chen uses Eq. 3, e.g. for the 0.95 threshold.',
    'I mean S2, not S3, and I am not sure.',
    'We measured 95%. Not 0.95.',
    'We fit a signal. Model results appear later.',
    'We study a material. Model selection is important.',
    'We changed a variable. Measurement follows.',
])
def test_complete_answers_new_subjects_acronyms_and_abbreviations_stay_separate(source):
    assert repair_fragment_boundaries(source) == source


def test_symbol_case_numbers_negation_and_units_remain_exact():
    raw = ('We are. Not measuring S3 at 532 nm. The bias has. Already been measured '
           'as -0.95 with 1,000 samples, not 95%. ExampleNet and hpBRDF stay unchanged.')
    repaired = repair_fragment_boundaries(raw)
    for token in ('not measuring S3', '532 nm', '-0.95', '1,000', 'not 95%',
                  'ExampleNet', 'hpBRDF'):
        assert token in repaired
    assert lexical_tokens(repaired) == lexical_tokens(raw)


def test_missing_audio_is_not_filled_in_from_the_research_domain():
    raw = ('Much do we let the system learn end-to-end, right? '
           'The motion field applies a particular mathematical.')
    result = restore_utterance(raw)
    assert lexical_tokens(result.clean) == lexical_tokens(raw)
    assert not result.clean.startswith('How')
    assert 'operator' not in result.clean.casefold()


def test_repair_still_has_two_distinct_completed_questions():
    raw = ('The field is. Jointly, optimized. How do you validate it? '
           'Which assumptions fail?')
    result = restore_utterance(raw)
    assert result.sentences == (
        'The field is jointly optimized.', 'How do you validate it?', 'Which assumptions fail?')


@pytest.mark.parametrize('previous,incoming', [
    ('The weights are learned and are.', 'Completely, back-propagated.'),
    ('I, Mean, it.', 'Basically, all of the weights are learned.'),
    ('We are.', 'Not measuring S3.'),
    ('We use neural.', 'networks for this part.'),
    ('The field applies a particular mathematical.', 'model to the scene.'),
    ('We choose spectral', 'measurements with low noise.'),
])
def test_continuation_hint_needs_supported_cross_chunk_grammar(previous, incoming):
    assert sentence_continues(previous, incoming)


@pytest.mark.parametrize('previous,incoming', [
    ('Yes, they are.', 'Next question.'),
    ('Are they?', 'Completely sure.'),
    ('It is mathematical.', 'Models can vary.'),
    ('It seems highly mathematical.', 'Models are useful.'),
    ('We used something mathematical.', 'Models can vary.'),
    ('The result is complete.', 'What happens next?'),
    ('I mean it.', 'Basically, all parameters are learned.'),
    ('I, mean, it. The field is complete.', 'Next question.'),
    ('Prof. Chen measured 0.95.', 'Dr. Smith used 95%.'),
    ('We use spectral.', 'UNKNOWN is an acronym.'),
    ('', 'network'),
])
def test_continuation_hint_does_not_swallow_complete_sentences(previous, incoming):
    assert not sentence_continues(previous, incoming)


@pytest.mark.parametrize('tail', [
    'The weights are learned and are.',
    'The motion field applies a particular mathematical.',
    'We use neural',
    'I, mean, it.',
    'We use a differentiable spectral',
    'We are going to compare with',
])
def test_known_incomplete_caption_tails_do_not_invent_completion(tail):
    assert caption_tail_incomplete(tail)
    source = tail.rstrip('.')
    result = TranscriptRestorer(punctuator=lambda text: text + '.').restore(source)
    assert not result.clean.endswith(('.', '?', '!'))
    assert lexical_tokens(result.clean) == lexical_tokens(source)


@pytest.mark.parametrize('tail', [
    'Yes, they are.', 'I mean it.', 'The model is mathematical.', 'We use neural networks.',
    'We have measured S3.', 'How did you validate this?', 'The loss is spectral.',
])
def test_complete_caption_tails_are_not_inferred_incomplete(tail):
    assert not caption_tail_incomplete(tail)
    assert restore_utterance(tail).clean.endswith(('.', '?', '!'))


@pytest.mark.parametrize('previous,incoming', [
    ('I mean, it basically all of the weights in the motion synthesis network.',
     'And the encoder and decoder are learned and are completely, back propagated.'),
    ('All of the parameters in the first layer.', 'And the projection head are optimized together.'),
    ('Both the spectral channels.', 'And the polarization channels are measured independently.'),
    ('The optical encoder.', 'And the detector are calibrated together.'),
    ('Some of our coefficients in the forward model.', 'And the camera parameters have been fixed.'),
])
def test_coordinated_subject_is_kept_together_until_its_first_predicate(previous, incoming):
    source = previous + ' ' + incoming
    assert sentence_continues(previous, incoming)
    result = restore_utterance(source)
    assert '. And ' not in result.clean
    assert len(result.sentences) == 1
    assert lexical_tokens(result.clean) == lexical_tokens(source)
    assert result.raw == source


@pytest.mark.parametrize('previous,incoming', [
    ('All of the weights in the network are fixed.', 'And the encoder is learned.'),
    ('All of the weights changed.', 'And the decoder is still fixed.'),
    ('The model corrupts images.', 'And the encoder is fixed.'),
    ('The model predicts images.', 'And the decoder is learned.'),
    ('We trained the motion synthesis network.', 'And the encoder and decoder are learned.'),
    ('The optical encoder.', 'And does this generalize?'),
    ('Which optical encoder?', 'And the decoder is fixed.'),
    ('All of the parameters in the network.', 'And why are they learned?'),
    ('The optical encoder.', 'And this is the next question.'),
    ('The model uses the network.', 'And the decoder is fixed.'),
])
def test_and_does_not_join_completed_claims_questions_or_new_subjects(previous, incoming):
    assert not sentence_continues(previous, incoming)
    source = previous + ' ' + incoming
    assert repair_fragment_boundaries(source) == source


@pytest.mark.parametrize('previous,incoming', [
    ('The loss is purely spectral.', 'Measurements are independent.'),
    ('The model is entirely mathematical.', 'Models can vary.'),
    ('The design seems absolutely optical.', 'Systems have limitations.'),
    ('We have.', 'Increased costs require further study.'),
    ('We have.', 'Learned parameters change across datasets.'),
    ('They are.', 'Optimized networks perform better.'),
    ('We use neural.', 'Networks are different.'),
    ('We use a mathematical.', 'Model predictions require validation.'),
    ('We use a mathematical.', 'Models can vary.'),
])
def test_review_false_merges_keep_independent_finite_clauses(previous, incoming):
    source = previous + ' ' + incoming
    assert repair_fragment_boundaries(source) == source
    assert not sentence_continues(previous, incoming)


def test_ellipsis_nominal_restart_keeps_source_and_repairs_the_continuation():
    raw = ('I mean, basically, all of the weights... weights in the motion synthesis network. '
           'And the encoder and decoder are learned.')
    result = restore_utterance(raw)
    assert result.raw == raw
    assert result.clean.count('weights') == 1
    assert 'weights in the motion synthesis network and the encoder and decoder are learned' in result.clean
    assert 'disfluencies' in result.changes


@pytest.mark.parametrize('raw', [
    'I am not sure... not sure in this case.',
    'The value is 95%... 95% in that setting.',
    'I like it... it is good.',
    'Very... very strong.',
    'All of the S3... S3 in the measurement.',
])
def test_ambiguous_or_protected_ellipsis_repetitions_stay_visible(raw):
    result = restore_utterance(raw)
    assert lexical_tokens(result.clean) == lexical_tokens(raw)
    if not re.search(r'\d%?\.\.\.', raw):
        assert '…' in result.clean


def test_punctuation_model_cannot_convert_an_unresolved_ellipsis_into_a_period():
    raw = 'I am unsure... maybe the model is wrong'
    model = lambda _: 'I am unsure. Maybe the model is wrong.'
    result = TranscriptRestorer(punctuator=model).restore(raw)
    assert 'unsure…' in result.clean
    assert len(result.sentences) == 1


@pytest.mark.parametrize('raw', [
    'This is an end -to-end learning system.',
    'This is an end. -To-end learning system.',
    'This is an end-to -end learning system.',
])
def test_only_the_known_end_to_end_compound_is_rejoined(raw):
    result = restore_utterance(raw)
    assert result.raw == raw
    assert result.clean == 'This is an end-to-end learning system.'
    assert restore_utterance('The result is x - y and -0.95.').clean == 'The result is x - y and -0.95.'


@pytest.mark.parametrize('raw,expected', [
    ('And. Before we had the other model.', 'And before we had the other model.'),
    ('But. After the calibration we measured S0.', 'But after the calibration we measured S0.'),
    ('And. Before.', 'And. Before.'),
])
def test_stranded_conjunction_attaches_only_to_an_observed_time_clause(raw, expected):
    assert repair_fragment_boundaries(raw) == expected


def test_long_quantified_nominal_subject_is_open_but_short_nominal_answers_remain_valid():
    assert caption_tail_incomplete('All of the weights in the motion synthesis network.')
    assert not caption_tail_incomplete('The model.')

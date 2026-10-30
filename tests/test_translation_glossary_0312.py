"""Technical terminology is bounded; source meaning is never guessed by glossary."""
import pytest

from zoom_copilot.translation import protect_research_terms, restore_research_terms


@pytest.mark.parametrize(('source', 'expected'), [
    ('LIGHT STAGE', '光照采集装置'),
    ('light stages', '光照采集装置'),
    ('ground-truth albedo', '反照率参考真值'),
    ('ground truth', '参考真值'),
    ('anisotropic', '各向异性'),
    ('subsurface scattering', '次表面散射'),
    ('relighting', '重光照'),
    ('multi-illumination data set', '多光照数据集'),
    ('spectral-polarimetric ghost imaging', '光谱偏振鬼成像'),
    ('forward instrument matrix', '前向仪器矩阵'),
    ('inverse demodulation', '逆解调'),
    ('pseudoinverse', '伪逆'),
])
def test_exact_domain_phrase_protection(source, expected):
    protected, bindings = protect_research_terms(source)
    assert len(bindings) == 1
    assert bindings[0].source == source
    assert restore_research_terms(protected, bindings) == expected


def test_glossary_does_not_repair_suspected_asr_or_speaker_view():
    source = 'The multi elimination data set was not anisotropic.'
    protected, bindings = protect_research_terms(source)
    assert 'multi elimination data set' in protected
    assert 'was not' in protected
    assert len(bindings) == 1 and bindings[0].source == 'anisotropic'
    assert restore_research_terms(protected, bindings) == 'The multi elimination data set was not 各向异性.'


def test_glossary_never_changes_numbers_negation_or_uncertainty():
    source = 'I am not sure that 95% confidence proves identifiability; it may fail.'
    protected, bindings = protect_research_terms(source)
    assert 'not sure' in protected and '95%' in protected and 'may fail' in protected
    assert len(bindings) == 1 and bindings[0].source == 'identifiability'


def test_longest_phrases_preserve_operator_mapping_direction():
    source = ('The forward instrument matrix maps incident Stokes vectors to analyzer intensities; '
              'the inverse demodulation uses its pseudoinverse.')
    protected, bindings = protect_research_terms(source)
    assert [term.source for term in bindings] == [
        'forward instrument matrix', 'incident Stokes vectors', 'analyzer intensities',
        'inverse demodulation', 'pseudoinverse']
    assert protected.index(bindings[1].marker) < protected.index(' to ') < protected.index(bindings[2].marker)
    assert restore_research_terms(protected, bindings).endswith('uses its 伪逆.')


def test_non_domain_similar_words_are_untouched():
    source = 'The stage lights are on. Isotopic samples are labeled. The albedometer is nearby.'
    assert protect_research_terms(source) == (source, [])


def test_new_terms_keep_existing_integrity_rejection():
    _, bindings = protect_research_terms('Light stages capture ground truth albedo.')
    assert len(bindings) == 2
    assert restore_research_terms('[TERM100]采集。', bindings) is None
    assert restore_research_terms('[TERM100]采集[TERM101][TERM101]。', bindings) is None

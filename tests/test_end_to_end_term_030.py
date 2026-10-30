"""Protect the actual Whisper spaced-hyphen error without adding absent terms."""
import pytest

from zoom_copilot.translation import protect_research_terms, restore_research_terms


@pytest.mark.parametrize('term', [
    'end-to-end', 'end -to-end', 'end- to -end', 'end to end', 'END - TO - END',
])
def test_actual_asr_hyphen_spacing_preserves_end_to_end_meaning(term):
    source = f'One alternative would be a complete {term} learning approach.'
    protected, bindings = protect_research_terms(source)
    assert len(bindings) == 1
    assert bindings[0].source == term
    assert restore_research_terms('另一种方法是完整的[TERM100]学习。', bindings) == '另一种方法是完整的端到端学习。'
    assert protected == 'One alternative would be a complete [TERM100] learning approach.'


@pytest.mark.parametrize('source', [
    'We train a neural network to generate videos.',
    'Please read to the end of the section.',
    'The label is end to endosome.',
    'We do not know the ending.',
])
def test_end_to_end_term_never_injected_when_source_does_not_contain_it(source):
    assert protect_research_terms(source) == (source, [])

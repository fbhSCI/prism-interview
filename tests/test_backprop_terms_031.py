"""Preserve the meaning of observed neural-learning terms without mass edits."""
import pytest

from zoom_copilot.translation import (
    _correct_known_terms, protect_research_terms, restore_research_terms,
)


@pytest.mark.parametrize('term', [
    'back propagation', 'back-propagation', 'backpropagation',
    'back propagated', 'back-propagated', 'backpropagated',
    'back propagate', 'back-propagate', 'backpropagate',
])
def test_backprop_spelling_variants_are_protected_as_the_same_term(term):
    protected, bindings = protect_research_terms(f'The network uses {term}.')
    assert len(bindings) == 1
    assert bindings[0].source == term
    assert restore_research_terms(protected, bindings) == 'The network uses 反向传播.'


@pytest.mark.parametrize('term', ['back propagation', 'back-propagated', 'backpropagate'])
def test_known_backprop_mistranslation_correction_requires_the_source(term):
    assert _correct_known_terms(f'The network uses {term}.', '背面传播') == '反向传播'
    assert _correct_known_terms('The rear surface scatters light.', '背面传播') == '背面传播'
    assert _correct_known_terms(f'The network uses {term}.', '该网络进行学习。') == '该网络进行学习。'


@pytest.mark.parametrize('source', [
    'All of the weights in the motion synthesis network and the encoder and decoder are learned.',
    'The network weights are trained.',
    'The neural network weights are learned.',
    'The model weights are back-propagated.',
])
def test_learned_network_weight_mistranslation_is_corrected(source):
    assert _correct_known_terms(source, '所有重量都是被学习的。') == '所有权重都是被学习的。'


@pytest.mark.parametrize('source', [
    'The neural network predicts body weights in kilograms.',
    'The model learned body weights from the patients.',
    'The weights in the model predict body mass after training.',
    'The learned model predicts weights in kilograms.',
    'We weigh the network equipment and compare the weights.',
    'The weights are learned.',
    'The weights in the model are measured with scales.',
])
def test_physical_or_ambiguous_weights_are_not_changed(source):
    assert _correct_known_terms(source, '测量重量。') == '测量重量。'

"""Complete fact cards, content provenance and single-call answer integration."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from zoom_copilot.answers import AnswerService, build_prompt, parse_reply, _sent_evidence, estimate_prompt_tokens, LOCAL_CONTEXT_MARGIN, LOCAL_MAX_OUTPUT_TOKENS
from zoom_copilot.config import Preferences
from zoom_copilot.knowledge import KnowledgeBase, Passage
from zoom_copilot.research_facts import FACT_SOURCE, FactSelection, ResearchFactStore, rebind_fact_library


@pytest.fixture
def library(tmp_path):
    source = tmp_path / 'study.txt'
    text = ('The lumen method uses a calibrated optical measurement model. '
            'Against baseline Z, its mean error was lower on three held-out scenes. '
            'Calibration remained fixed. This does not establish performance in all environments.')
    source.write_text(text, encoding='utf8')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    kb = KnowledgeBase(tmp_path/'knowledge.json')
    kb.passages = [Passage(str(source), 'section 1', text, 'passage-one')]
    kb.document_names = [str(source)]
    kb.document_records = {str(source): {'sha256': digest}}
    card = {'id': 'fact-lumen-method', 'topic': 'lumen_reconstruction', 'kind': 'paper_method',
        'claim_en': 'The lumen method uses a calibrated optical measurement model.',
        'owner': 'The method described in study.txt', 'comparator': 'baseline Z',
        'conditions': ['Three held-out scenes.', 'Calibration remained fixed.'],
        'limitations': ['Not established for all environments.'], 'status': 'document_reported',
        'provenance': 'document_derived', 'sources': [{'passage_id': 'passage-one',
            'source': str(source), 'location': 'section 1', 'excerpt': text}]}
    data = {'schema_version': 1, 'documents': [{'source': str(source), 'sha256': digest}], 'facts': [card]}
    path = tmp_path / 'research_facts.json'
    path.write_text(json.dumps(data), encoding='utf8')
    return ResearchFactStore(path, kb), card, data, source


def reply(ids):
    return json.dumps({'answer_en': 'The result concerns a fixed calibration.',
        'answer_zh': '该结果适用于固定标定条件。', 'question_zh': '条件是什么？', 'citations': ids})


def test_card_selection_preserves_all_conditions_limitations_and_exact_sources(library):
    store, card, _, _ = library
    selection = store.select('How does lumen reconstruction work?')
    assert selection.cards == [card]
    prefs = Preferences(provider='deepseek', cloud_consent=True)
    system, user, sent, warnings = AnswerService(prefs, research_facts=store).prepare_prompt(
        'How does lumen reconstruction work?', [])
    assert json.loads(user)['research_fact_cards_untrusted'] == [card]
    result = parse_reply(reply([card['id'], 'passage-one']), sent)
    assert result.citations == [card['id'], 'passage-one']
    assert result.evidence[0]['source'] == FACT_SOURCE
    assert json.loads(result.evidence[0]['text']) == card
    assert result.evidence[1]['text'] == card['sources'][0]['excerpt']
    assert not warnings and 'not user-confirmed statements' in system


def test_changed_index_document_hash_invalidates_card(library):
    store, _, _, source = library
    assert store.select('lumen reconstruction').cards
    store.knowledge.document_records[str(source)]['sha256'] = 'a' * 64
    selection = store.select('lumen reconstruction')
    assert not selection.cards and len(selection.invalid_cards) == 1
    assert any('已停用' in warning for warning in selection.warnings)


def test_changed_real_file_invalidates_even_if_index_hash_is_stale(library):
    store, _, _, source = library
    assert store.select('lumen reconstruction').cards  # Populate stat/hash cache.
    source.write_text('A changed original with a different conclusion.', encoding='utf8')
    assert store.select('lumen reconstruction').cards == []
    assert store.invalid_cards


def test_excerpt_must_be_exact_substring_of_matching_indexed_passage(library):
    store, _, data, _ = library
    data['facts'][0]['sources'][0]['excerpt'] = 'A plausible but absent source statement.'
    store.path.write_text(json.dumps(data), encoding='utf8')
    selection = store.select('lumen reconstruction')
    assert not selection.cards and selection.warnings


def test_claim_provenance_cannot_be_relabelled_user_confirmed(library):
    store, _, data, _ = library
    data['facts'][0]['provenance'] = 'user_confirmed'
    store.path.write_text(json.dumps(data), encoding='utf8')
    assert not store.select('lumen reconstruction').cards
    assert any('读取或校验失败' in message for message in store.warnings)


def test_small_budget_omits_whole_card_never_only_its_limitations(library):
    store, card, _, _ = library
    assert not store.select('lumen reconstruction', maximum_chars=50).cards
    context = FactSelection(cards=[card])
    prefs = Preferences(provider='ollama', num_ctx=4096)
    system, user = build_prompt('Explain lumen reconstruction.', [], prefs, fact_selection=context)
    sent_cards = json.loads(user).get('research_fact_cards_untrusted', [])
    assert sent_cards in ([], [card])
    assert estimate_prompt_tokens(system + user) + LOCAL_CONTEXT_MARGIN + LOCAL_MAX_OUTPUT_TOKENS <= 4096
    accepted = parse_reply(reply([card['id']]), _sent_evidence(user, []))
    assert accepted.citations == ([card['id']] if sent_cards else [])


def test_shared_source_id_merges_only_actually_sent_snippets(library):
    store, card, _, source = library
    card['sources'][0]['excerpt'] = 'Calibration remained fixed.'
    visible = 'The lumen method uses a calibrated optical measurement model.'
    retrieval = [Passage(str(source), 'section 1', visible, 'passage-one')]
    _, user = build_prompt('Explain lumen reconstruction.', retrieval, Preferences(provider='deepseek'),
                           fact_selection=FactSelection(cards=[card]))
    result = parse_reply(reply(['passage-one']), _sent_evidence(user, retrieval))
    assert result.citations == ['passage-one']
    assert visible in result.evidence[0]['text'] and 'Calibration remained fixed.' in result.evidence[0]['text']
    assert 'three held-out scenes' not in result.evidence[0]['text']  # Never fill an unseen gap.
    collision = deepcopy(card)
    collision['sources'][0]['source'] = str(source.parent / 'another.txt')
    _, collided = build_prompt('Explain lumen reconstruction.', retrieval, Preferences(provider='deepseek'),
                               fact_selection=FactSelection(cards=[collision]))
    assert parse_reply(reply(['passage-one']), _sent_evidence(collided, retrieval)).citations == []


def test_overlap_suppressed_but_identity_and_unrelated_self_entry_preserved(library):
    store, card, _, _ = library
    stale = 'Personal contribution: lumen reconstruction with calibrated optical measurements in every environment. Source: study.txt.'
    profile = 'Identity: Example Candidate.\n' + stale + '\nI led an unrelated robotics project.'
    prefs = Preferences(provider='deepseek', cloud_consent=True, candidate_profile=profile)
    _, user, sent, warnings = AnswerService(prefs, research_facts=store).prepare_prompt(
        'Explain your lumen reconstruction background.', [])
    assert stale not in user
    assert json.loads(user)['research_fact_cards_untrusted'] == [card]
    assert any('主题重叠' in warning for warning in warnings)
    # Suppression does not edit the profile setting or promote extracted facts.
    assert prefs.candidate_profile == profile
    assert all(p.source != '候选人背景（用户提供，非论文原文）' or p.text != stale for p in sent)


def test_excerpt_sharing_disabled_never_sends_cards_or_loads_sources(library, monkeypatch):
    store, _, _, _ = library
    monkeypatch.setattr(store, 'select', lambda *args, **kwargs: pytest.fail('card selection not authorized'))
    prefs = Preferences(provider='deepseek', cloud_consent=True, share_evidence=False)
    _, user, sent, _ = AnswerService(prefs, research_facts=store).prepare_prompt('Explain lumen reconstruction.', [])
    assert 'research_fact_cards_untrusted' not in json.loads(user)
    assert not sent


def test_cloud_consent_still_required_and_fact_path_never_serialized(library, monkeypatch, tmp_path):
    store, _, _, _ = library
    monkeypatch.setattr(store, 'select', lambda *args, **kwargs: pytest.fail('no cloud consent'))
    prefs = Preferences(provider='deepseek', cloud_consent=False, prefer_prepared_answers=False)
    service = AnswerService(prefs, research_facts=store)
    with pytest.raises(ValueError, match='云端授权'):
        service.answer('Explain lumen reconstruction.', [])
    monkeypatch.setenv('PRISM_INTERVIEW_HOME', str(tmp_path/'prefs'))
    prefs.save()
    saved = (tmp_path/'prefs'/'preferences.json').read_text(encoding='utf8')
    assert 'research_facts' not in saved and 'lumen' not in saved
    assert Preferences.load().qwen_paid_usage_authorized is False


def test_single_model_request_uses_same_prepared_prompt(library, monkeypatch):
    store, card, _, _ = library
    prefs = Preferences(provider='deepseek', cloud_consent=True, prefer_prepared_answers=False)
    service = AnswerService(prefs, research_facts=store)
    expected_system, expected_user, _, _ = service.prepare_prompt('Explain lumen reconstruction.', [])
    calls = []
    class Client:
        model = last_model = 'offline-test-model'
        def __init__(self, preferences): pass
        def complete(self, system, user, on_delta, cancel_event):
            calls.append((system, user))
            return reply([card['id']])
    monkeypatch.setattr('zoom_copilot.providers.OpenAICompatibleClient', Client)
    result = service.answer('Explain lumen reconstruction.', [])
    assert calls == [(expected_system, expected_user)]
    assert result.citations == [card['id']]


def test_absent_library_keeps_legacy_prompt_without_error(tmp_path):
    store = ResearchFactStore(tmp_path/'missing.json', KnowledgeBase(tmp_path/'kb.json'))
    prefs = Preferences(provider='deepseek', candidate_profile='I developed lumen reconstruction.')
    service = AnswerService(prefs, research_facts=store)
    _, user, _, warnings = service.prepare_prompt('What did you develop?', [])
    assert 'research_fact_cards_untrusted' not in json.loads(user)
    assert 'I developed lumen reconstruction.' in user and not warnings


def test_subject_gate_rejects_generic_word_only_match(library):
    store, _, data, _ = library
    data['facts'][0]['claim_en'] += ' The paper reports a value from limited experimental data.'
    store.path.write_text(json.dumps(data), encoding='utf8')
    assert store.select('What phonon lifetime was achieved in superconducting qubits?').cards == []
    assert store.select('Explain the lumen reconstruction method.').cards


def test_overview_excludes_masters_unless_asked_and_does_not_fill_extra_topics(library):
    store, card, data, _ = library
    data['facts'] = []
    for index, topic in enumerate(('alpha_imaging', 'beta_sensing', 'gamma_calibration', 'masters_lumen')):
        item = deepcopy(card)
        item.update(id=f'card-{index}', topic=topic, kind='personal_contribution',
                    claim_en=f'The {topic} research has a documented contribution.')
        data['facts'].append(item)
    store.path.write_text(json.dumps(data), encoding='utf8')
    selected = store.select('What are your main PhD contributions?').cards
    assert {item['id'] for item in selected} == {'card-0', 'card-1', 'card-2'}
    assert [item['id'] for item in store.select('What was your masters lumen contribution?').cards] == ['card-3']


def test_exact_numeric_subject_not_lost_when_other_subquestion_has_more_matches(library):
    store, card, data, _ = library
    cost = deepcopy(card)
    cost.update(id='cost', topic='specific_network', claim_en='The network uses 0.15 million parameters and 0.70 GFLOPs.')
    gain = deepcopy(card)
    gain.update(id='gain', topic='specific_network', claim_en='The network improved PSNR by 2.80 dB under the held-out condition.')
    data['facts'] = [cost, gain]
    store.path.write_text(json.dumps(data), encoding='utf8')
    result = store.select('What do 0.15 million parameters, 0.70 GFLOPs and 2.80 dB describe?')
    assert {item['id'] for item in result.cards} == {'cost', 'gain'}


def portable_library(tmp_path):
    from zoom_copilot.knowledge import _passage
    old = tmp_path/'old'/'paper.txt'
    new = tmp_path/'new'/'paper.txt'
    old.parent.mkdir(); new.parent.mkdir()
    first, second = 'First full chunk. Shared clause.', 'Second full chunk. Shared clause.'
    old.write_text(first + second, encoding='utf8')
    new.write_bytes(old.read_bytes())
    digest = hashlib.sha256(old.read_bytes()).hexdigest()
    original = _passage(str(old), 'p1', first)
    kb = KnowledgeBase(tmp_path/'knowledge.json')
    kb.passages = [_passage(str(new), 'p1', first), _passage(str(new), 'p1', second)]
    kb.document_names = [str(new)]
    kb.document_records = {str(new): {'sha256': digest}}
    card = dict(id='portable-fact', topic='example_topic', kind='paper_method', claim_en='First claim.',
        owner='Document authors', comparator='', conditions=['First condition.'], limitations=['Scoped.'],
        status='document_reported', provenance='document_derived', sources=[dict(
            passage_id=original.passage_id, source=str(old), location='p1', excerpt='Shared clause.')])
    path = tmp_path/'research_facts.json'
    path.write_text(json.dumps(dict(schema_version=1, documents=[dict(source=str(old), sha256=digest)],
                                   facts=[card])), encoding='utf8')
    return path, kb, original, new


def test_portable_rebind_uses_original_full_chunk_id_to_disambiguate_overlap(tmp_path):
    path, kb, original, new = portable_library(tmp_path)
    report = rebind_fact_library(path, kb)
    assert report == dict(ok=True, valid_cards=1, documents=1, network_requests=0)
    source = json.loads(path.read_text(encoding='utf8'))['facts'][0]['sources'][0]
    assert source['source'] == str(new) and source['passage_id'] == kb.passages[0].passage_id
    assert source['original_source'] == original.source and source['original_passage_id'] == original.passage_id
    assert rebind_fact_library(path, kb)['ok']  # Repeated portable startup is safe.


def test_portable_rebind_failure_does_not_modify_library(tmp_path):
    path, kb, _, new = portable_library(tmp_path)
    before = path.read_bytes()
    new.write_text('Changed document content.', encoding='utf8')
    report = rebind_fact_library(path, kb)
    assert not report['ok'] and path.read_bytes() == before
    assert not path.with_name(path.name + '.rebind.tmp').exists()


def test_prepare_facts_cli_runs_without_starting_gui_or_requesting_model(tmp_path, monkeypatch):
    from zoom_copilot import app
    path, kb, _, _ = portable_library(tmp_path)
    kb._save()
    monkeypatch.setenv('PRISM_INTERVIEW_HOME', str(tmp_path))
    report = tmp_path/'report.json'
    monkeypatch.setattr('sys.argv', ['Prism', '--prepare-facts', str(path), '--facts-report', str(report)])
    assert app.main() == 0
    assert json.loads(report.read_text(encoding='utf8'))['valid_cards'] == 1


def test_only_our_exact_old_topic_default_migrates_to_fact_card_ownership():
    old = ('Personal achievements and publication status come only from candidate_confirmed_facts; '
           'literature alone does not establish authorship.')
    prefs = Preferences()
    prefs.research_profile = prefs.research_profile.split('Personal achievements')[0] + old
    system, _ = build_prompt('What did you develop?', [], prefs)
    assert old not in system and 'CV/script-backed fact card' in system
    custom = 'My own scope statement: ' + old
    prefs.research_profile = custom
    assert custom in build_prompt('What did you develop?', [], prefs)[0]

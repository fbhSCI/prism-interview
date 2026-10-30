"""Exercise the real service contract for mixed past/future questions."""
import json
import threading
import pytest

from zoom_copilot.answers import AnswerService, parse_mixed_reply, local_output_token_budget
from zoom_copilot.config import Preferences
from zoom_copilot.evidence_gap import EVIDENCE_GAP_EN
from zoom_copilot.evidence_gap import future_section_violation, mixed_evidence_answer_plan
from zoom_copilot.codex_client import AnswerCancelled

QUESTION = ('What did you personally measure in your completed enzyme trial, '
            'and how would you validate the next experiment?')

def test_missing_past_does_not_drop_separate_future_answer(tmp_path, monkeypatch):
    monkeypatch.setenv('PRISM_INTERVIEW_HOME', str(tmp_path))
    prefs = Preferences(provider='qwen', cloud_consent=True, prefer_prepared_answers=False)
    events, calls = [], []
    class Client:
        model = last_model = 'offline-model'
        def __init__(self, preferences): pass
        def complete(self, system, user, on_delta, cancel_event):
            data = json.loads(user)
            # Qwen JSON-object mode rejects a request lacking this keyword.
            assert 'json' in system.lower()
            calls.append(data)
            plan = data['mixed_answer_plan']
            assert [r['mode'] for r in plan] == ['local_evidence_gap', 'future_proposal']
            raw = json.dumps({'question_zh': '过去测了什么，下一步怎样验证？',
                'future_sections': [{'id': plan[1]['id'],
                    'answer_en': 'I would use an independent reference and repeated measurements to assess the proposed experiment.',
                    'answer_zh': '我会用独立参考和重复测量来评估拟议实验。', 'citations': []}]})
            on_delta(raw)
            assert all(kind == 'question_zh' for kind, _ in events)
            return raw
    monkeypatch.setattr('zoom_copilot.providers.OpenAICompatibleClient', Client)
    answer = AnswerService(prefs).answer(QUESTION, [], on_delta=lambda *x: events.append(x))
    assert len(calls) == 1
    assert EVIDENCE_GAP_EN in answer.answer_en
    assert 'independent reference' in answer.answer_en
    assert '独立参考' in answer.answer_zh
    assert answer.diagnostics['answer_completeness'] == 'all_subquestions_addressed_with_evidence_limit'
    assert answer.diagnostics['answer_origin'] == 'local_guard_with_structured_future'
    assert events[0] == ('question_zh', answer.question_zh)
    pairs = [json.loads(value) for kind, value in events if kind == 'answer_pair']
    assert pairs == answer.pairs
    assert '\n\n'.join(pair['en'] for pair in pairs) == answer.answer_en
    assert '\n\n'.join(pair['zh'] for pair in pairs) == answer.answer_zh

@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'unknown', 'past_en', 'past_zh', 'bad_citation'])
def test_invalid_future_sections_never_leak_personal_claims(kind):
    plan = [{'id':'Q1','mode':'local_evidence_gap'}, {'id':'Q2','mode':'future_proposal'}]
    row = {'id':'Q2','answer_en':'I would collect independent validation data.',
           'answer_zh':'我会采集独立验证数据。','citations':[]}
    if kind == 'past_en': row['answer_en'] += ' I achieved 99 percent accuracy.'
    if kind == 'past_zh': row['answer_zh'] += '我已经取得了99%的准确率。'
    if kind == 'bad_citation': row['citations'] = ['invented-source']
    rows = [] if kind == 'missing' else [row, dict(row)] if kind == 'duplicate' else [row]
    if kind == 'unknown': rows = [dict(row, id='Q9')]
    answer = parse_mixed_reply(json.dumps({'question_zh':'问题','future_sections':rows}), [], plan)
    assert answer.answer_en == EVIDENCE_GAP_EN
    assert answer.diagnostics['answer_completeness'] == 'partial'
    assert not answer.citations

def test_local_output_reservation_grows_with_bilingual_word_budget():
    assert local_output_token_budget(Preferences(max_answer_words=360)) > local_output_token_budget(Preferences(max_answer_words=80))

@pytest.mark.parametrize('tail', [
    'explain how you would validate the next experiment',
    'clarify how you would validate the next experiment',
    'tell us what you would test next',
    'walk us through how you would validate the next experiment',
])
def test_future_imperative_after_unsupported_history_is_not_lost(tail):
    question = 'What did you measure in your completed optical trial, and ' + tail + '?'
    plan = mixed_evidence_answer_plan(question, {})
    assert [item['mode'] for item in plan] == ['local_evidence_gap', 'future_proposal']

@pytest.mark.parametrize('claim', [
    'I did measure 99 percent accuracy.', 'Our prototype achieved 99 percent accuracy.',
    'Our previously tested instrument was successful.',
])
def test_future_section_rejects_alternate_completed_claim_grammar(claim):
    assert future_section_violation('I would validate it independently. ' + claim,
                                    '我会进行独立验证。') == 'personal_history_in_future_section'

"""Generic personal-evidence routing; entirely offline and domain-independent."""
import json
import threading

import pytest

from zoom_copilot.answers import AnswerService
from zoom_copilot.candidate_profile import candidate_fact_id
from zoom_copilot.codex_client import AnswerCancelled
from zoom_copilot.config import Preferences
from zoom_copilot.evidence_gap import (
    EVIDENCE_GAP_EN, EVIDENCE_GAP_ZH, personal_record_requests,
    personal_evidence_insufficient, has_unanswered_future_subquestion,
)


def card(topic='lumen_detector', **overrides):
    return {'id': 'a-personal-record', 'topic': topic, 'kind': 'personal_contribution',
            'claim_en': 'The supplied CV attributes development of a lumen detector to the candidate.',
            'owner': 'Candidate; explicitly attributed by the supplied CV',
            'status': 'document_reported', 'provenance': 'document_derived', **overrides}


def payload(*cards, text='', kind='fact'):
    entries = [{'id': candidate_fact_id(text), 'kind': kind, 'text': text}] if text else []
    return {'research_fact_cards_untrusted': list(cards), 'candidate_confirmed_facts': {
        'origin': 'explicit_user_candidate_profile', 'entries': entries}}


@pytest.mark.parametrize('question', [
    'What exact sensitivity did you already achieve in your completed lattice-clock experiments?',
    'Which parts of the cryogenic memory did you personally develop?',
    'Have you personally measured the recovery rate in the enzyme trials?',
    'Could you tell us what you personally developed for the sonar array?',
    'What was your role in the completed robotic grasping trial?',
    'What were your measured error values in the glacier survey?',
    '请介绍你曾经负责的机器人实验，具体做了什么？',
    '你亲自测得的催化剂产率是多少？',
])
def test_explicit_completed_personal_records_guard_without_support(question):
    assert personal_record_requests(question)
    assert personal_evidence_insufficient(question, payload())


@pytest.mark.parametrize('question', [
    'How would you design a lattice-clock experiment next year?',
    'What would you have done if you had built the sonar array?',
    'Which parts would you contribute if you joined our robotics project?',
    'What are your three main research contributions?',
    'How does a neural prior affect identifiability?',
    'Why does the enzyme reaction slow down?',
    'Because you developed the sonar array, how does diffraction limit resolution?',
    'What causes instability in your completed enzyme experiments?',
    '你未来打算怎样设计机器人的控制器？',
    '为什么散射会降低信噪比？',
])
def test_future_general_and_technical_premises_do_not_guard(question):
    assert not personal_record_requests(question)
    assert not personal_evidence_insufficient(question, payload())


def test_mixed_past_role_and_future_still_requires_personal_record():
    question = ('Which parts of the cryogenic memory did you personally develop, '
                'and what would you contribute if you joined our project?')
    assert len(personal_record_requests(question)) == 1
    assert personal_evidence_insufficient(question, payload())


@pytest.mark.parametrize('record', [
    card(kind='prior_work', status='reference_literature'),
    card(kind='personal_background'),
    card(kind='future_proposal', status='proposed_future'),
    card(status='proposed_future'),
    card(provenance='unverified'),
])
def test_reference_background_and_future_never_establish_completed_personal_work(record):
    assert personal_evidence_insufficient('What did you personally develop for the lumen detector?', payload(record))


def test_same_topic_cv_attributed_contribution_allows_normal_generation():
    assert not personal_evidence_insufficient('What did you personally develop for the lumen detector?', payload(card()))
    assert not personal_evidence_insufficient('What did you personally develop?', payload(card()))
    # This only routes, and must not claim that the specific metric was verified.
    assert not personal_evidence_insufficient('What error did you measure in your lumen detector?', payload(card()))


def test_unrelated_personal_record_cannot_fill_a_topic_gap():
    assert personal_evidence_insufficient('What yield did you measure in your enzyme experiments?', payload(card()))
    assert personal_evidence_insufficient(
        'Your lumen detector is interesting. What did you achieve in your enzyme experiments?', payload(card()))


@pytest.mark.parametrize('text,kind,supported', [
    ('I developed a lumen detector and measured its response.', 'fact', True),
    ('Personal contribution: lumen detector design and validation.', 'contribution', True),
    ('I plan to develop a lumen detector.', 'fact', False),
    ('Future: I developed skills relevant to lumen detectors.', 'future', False),
    ('Transfer: lumen detector design is a new application.', 'transfer', False),
    ('I have never developed a lumen detector.', 'fact', False),
    ('The paper authors developed a lumen detector.', 'fact', False),
])
def test_only_completed_user_confirmed_entries_support_personal_work(text, kind, supported):
    assert personal_evidence_insufficient('What did you personally develop for the lumen detector?',
                                         payload(text=text, kind=kind)) is not supported


def test_raw_paper_authorship_is_not_personal_attribution():
    data = payload()
    data['reference_passages_untrusted'] = [{'id': 'paper', 'excerpt': 'Candidate is an author of the lumen paper.'}]
    assert personal_evidence_insufficient('What did you personally develop for the lumen detector?', data)


@pytest.mark.parametrize('unsafe_en', [
    'I achieved 97 percent accuracy and personally built the entire device.',
    'I never did these experiments and none of my publications contain this result.',
    'The supplied evidence does not establish this personal result.',
])
def test_guard_buffers_all_deltas_replaces_claims_and_retains_translation(tmp_path, monkeypatch, unsafe_en):
    monkeypatch.setenv('PRISM_INTERVIEW_HOME', str(tmp_path))
    prefs = Preferences(provider='deepseek', cloud_consent=True, prefer_prepared_answers=False,
                        candidate_profile='')
    service = AnswerService(prefs)
    calls, events = [], []
    raw = json.dumps({'answer_en': unsafe_en, 'answer_zh': '不应先展示的回答。',
                      'question_zh': '你在酶实验中取得了什么结果？', 'citations': []}, ensure_ascii=False)

    class Client:
        model = last_model = 'offline-model'
        def __init__(self, preferences): pass
        def complete(self, system, user, on_delta, cancel_event):
            calls.append((system, user))
            for char in raw:
                on_delta(char)
                # Complete question translation may appear early, never the
                # unsupported claim or a later correction of a leaked claim.
                assert all(kind == 'question_zh' for kind, _ in events)
            return raw

    monkeypatch.setattr('zoom_copilot.providers.OpenAICompatibleClient', Client)
    question = 'What yield did you personally achieve in your completed enzyme experiments?'
    expected = service.prepare_prompt(question, [])[:2]
    result = service.answer(question, [], on_delta=lambda name, text: events.append((name, text)))
    assert calls == [expected]  # Exactly one unchanged prompt and model call.
    assert result.answer_en == EVIDENCE_GAP_EN and result.answer_zh == EVIDENCE_GAP_ZH
    assert result.question_zh == '你在酶实验中取得了什么结果？'
    assert 45 <= len(result.answer_en.split()) <= 60
    assert result.citations == result.evidence == result.sources == []
    assert result.diagnostics['guarded_evidence_gap'] is True
    assert result.diagnostics['answer_origin'] == 'local_evidence_gap_guard'
    assert unsafe_en not in json.dumps(result.diagnostics)
    assert not any('enzyme' in str(v) for v in result.diagnostics.values())
    assert any('不是模型已核实' in warning for warning in result.warnings)
    assert events[0] == ('question_zh', result.question_zh)
    pairs = [json.loads(text) for kind, text in events if kind == 'answer_pair']
    assert pairs == result.pairs
    assert pairs[0]['en'] == EVIDENCE_GAP_EN and pairs[0]['zh'] == EVIDENCE_GAP_ZH


def test_supported_personal_result_keeps_live_streaming(tmp_path, monkeypatch):
    monkeypatch.setenv('PRISM_INTERVIEW_HOME', str(tmp_path))
    prefs = Preferences(provider='deepseek', cloud_consent=True, prefer_prepared_answers=False,
                        candidate_profile='I developed a lumen detector.')
    service = AnswerService(prefs)
    events = []
    raw = json.dumps({'answer_en': 'I developed the lumen detector.', 'answer_zh': '我开发了该探测器。',
                      'question_zh': '你开发了什么？', 'citations': []})

    class Client:
        model = last_model = 'offline-model'
        def __init__(self, preferences): pass
        def complete(self, system, user, on_delta, cancel_event):
            on_delta(raw)
            assert events
            return raw

    monkeypatch.setattr('zoom_copilot.providers.OpenAICompatibleClient', Client)
    result = service.answer('What did you personally develop for the lumen detector?', [],
                            on_delta=lambda name, text: events.append((name, text)))
    assert not result.diagnostics.get('guarded_evidence_gap')
    assert result.answer_en == 'I developed the lumen detector.'


def test_guard_does_not_emit_after_cancel(tmp_path, monkeypatch):
    monkeypatch.setenv('PRISM_INTERVIEW_HOME', str(tmp_path))
    prefs = Preferences(provider='deepseek', cloud_consent=True, prefer_prepared_answers=False)
    cancel, events = threading.Event(), []

    class Client:
        model = last_model = 'offline-model'
        def __init__(self, preferences): pass
        def complete(self, system, user, on_delta, cancel_event):
            on_delta('{"answer_en":"I invented this result"')
            cancel_event.set()
            return '{}'

    monkeypatch.setattr('zoom_copilot.providers.OpenAICompatibleClient', Client)
    with pytest.raises(AnswerCancelled):
        AnswerService(prefs).answer('What did you measure in your enzyme experiments?', [],
                                   on_delta=lambda *args: events.append(args), cancel_event=cancel)
    assert not events


@pytest.mark.parametrize('question', [
    'What is the roughness RMSE in your completed real-material inverse-rendering experiments?',
    'What was the measured error in your completed benchmark?',
    'Please give the score from my completed field trial.',
    'Which result did you obtain in your previous experiment?',
    'What was your specific role in the published prototype?',
    'How large was the uncertainty in your last experiment?',
    'Can you give me the measured value from your completed experiment?',
    'Could you give me the measured value from your completed experiment?',
    'What was the QLX in your completed protein trial?',
    'Please provide the ZUF coefficient from your completed aquifer study.',
])
def test_nominal_completed_record_requests_accept_unknown_metric_vocabulary(question):
    assert personal_record_requests(question)
    assert personal_evidence_insufficient(question, payload())


@pytest.mark.parametrize('question', [
    'What does RMSE mean in your published paper?',
    'How is uncertainty calculated in a completed experiment?',
    'What value would you expect in your next experiment?',
    'What result did the cited authors obtain?',
    'Why can calibration drift bias the error reported by your method?',
    'Your published method uses four channels; how would you extend it in future?',
    'Is a low error in your completed benchmark sufficient evidence of robustness?',
    'What is the meaning of QLX in your published paper?',
    'What was the objective in your completed aquifer study?',
])
def test_nominal_guard_preserves_definitions_logic_and_future_questions(question):
    assert not personal_record_requests(question)
    assert not personal_evidence_insufficient(question, payload())


def test_unknown_metric_with_same_topic_personal_record_stays_normal():
    question = 'What was the QLX in your completed protein trial?'
    personal = card(topic='protein_trial', claim_en='The candidate completed a protein trial.')
    assert not personal_evidence_insufficient(question, payload(personal))
    assert personal_evidence_insufficient(question, payload(card()))


def test_mixed_future_question_reports_partial_completion_without_reusing_draft(tmp_path, monkeypatch):
    monkeypatch.setenv('PRISM_INTERVIEW_HOME', str(tmp_path))
    prefs = Preferences(provider='qwen', cloud_consent=True, prefer_prepared_answers=False)
    question = ('Which parts of the cryogenic memory did you personally develop, '
                'and what would you contribute if you joined our existing project?')
    assert has_unanswered_future_subquestion(question)
    assert not has_unanswered_future_subquestion('Could you describe what you personally developed?')
    events, calls = [], []

    class Client:
        model = last_model = 'offline-model'
        def __init__(self, preferences): pass
        def complete(self, system, user, on_delta, cancel_event):
            calls.append(user)
            raw = json.dumps({'answer_en': 'I built it. I would extend my invented prior experiment.',
                              'answer_zh': '未经支持的过去与未来断言。',
                              'question_zh': '过去负责什么，未来如何贡献？', 'citations': []})
            on_delta(raw)
            assert all(kind == 'question_zh' for kind, _ in events)
            return raw

    monkeypatch.setattr('zoom_copilot.providers.OpenAICompatibleClient', Client)
    result = AnswerService(prefs).answer(question, [], on_delta=lambda *args: events.append(args))
    assert len(calls) == 1
    assert result.answer_en == EVIDENCE_GAP_EN
    assert result.diagnostics['answer_completeness'] == 'partial'
    assert result.diagnostics['unanswered_subquestion_kind'] == 'future_or_proposal'
    assert any('尚未回答未来子问' in message for message in result.warnings)
    assert not any('invented prior' in str(event) for event in events)


@pytest.mark.parametrize('question', [
    'Could you give me the F-score from your already completed experiment, then suggest what we should test next?',
    'Please provide the QLX from your completed trial and then outline your plans for the next experiment.',
    'What was the measured value in your previous test then suggest what we should investigate next?',
    'What was the error in your completed study, after that recommend what we should measure next?',
    'What was the value in your completed experiment, under cold conditions, then describe your future plans?',
])
def test_future_followup_cannot_hide_a_completed_record_request(question):
    assert personal_record_requests(question)
    assert personal_evidence_insufficient(question, payload())
    assert has_unanswered_future_subquestion(question)


@pytest.mark.parametrize('question', [
    'Could you explain what F-score means, then suggest what we should test next?',
    'What value would you expect in a future experiment, then outline how you would measure it?',
    'What does the error mean in your completed study, after that recommend what we should measure next?',
    'Please describe the principle, under cold conditions, then outline your future plans.',
])
def test_mixed_future_and_conceptual_requests_still_do_not_guard(question):
    assert not personal_record_requests(question)
    assert not personal_evidence_insufficient(question, payload())

import json
from zoom_copilot.answers import build_prompt, _focused_excerpt
from zoom_copilot.config import Preferences
from zoom_copilot.knowledge import Passage
from zoom_copilot.evidence_gap import split_answer_subquestions, future_section_violation

def test_condition_is_attached_to_actual_request_not_a_fake_subquestion():
    question = 'Which wavelengths are proven optimal? If that is not established, how would you choose them?'
    parts = split_answer_subquestions(question)
    assert len(parts) == 2
    assert parts[1] == 'If that is not established. how would you choose them?'

def test_overlapping_chunks_do_not_spend_budget_repeating_first_topic():
    repeated = 'The acquisition sequence uses thirty correlated frames at six operating conditions.'
    special = 'Uncertainty in edge widths was assessed with a moving block resampling procedure on those frames.'
    text = repeated + ' ' + special + ' Other observations do not establish independent replication.'
    selected = _focused_excerpt(text, 'How many frames were acquired? What uncertainty method was applied?',
        160, excluded={repeated.casefold()})
    assert special in selected
    assert repeated not in selected

def test_distinct_documents_keep_independent_provenance_despite_same_sentence():
    passages = [Passage('a.pdf','p1','Reference targets were measured independently.','a'),
                Passage('b.pdf','p1','Reference targets were measured independently.','b')]
    _, raw = build_prompt('How were reference targets measured?', passages, Preferences())
    assert [p['id'] for p in json.loads(raw)['reference_passages_untrusted']] == ['a','b']

def test_prospective_chinese_before_clause_is_not_a_past_claim():
    assert not future_section_violation('I would validate before estimating the normals.',
                                       '我将在估计法线之前进行验证。')
    assert future_section_violation('I would validate the result. I achieved high accuracy.',
                                   '我会验证结果。我已经取得了高精度。')

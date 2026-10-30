"""Personal facts must survive the answer boundary without paper authorship guesses."""
from dataclasses import replace
import json

import pytest

from zoom_copilot.answers import build_prompt, estimate_prompt_tokens, LOCAL_MAX_OUTPUT_TOKENS, LOCAL_CONTEXT_MARGIN
from zoom_copilot.candidate_profile import candidate_context, personal_question_kind, retrieval_query, missing_personal_warning
from zoom_copilot.config import Preferences
from zoom_copilot.knowledge import Passage, read_document
from zoom_copilot.question_packet import QuestionContext


def prefs(profile='', **kwargs):
    result = Preferences(**kwargs)
    result.candidate_profile = profile
    return result


def test_own_contribution_is_separate_from_literature_and_gets_priority():
    profile = ('My PhD topic: spectral-polarimetric imaging.\n'
               'I performed fixed-reference calibration at six airflow levels. Source: my_cv.pdf page 1.\n'
               'I have not yet demonstrated material inverse rendering.')
    literature = [Passage('background.pdf', 'p. 3', 'We recovered albedo and roughness.', 'ref-1')]
    system, user = build_prompt('What were your main PhD contributions?', literature, prefs(profile))
    data = json.loads(user)
    assert data['candidate_confirmed_facts']['question_kind'] == 'personal_history'
    assert profile.splitlines()[1] in [entry['text'] for entry in data['candidate_confirmed_facts']['entries']]
    assert 'fixed-reference calibration' not in data['reference_passages_untrusted'][0]['excerpt']
    assert 'Literature does not establish personal authorship' in system
    assert 'Use a relevant confirmed personal fact first' in system
    assert data['reference_passages_untrusted'][0]['file'] == 'background.pdf'


def test_empty_profile_does_not_turn_paper_into_candidate_achievement():
    system, user = build_prompt('What did you personally develop?',
        [Passage('other_person.pdf', 'p. 1', 'The authors developed an underwater method.')], prefs())
    assert json.loads(user)['candidate_confirmed_facts']['personal_contribution_missing']
    assert 'do not invent' in system.lower()
    assert '候选人经历或贡献' in missing_personal_warning('What did you personally develop?')
    assert not missing_personal_warning('How could scattering be modeled?')


@pytest.mark.parametrize('question,kind', [
    ('What are your main PhD contributions?', 'personal_history'),
    ('Could you describe your doctoral research?', 'personal_history'),
    ('What did you personally implement?', 'personal_history'),
    ('你的博士工作的核心贡献是什么？', 'personal_history'),
    ('How does your PhD experience relate to inverse rendering?', 'research_bridge'),
    ('How would you estimate material roughness?', 'scientific'),
    ('How does the forward instrument matrix differ from demodulation?', 'scientific'),
])
def test_question_kinds(question,kind):
    assert personal_question_kind(question) == kind


def test_personal_lookup_expands_context_but_scientific_query_stays_exact():
    scientific = 'How does the Stokes matrix map the channels?'
    assert retrieval_query(scientific, 'I used underwater low-rank constraints.') == scientific
    personal = retrieval_query('What are your main contributions?',
        'My manuscript studies fixed no-flow calibration and airflow crosstalk.', 'Spectral-polarimetric ghost imaging')
    assert 'airflow crosstalk' in personal and 'Spectral-polarimetric ghost imaging' in personal


def test_candidate_selection_keeps_whole_entries_and_explicit_omission_count():
    profile = '\n'.join(['Name: Candidate.', 'Topic: calibration.',
                         'I developed a low-rank method for underwater recovery. Source: cv.pdf p1.',
                         'I led wind tunnel calibration at six airflow levels. Source: cv.pdf p2.'])
    result = candidate_context('What were your airflow experiments?', profile, maximum=150)
    assert 'I led wind tunnel' in ' '.join(result['facts'])
    assert result['omitted_entries'] > 0
    assert all(row in profile.splitlines() for row in result['facts'])


def test_profile_never_silently_truncates_oversize_input():
    with pytest.raises(ValueError,match='未截断'):
        build_prompt('What are your contributions?', [], prefs('x' * 12001))


def test_packet_speaker_attribution_retained_without_repeating_question_bodies():
    q = 'How would you quantitatively validate material uncertainty?'
    context=QuestionContext()
    context.record('source1',q,speaker_id='p',speaker_label='Professor',accepted=True)
    system,user=build_prompt(q, [], prefs(), interview_packet=context.build(q).to_prompt_data())
    packet=json.loads(user)['interview_input']
    row=packet['speaker_attributed_question_turns'][0]
    assert row['speaker_label']=='Professor' and row['text_in_complete_current_question']
    assert 'sentences' not in row
    assert 'source_policy' not in packet
    assert 'i=M s' not in system  # Irrelevant matrix instructions do not delay this question.


def test_domain_facts_are_supplied_by_documents_not_default_instructions():
    system,user=build_prompt('Explain calibration.',[],prefs())
    assert 'i=M s' not in system and 'GISC' not in system
    assert not json.loads(user)['reference_passages_untrusted']


def test_small_local_context_keeps_user_fact_and_source():
    profile='I established fixed no-flow calibration across six airflow levels. Source: cv.pdf p1.'
    system,user=build_prompt('What was your airflow contribution?',
        [Passage('manuscript.pdf','p. 5','The fixed no-flow matrix permits common-reference comparison.')],
        prefs(profile,provider='ollama',num_ctx=4096))
    data=json.loads(user)
    assert [entry['text'] for entry in data['candidate_confirmed_facts']['entries']]==[profile]
    assert data['reference_passages_untrusted']
    assert estimate_prompt_tokens(system+user)+LOCAL_MAX_OUTPUT_TOKENS+LOCAL_CONTEXT_MARGIN<=4096


def test_docx_script_citations_keep_slide_and_prepared_question_boundaries(tmp_path):
    docx=pytest.importorskip('docx')
    document=docx.Document()
    for line in ('Speaking script','Slide 19 Fixed reference inversion',
                 'We retrieved three linear Stokes components.',
                 'Slide 24 Future material identifiability',
                 'I would begin with known geometry.',
                 'Q1. How would you validate it?',
                 'I would use independent reference targets.'):
        document.add_paragraph(line)
    path=tmp_path/'script.docx';document.save(path)
    extracted=read_document(path)
    slide19=next(text for label,text in extracted if label.startswith('Slide 19'))
    assert 'three linear Stokes' in slide19 and 'known geometry' not in slide19
    q1=next(text for label,text in extracted if label.startswith('Q1.'))
    assert 'reference targets' in q1 and 'three linear Stokes' not in q1

"""Personal routing and evidence relevance without private runtime documents."""
import json
import pytest

from zoom_copilot.candidate_profile import candidate_context, personal_question_kind, retrieval_query
from zoom_copilot.knowledge import KnowledgeBase, _remove_contact_details


PROFILE = '\n'.join([
    'Identity: Example Candidate, Example University. Source: cv.pdf page 1.',
    'PhD topic: computational imaging. Source: cv.pdf page 1.',
    'Personal contribution — underwater reconstruction: constrained recovery with total variation and non-negativity. The 1.7 dB gain was an average over five measured scenes, not a universal gain. Source: cv.pdf page 1.',
    'Personal contribution — CompactNet: calibrated sensing for unseen analyzer channels. Manuscript under review, not accepted or published. Source: cv.pdf page 2.',
    'Personal contribution — airflow reliability: fixed no-flow calibration for S0, S1 and S2, not S3/full Stokes. Source: cv.pdf page 1.',
    "Master's work, distinct from PhD: EarlierNet was completed during the MSc. Source: cv.pdf page 1.",
    'Future postdoc proposal, not completed work: inverse rendering of material roughness, starting with known lighting and geometry. Source: letter.pdf page 2.',
    'Transferable strengths: calibration and validation; material inverse rendering is a new application, with no claimed material-recovery results. Source: letter.pdf page 2.',
])


@pytest.mark.parametrize('question',[
    'What are your three main PhD contributions?',
    'Please explain your two most important doctoral results.',
    'Have you already recovered material roughness?',
    'What is your measured material-roughness error?',
])
def test_flexible_personal_questions_are_not_generic_science(question):
    assert personal_question_kind(question)=='personal_history'


def test_doctoral_overview_reserves_core_contributions_not_masters_or_proposals():
    facts=candidate_context('What are your three main PhD contributions?',PROFILE)['facts']
    joined='\n'.join(facts)
    assert all(term in joined for term in ('underwater reconstruction','CompactNet','airflow reliability'))
    assert 'EarlierNet' not in joined and 'Future postdoc proposal' not in joined
    assert 'under review, not accepted or published' in joined
    assert 'not S3/full Stokes' in joined


def test_bridge_contains_doctoral_methods_and_explicit_future_scope():
    facts=candidate_context('How does your PhD experience relate to material inverse rendering?',PROFILE,maximum=2600)['facts']
    joined='\n'.join(facts)
    assert all(term in joined for term in ('underwater reconstruction','CompactNet','airflow reliability','not completed work','new application'))
    assert 'EarlierNet' not in joined


def test_narrow_question_keeps_qualifiers_without_unrelated_biography():
    facts=candidate_context('Have you already published CompactNet?',PROFILE)['facts']
    assert len(facts)==1 and 'under review, not accepted or published' in facts[0]
    assert 'Example University' not in facts[0]


def test_requested_master_work_is_not_dropped_or_relabelled():
    facts=candidate_context("Was EarlierNet part of your PhD or master's work?",PROFILE)['facts']
    assert any("Master's work, distinct from PhD" in row and 'EarlierNet' in row for row in facts)


def test_unrelated_science_does_not_fill_prompt_with_personal_facts():
    assert candidate_context('What is a quasar redshift?',PROFILE)['facts']==[]


def library(tmp_path, passages):
    path=tmp_path/'knowledge.json'
    path.write_text(json.dumps({'passages':[dict(source=f'{i}.pdf',location='p.1',text=text)
        for i,text in enumerate(passages)]}),encoding='utf8')
    return KnowledgeBase(path)


@pytest.mark.parametrize('question',[
    'What is the Shor algorithm complexity for factoring a 2048-bit RSA integer?',
    'Explain the Bellman algorithm for a Markov process.',
])
def test_unknown_named_subject_cannot_match_generic_algorithm_words(tmp_path,question):
    kb=library(tmp_path,['Our reconstruction algorithm reduces complexity using a calibrated sensing model.',
                         'The algorithm estimates spectral images with a compact network.'])
    assert kb.search(question,top_k=5)==[]


def test_generic_only_question_and_known_subject_still_retrieve(tmp_path):
    kb=library(tmp_path,['The algorithm complexity depends on the reconstruction model.',
                         'Stokes demodulation follows the instrument matrix calibration.'])
    assert kb.search('What is the algorithm complexity?')
    assert 'Stokes' in kb.search('How does Stokes demodulation work?')[0].text


def test_contact_filter_preserves_quantities_and_raw_index(tmp_path):
    original=('Candidate | candidate@example.edu | +86 156 1234 5678\n'
              'Google Scholar | LinkedIn\n'
              'Fixed calibration: 27 bands over 430–690 nm, 32 frames per condition.\n'
              'PSNR 27.47 to 29.16 dB; 4,800 observations; 0.285M parameters.')
    kb=library(tmp_path,[original])
    hit=kb.search('What calibration bands and PSNR were measured?')[0]
    assert '@' not in hit.text and '156 1234' not in hit.text
    assert all(value in hit.text for value in ('27 bands','430–690 nm','32 frames','27.47','29.16','4,800','0.285M'))
    assert kb.passages[0].text==original
    assert _remove_contact_details('A signed value +12345678901 counts is not a phone.')=='A signed value +12345678901 counts is not a phone.'


def test_personal_retrieval_expansion_does_not_repeat_cv_source_metadata():
    query=retrieval_query('What are your three main PhD contributions?',PROFILE,'Imaging')
    assert all(value in query for value in ('underwater reconstruction','CompactNet','airflow reliability'))
    assert 'cv.pdf' not in query and 'Example University' not in query


@pytest.mark.parametrize('question',[
    'What did you develop?', 'What have you built?',
    'What did you implement?', 'What did you design?',
])
def test_general_contribution_question_accepts_unlabelled_first_person_facts(question):
    profile='I developed an underwater recovery method. Source: cv.pdf.'
    result=candidate_context(question,profile)
    assert result['question_kind']=='personal_history'
    assert result['facts']==[profile]
    assert not result['personal_contribution_missing']


def test_specific_unlabelled_contribution_remains_subject_focused():
    underwater='I developed an underwater recovery method. Source: cv.pdf.'
    calibration='I implemented airflow calibration checks. Source: cv.pdf.'
    result=candidate_context('What did you implement for airflow calibration?',underwater+'\n'+calibration)
    assert result['facts']==[calibration]

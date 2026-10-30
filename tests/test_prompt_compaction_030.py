import json

from zoom_copilot.answers import build_prompt
from zoom_copilot.config import Preferences
from zoom_copilot.question_packet import QuestionContext


def test_old_default_disclaimer_does_not_contradict_present_candidate_facts():
    prefs = Preferences(candidate_profile='I developed an underwater recovery method. Source: cv.pdf.')
    prefix = prefs.research_profile.split('Personal achievements')[0]
    old = 'No specific personal publications, contributions or experiment values have been verified.'
    prefs.research_profile = prefix + old
    system, user = build_prompt('What did you develop?', [], prefs)
    assert old not in system
    assert 'underwater recovery method' in user
    # A user-authored caveat is not overwritten by this exact-default migration.
    prefs.research_profile = 'My own caveat: ' + old
    assert prefs.research_profile in build_prompt('What did you develop?', [], prefs)[0]


def test_prompt_omits_unselected_file_inventory_but_keeps_raw_and_speaker():
    raw = 'Can your four linear channels recover S3?'
    q = 'Can the four linear channels recover S3?'
    context = QuestionContext()
    context.record('q', raw, clean_en=q, speaker_id='remote', speaker_label='Interviewer', speaker_status='source', accepted=True)
    packet = context.build(q, raw_question=raw).to_prompt_data()
    packet['prepared_local_library']['documents'] = ['unselected-private-file.pdf']
    _, user = build_prompt(q, [], Preferences(), raw_question=raw, interview_packet=packet)
    data = json.loads(user)
    assert 'unselected-private-file.pdf' not in user
    assert data['recognized_english_before_cleanup'] == raw
    assert data['current_question'] == q
    assert data['interview_input']['speaker_attributed_question_turns'][0]['speaker_label'] == 'Interviewer'
    assert data['interview_task']['question_checklist']
    assert 'subquestions_to_cover' not in data['interview_input']
    assert packet['prepared_local_library']['documents'] == ['unselected-private-file.pdf']

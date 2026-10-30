"""Hidden question preparation is not an endpoint or a displayed answer.

Recorded streams are controller/confirmed-text evidence, not new ASR or model
latency measurements. The original endpoint and complete source remain intact.
"""
import json
from pathlib import Path
from public_fixtures import read_optional_recording

import pytest

from zoom_copilot.turns import confirmed_question_boundary, speculative_question_snapshot


@pytest.mark.parametrize('tail', [
    'this honestly',
    'this honestly so at this stage',
    'that frankly uh well at that point in the',
])
def test_tentative_snapshot_does_not_confirm_endpoint(tail):
    question='What additional measurements do you use?'
    text=question+' '+tail
    assert speculative_question_snapshot(text)==question
    assert confirmed_question_boundary(text) is None


@pytest.mark.parametrize('question', [
    'What additional measurements do you use under variable illumination?',
    'Why do you not use the measured illumination in this reconstruction',
    'I have a question about polarization. Do you calibrate the sensor separately?',
])
def test_snapshot_keeps_exact_negation_conditions_and_background(question):
    assert speculative_question_snapshot(question+' this honestly')==question


@pytest.mark.parametrize('text', [
    'Do you have any questions?',
    'Do you have any',
    'What additional measurements do you',
    'How would you interpret "this honestly so at this stage"?',
    'What additional measurements do you use if this honestly',
    'What additional measurements do you use with this honestly',
    'The system is end-to-end learned, right?',
    'I use a forward model.',
])
def test_no_snapshot_from_invitation_incomplete_or_quoted_restart(text):
    candidate=speculative_question_snapshot(text)
    if text.startswith('How would you interpret'):
        # The quoted phrase cannot truncate a real complete ask.
        assert candidate==text
    else:
        assert candidate==''


def test_same_questioner_qualifier_is_preserved_even_if_hidden_snapshot_was_started():
    question='What additional measurements do you use?'
    first=question+' This honestly so at this stage'
    snapshot=speculative_question_snapshot(first)
    assert snapshot==question
    revised=first+' I need quantitative results under variable illumination.'
    assert confirmed_question_boundary(revised) is None
    # A final silence snapshot contains all current source, so matching just
    # the speculative prefix is explicitly insufficient for result reuse.
    assert revised!=snapshot
    assert 'under variable illumination' in revised


@pytest.mark.parametrize('filename,prefix,expected_lead', [
    ('svsiggraph_full_timed_038.json', "I've got a question about", 7.188),
    ('tum_q2_timed_037_final4.json', 'Alex, one question', 2.453),
])
def test_recorded_question_preparation_can_overlap_confirmation_wait(filename,prefix,expected_lead):
    data=json.loads(read_optional_recording((Path(__file__).parent/'fixtures'/filename)))
    text='';started=None;sealed=None;snapshot=''
    for event in data['events']:
        if event['kind']!='final':
            continue
        fragment=event['text']
        if prefix in fragment:
            text=fragment[fragment.index(prefix):]
            started=None;snapshot=''
        elif text:
            text+=' '+fragment
        else:
            continue
        endpoint=confirmed_question_boundary(text)
        candidate=speculative_question_snapshot(text)
        if candidate and candidate!=snapshot:
            snapshot=candidate;started=event['replay_seconds']
        if endpoint is not None:
            sealed=text[:endpoint].strip()
            break
    assert sealed and snapshot==sealed
    assert event['replay_seconds']-started==pytest.approx(expected_lead,abs=.001)

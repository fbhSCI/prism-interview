"""Moderator invitations are captions; only actual interview asks auto-dispatch."""
from dataclasses import replace
import json
from pathlib import Path
from public_fixtures import read_optional_recording
from types import SimpleNamespace
import time

import pytest

from test_controller_v2 import controller
from zoom_copilot.question_packet import QuestionContext
from zoom_copilot.turns import explicit_question_focus, confirmed_question_boundary


def qa(c):
    c.pref.caption_speaker_mode = 'single_remote'
    c._listen_pref = replace(c.pref)
    c.change_interview_phase('qa')
    c.listening = True


def pause(c):
    c.assembler.last_voice = c.assembler.last_text = time.monotonic() - 2
    c.auto_answer()


@pytest.mark.parametrize('invitation', [
    'Any questions?', 'Any further questions from the audience?',
    'Do you have any questions?', 'Does anyone have a question?',
    'Are there any more questions?', 'Would anyone like to ask a question?',
    'Can I ask a question?', 'Could we take another question?',
    'Let us open the floor for questions.', 'We have time for one more question.',
    "I'll be happy to take any questions, thank you.",
    'Do we have questions?', 'Who has a question?',
    'And yes, I think Bjorn, you have a question?',
    'Could you repeat your question?',
])
def test_invitation_waits_through_pause_for_actual_question(controller, invitation):
    c = controller; qa(c)
    c.on_transcript('final', invitation)
    pause(c)
    assert not c.busy and not c._requests
    c.on_transcript('final', 'How would you validate the forward model under scattering?')
    c.on_transcript('final', 'Sure.')
    assert c.busy
    raw = c._question_raw_by_request[c._active]
    assert raw == 'How would you validate the forward model under scattering?'
    assert all(invitation != turn.raw_en for turn in c._question_packets[c._active].recent_context)
    assert invitation in ' '.join(row['raw_en'] for row in c._transcripts)


def test_invitation_and_actual_question_in_one_confirmed_crop(controller):
    c = controller; qa(c)
    c.on_transcript('final', 'Do you have any questions? How would you estimate uncertainty? Sure.')
    assert c.busy and not c._requests
    assert c._question_raw_by_request[c._active] == 'How would you estimate uncertainty?'


def test_fragmented_invitation_does_not_dispatch_or_pollute_actual_question(controller):
    c = controller; qa(c)
    for fragment in ('Do you have any', 'questions?'):
        c.on_transcript('final', fragment)
        pause(c)
        assert not c.busy
    c.on_transcript('final', 'How would you calibrate the instrument? Sure.')
    assert c._question_raw_by_request[c._active] == 'How would you calibrate the instrument?'


def test_unpunctuated_compliment_then_invitation_waits_for_actual_ask(controller):
    c = controller; qa(c)
    c.on_transcript('final', 'Thank you really amazing do we have questions?')
    pause(c)
    assert not c.busy and not c.assembler.text
    c.on_transcript('final', 'Which reconstruction approach would you recommend? Sure.')
    assert c.busy and c._question_raw_by_request[c._active].startswith('Which reconstruction')


@pytest.mark.parametrize('question', [
    'Do you have any evidence of improved reconstruction accuracy?',
    'Which scientific questions remain open in inverse rendering?',
    'What questions would your validation experiment answer?',
    'Can you explain the question of identifiability under scattering?',
])
def test_substantive_question_vocabulary_is_not_filtered(controller, question):
    c = controller; qa(c)
    c.on_transcript('final', question)
    pause(c)
    assert c.busy and c._question_raw_by_request[c._active] == question


def test_possible_invitation_prefix_becomes_substantive_evidence_question(controller):
    c = controller; qa(c)
    c.on_transcript('final', 'Do you have any')
    pause(c)
    assert not c.busy
    c.on_transcript('final', 'evidence that this approach reduces reconstruction error?')
    pause(c)
    assert c.busy
    assert 'any evidence that this approach' in c._question_raw_by_request[c._active]


def test_invitation_between_valid_questions_preserves_both_frozen_requests(controller):
    c = controller; qa(c)
    c.on_transcript('final', 'How would you calibrate the instrument? Sure.')
    first = c._active
    c.on_transcript('final', 'Any other questions?')
    pause(c)
    assert c._active == first and not c._cancel.is_set() and not c._requests
    c.on_transcript('final', 'How would you validate uncertainty? Sure.')
    assert len(c._requests) == 1 and not c._cancel.is_set()
    assert 'calibrate' in c._question_raw_by_request[first]
    assert c._question_raw_by_request[c._requests[0][0]] == 'How would you validate uncertainty?'


def test_invitation_metadata_has_no_actual_subquestions():
    packet = QuestionContext().build('Do you have any questions?')
    assert packet.interaction_kind == 'invitation'
    assert not packet.answer_question_en and not packet.subquestions
    assert packet.to_prompt_data()['answer_question_en'] == ''


def test_actual_question_preview_excludes_moderation_but_retains_scientific_background():
    text = ('Do you have any questions? Thank you for the talk. '
            'Assume the illumination and scattering are unknown. '
            'How would you estimate material reflectance?')
    packet = QuestionContext().build(text)
    assert packet.answer_question_en == 'How would you estimate material reflectance?'
    assert 'illumination and scattering are unknown' in packet.background_en
    assert packet.raw_question_en == text
    assert packet.subquestions == ('How would you estimate material reflectance?',)


def test_continuation_windows_coalesce_into_one_latest_automatic_revision(controller):
    c = controller; qa(c)
    c.on_transcript('final', 'How would you estimate material reflectance?')
    pause(c)
    first = c._active
    for fragment in ('Especially under unknown illumination',
                     'and strong scattering,',
                     'with calibrated reference targets.'):
        c.assembler.voice()
        c.on_transcript('final', fragment)
        c.auto_answer()
        assert not c._requests
    pause(c)
    assert c._active == first and c._cancel.is_set() and len(c._requests) == 1
    raw = c._question_raw_by_request[c._requests[0][0]]
    assert 'material reflectance' in raw and 'unknown illumination' in raw
    assert 'strong scattering' in raw and 'calibrated reference targets' in raw


def test_invitation_does_not_delete_or_restart_completed_answer(controller):
    c = controller; qa(c)
    c.on_transcript('final', 'How would you calibrate the instrument? Sure.')
    c.test_run_worker()
    original_answer = c.window._answer_en
    c.on_transcript('final', 'Do we have any more questions?')
    pause(c)
    assert not c.busy and not c._requests and len(c._answers) == 1
    assert c.window._answer_en == original_answer


def test_manual_answer_remains_available_for_invitation(controller):
    c = controller; qa(c)
    c.request_answer('Do you have any questions?')
    c.test_run_worker()
    assert len(c.service.calls) == 1


@pytest.mark.parametrize('fragment', [
    'The video.', 'The calibrated operator.', 'The forward model.',
    'The spectral measurements.', 'Our reconstructed images.', 'The matrix?',
])
def test_compact_question_keeps_adjacent_short_nominal_complement(fragment):
    question = 'Do you think we should estimate it directly?'
    source = 'The illumination is unknown. ' + question + ' ' + fragment
    packet = QuestionContext().build(source)
    assert packet.answer_question_en == question + ' ' + fragment
    assert packet.to_prompt_data()['answer_question_en'].endswith(fragment)
    assert packet.raw_question_en == source
    assert 'illumination is unknown' in packet.background_en


@pytest.mark.parametrize('following', [
    'Sure.', 'That is a good question.', 'The operator is calibrated.',
    'The model measured the result.', 'Under unknown illumination.',
    'The signal under strong scattering.', 'If the reference is unavailable.',
    'The operator failed.',
])
def test_compact_question_does_not_attach_answer_or_condition_as_nominal_object(following):
    question = 'Do you think we should estimate it directly?'
    focus, background = explicit_question_focus(question + ' ' + following)
    assert focus == question
    assert following in background


def test_nominal_complement_does_not_jump_over_intervening_background():
    question = 'How would you calibrate it?'
    focus, background = explicit_question_focus(question + ' Under unknown illumination. The operator.')
    assert focus == question
    assert 'Under unknown illumination. The operator.' == background


def test_distinct_followup_remains_a_distinct_actual_question():
    first = 'How would you calibrate it?'
    second = 'How would you validate the result?'
    focus, background = explicit_question_focus(first + ' The operator. ' + second)
    assert focus == first + ' The operator. ' + second
    assert not background


@pytest.mark.parametrize('second', [
    'Can these channels recover S3?',
    'Could a learned operator generalize under scattering?',
    'Will spectral measurements resolve the ambiguity?',
    'Might polarimetric sensing help identify reflectance?',
    'Has M been calibrated independently?',
])
def test_actual_auxiliary_subquestions_with_nonpersonal_subjects_remain_in_focus(second):
    first = 'How would you calibrate M?'
    raw = 'We have four linear analyzer channels. ' + first + ' ' + second
    packet = QuestionContext().build(raw, raw_question=raw)
    assert packet.answer_question_en == first + ' ' + second
    assert packet.subquestions == (first, second)
    assert packet.raw_question_en == raw
    assert confirmed_question_boundary(second + ' Sure.') == len(second) + 1


def test_missing_stop_after_moderator_gratitude_still_seals_actual_ask(controller):
    c = controller; qa(c)
    for fragment in ("Yes, I'm just going to", 'forward YouTube questions.',
                     'so Juan', 'Rauwil Padrón Grif asks,',
                     'first thank you for the fantastic talk', 'Which do you think',
                     'is the direction more promising?',
                     'focusing on improving the existing reconstruction',
                     'pipelines or fixing', 'them with neural rendering.'):
        c.assembler.voice()
        c.on_transcript('final', fragment)
    pause(c)
    first = c._active
    assert first and c.busy
    raw = c._question_raw_by_request[first]
    assert 'forward YouTube questions' not in raw
    assert 'existing reconstruction' in raw and 'neural rendering' in raw
    c.on_transcript('final', "Oh, that's that's a great question. I")
    assert c._semantic_response
    assert not c._cancel.is_set() and not c._requests
    c.on_transcript('final', 'actually had not thought of it. I was late to the neural network game.')
    pause(c)
    c.test_run_worker()
    assert len(c.service.calls) == 1
    assert 'great question' not in c.service.calls[0]['raw_question']


def test_unpunctuated_reported_direct_ask_is_visible_without_rewriting_source():
    source = ('so Juan Rauwil Padrón Grif asks, first thank you for the fantastic talk '
              'Which do you think is the direction more promising? '
              'focusing on improving the existing reconstruction pipelines or fixing them with neural rendering.')
    packet = QuestionContext().build(source)
    assert packet.answer_question_en.startswith('Which do you think')
    assert packet.raw_question_en == source
    assert 'existing reconstruction pipelines' in packet.question_en
    assert confirmed_question_boundary(source + " Oh, that's that's a great question. I") == len(source) + 1


@pytest.mark.parametrize('source', [
    'Our method asks which measurements should be acquired first.',
    'The experiment asks whether reconstruction is identifiable.',
    'I ask the network to predict which pixels move.',
    'I would like to know which operator we should calibrate.',
])
def test_reported_question_detection_never_discards_scientific_or_indirect_question(source):
    packet = QuestionContext().build(source)
    assert packet.raw_question_en == source and packet.question_en == source
    assert ' '.join((packet.answer_question_en, packet.background_en)).strip() == source


@pytest.mark.parametrize('complete_immediately', [False, True])
@pytest.mark.parametrize('fixture_name', ['tum_invitation_timed_036.json', 'tum_handoff_smoke_timed_036.json'])
def test_actual_invitation_q1_and_timer_replay_keeps_one_request_without_reply_pollution(controller, complete_immediately, fixture_name):
    c = controller; qa(c)
    fixture = json.loads(read_optional_recording((Path(__file__).parent / 'fixtures' /
                          fixture_name)))
    c.speech = SimpleNamespace(pending=0, close=lambda: None)
    for event in fixture['events']:
        if event['kind'] == 'timer':
            now = time.monotonic()
            c.assembler.last_voice = now - event['voice_silence_ms'] / 1000
            c.assembler.last_text = now - event['postcommit_ms'] / 1000
            c.speech.pending = event['asr_pending']
            c._partial_id = 'recorded-unconfirmed-tail' if event['partial'] else None
            c.auto_answer()
        else:
            c.assembler.voice()
            c.on_transcript(event['kind'], event['text'], event['context'])
        if event['video_seconds'] < 110:
            assert not c.busy and not c._requests and not c.service.calls
        if complete_immediately and c.busy:
            c.test_run_worker(); c._next_answer()
    c.speech = None
    if not complete_immediately:
        assert c.busy and not c._cancel.is_set() and not c._requests
        c.test_run_worker()
    assert len(c.service.calls) == 1
    call = c.service.calls[0]
    raw = call['raw_question']
    assert 'Which do you think is the direction more promising?' in raw
    assert 'existing reconstruction pipelines' in raw and 'neural rendering' in raw
    assert 'forward YouTube questions' not in raw
    assert 'great question' not in raw and 'late to the neural network game' not in raw
    assert call['interview_packet']['answer_question_en'].startswith('Which do you think')
    assert c._semantic_response


@pytest.mark.parametrize('handoff', [
    "I just gonna forward YouTube questions.",
    "I'm gonna read the chat questions.",
    'We are just going to relay audience questions.',
])
def test_handoff_rejection_does_not_poison_new_cross_window_actual_ask(controller, handoff):
    c = controller; qa(c)
    c.on_transcript('final', handoff)
    c.on_transcript('rejected', '')
    assert c._question_uncertain
    for part in ('Juan, Raúl, Padrón,', 'Griff asks, first, thank you for',
                 'the fantastic talk. Which'):
        c.on_transcript('final', part)
    assert not c._question_uncertain
    assert c.assembler.text == 'Which'
    c.on_transcript('final', 'do you think is the more promising reconstruction direction? Sure.')
    assert c.busy
    assert c._question_raw_by_request[c._active].startswith('Which do you think')


@pytest.mark.parametrize('start', [
    'Which', 'How would you calibrate',
    'Assume unknown illumination and strong scattering.',
])
def test_rejection_after_actual_question_or_scientific_setup_still_blocks(controller, start):
    c = controller; qa(c)
    c.on_transcript('final', "I'm going to read the chat questions.")
    c.on_transcript('final', start)
    c.on_transcript('rejected', '')
    c.on_transcript('final', 'How would you estimate the sensing operator? Sure.')
    pause(c)
    assert c._question_uncertain and not c.busy and not c._requests


def test_invitation_to_new_question_preserves_scientific_background(controller):
    c = controller; qa(c)
    c.on_transcript('final', 'Any questions?')
    setup = 'Assume the illumination is unknown and the target is moving.'
    c.on_transcript('final', setup)
    c.on_transcript('final', 'How would you estimate material reflectance? Sure.')
    assert c.busy
    assert c._question_raw_by_request[c._active].startswith(setup)

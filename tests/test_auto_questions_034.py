"""Real interview request/continuation cases; local Qt stubs, no model or API."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from test_controller_v2 import controller
from zoom_copilot.interview_context import is_interview_request, question_parts
from zoom_copilot.question_packet import QuestionContext
from zoom_copilot.knowledge import Passage
from zoom_copilot.turns import QuestionAssembler


def qa(c):
    c.pref.caption_speaker_mode='single_remote'
    c._listen_pref=replace(c.pref)
    c.change_interview_phase('qa')
    c.listening=True


def finish_pause(c):
    import time
    c.assembler.last_voice=c.assembler.last_text=time.monotonic()-2
    c.auto_answer()


@pytest.mark.parametrize('question', [
    'I would be interested in your research limitations and validation plans.',
    'Talk us through your calibration procedure.',
    'Please walk us through the acquisition and reconstruction pipeline.',
    "I'd like to understand your main doctoral contribution.",
])
def test_polite_requests_without_question_mark_automatically_dispatch(controller, question):
    c=controller;qa(c)
    c.on_transcript('final',question)
    finish_pause(c)
    assert c.busy
    c.test_run_worker()
    assert len(c.service.calls)==1
    assert question.strip('.') in c.service.calls[0]['raw_question']


def test_plain_research_background_is_not_automatically_answered(controller):
    c=controller;qa(c)
    text='I am interested in inverse rendering and spectral acquisition.'
    c.on_transcript('final',text)
    assert not is_interview_request(text)
    # Keep it as setup at an ordinary pause, rather than mark it as answered.
    import time
    c.assembler.last_voice=c.assembler.last_text=time.monotonic()-2
    c.auto_answer()
    assert not c.busy and c.assembler.text==text


@pytest.mark.parametrize('tail', ['What is?', 'Where are?', 'Could you?', 'Do you?'])
def test_open_interrogative_waits_for_missing_content(controller, tail):
    c=controller;qa(c)
    c.on_transcript('final',tail)
    finish_pause(c)
    assert c.assembler.incomplete and not c.busy
    c.on_transcript('final','your main research contribution?')
    finish_pause(c)
    assert c.busy
    assert 'your main research contribution' in c.assembler.text


def test_noninverted_and_imperative_subquestions_get_distinct_checklist_entries():
    source=('How would you estimate reflectance and what assumptions are required '
            'and explain how you would measure uncertainty?')
    parts=question_parts(source)
    assert parts==['How would you estimate reflectance', 'what assumptions are required',
                   'explain how you would measure uncertainty?']
    packet=QuestionContext().build(source)
    assert packet.question_en==source
    assert packet.subquestions==tuple(parts)
    assert packet.to_prompt_data()['subquestions_to_cover']==parts


def test_coordinated_how_and_why_is_not_an_orphan_question():
    assert question_parts('How and why does scattering change polarization?')==[
        'How and why does scattering change polarization?']
    assert question_parts('What do you know about how and why this occurs?')==[
        'What do you know about how and why this occurs?']


def test_more_than_eight_actual_asks_are_not_silently_truncated():
    asks=[f'How would you test condition {i}?' for i in range(1,10)]
    packet=QuestionContext().build(' '.join(asks))
    assert packet.subquestions==tuple(asks)
    assert packet.to_prompt_data(max_chars=8000)['subquestions_to_cover']==asks


def test_explicit_different_question_does_not_repeat_previous_ask(controller):
    c=controller;qa(c)
    c.on_transcript('final','How do you calibrate the instrument?')
    finish_pause(c)
    first=c._active
    c.on_transcript('final','Now a different question. How would you evaluate uncertainty?')
    assert 'calibrate' not in c.assembler.text
    finish_pause(c)
    # An explicit next-question boundary freezes the prior complete ask.
    assert c._active==first and not c._cancel.is_set()
    assert len(c._requests)==1
    assert 'uncertainty' in c._requests[0][1] and 'calibrate' not in c._requests[0][1]
    packet=c._question_packets[c._requests[0][0]]
    assert all('calibrate' not in turn.raw_en for turn in packet.turns)
    assert any('calibrate' in turn.raw_en for turn in packet.recent_context)


def test_supplement_after_pause_keeps_original_ask_and_replaces_obsolete_answer(controller):
    c=controller;qa(c)
    c.on_transcript('final','How would you estimate material reflectance?')
    finish_pause(c)
    prefix=c.assembler.text
    c.on_transcript('final','Especially under unknown illumination and strong scattering.')
    assert c.assembler.text.startswith(prefix)
    finish_pause(c)
    assert c._cancel.is_set() and len(c._requests)==1
    assert 'material reflectance' in c._requests[0][1] and 'strong scattering' in c._requests[0][1]


@pytest.mark.parametrize('ack', ['Okay.', 'Right.', 'I see.', 'Got it.', 'That makes sense.'])
def test_pure_ack_does_not_cancel_or_restart_inflight_answer(controller, ack):
    c=controller;qa(c)
    c.on_transcript('final','How would you validate the forward model?')
    finish_pause(c)
    first,version=c._active,c.assembler.version
    c.on_transcript('final',ack)
    finish_pause(c)
    assert c._active==first and not c._cancel.is_set() and not c._requests
    assert c.assembler.version==version
    assert ack in c._transcripts[-1]['raw_en']
    c.test_run_worker()
    assert len(c.service.calls)==1


def test_ack_with_new_question_remains_actionable(controller):
    c=controller;qa(c)
    text='Okay, but how would you measure uncertainty?'
    c.on_transcript('final',text)
    finish_pause(c)
    assert c.busy and text in c._question_original


def test_evidence_budget_reserves_each_distinct_scientific_subquestion(controller):
    c=controller
    asks=('How would you acquire spectra?','Which material properties are observable?',
          'How would you measure runtime?','What numerical uncertainty metric would you report?')
    domains=[Passage(f'{name}.pdf','p1',text,name) for name,text in (
        ('spectra','Spectral acquisition uses wavelength encoding.'),
        ('material','Material BRDF parameters require controlled reflectance measurements.'),
        ('runtime','Timing separates acquisition from reconstruction.'),
        ('uncertainty','Uncertainty intervals are calibrated with held-out residuals.'))]
    distractors=[Passage('context.pdf',f'p{i}',f'Optical background only {i}',f'bg{i}') for i in range(5)]
    lookup=dict(zip(asks,[[p] for p in domains]))
    c.preparation.search=lambda kb,q,top_k:lookup.get(q,distractors)
    hits=c._retrieve_question(' '.join(asks),SimpleNamespace(subquestions=('UI grouped paragraph',)),5)
    assert hits[:4]==domains
    assert len(hits)==5
    assert c.last_retrieval_coverage['covered_subquestion_indices']==[1,2,3,4]
    assert c.last_retrieval_coverage['uncovered_subquestion_indices']==[]


def test_missing_or_budget_excluded_subquestion_is_recorded(controller):
    c=controller
    asks=('How acquire spectra?','Which material?','What runtime?','Which missing experiment?')
    documents=[Passage(f'{i}.pdf','p1',f'Distinct evidence {i}',str(i)) for i in range(3)]
    lookup=dict(zip(asks,[[p] for p in documents]+[[]]))
    c.preparation.search=lambda kb,q,top_k:lookup.get(q,[])
    hits=c._retrieve_question(' '.join(asks),SimpleNamespace(subquestions=('UI grouped paragraph',)),2)
    coverage=c.last_retrieval_coverage
    assert hits==documents[:2]
    assert coverage['uncovered_subquestion_indices']==[3,4]
    assert coverage['no_hit_subquestion_indices']==[4]
    assert coverage['selected_passage_count']==2
    assert all('evidence' not in str(value) for value in coverage.values())


def test_missing_evidence_warning_and_coverage_reach_model_packet(controller):
    c=controller;qa(c)
    c.on_transcript('final','How do you calibrate the system and what assumptions are required?')
    finish_pause(c)
    c.test_run_worker()
    data=c.service.calls[0]['interview_packet']
    assert data['retrieval_coverage']['uncovered_subquestion_indices']==[1,2]
    assert c._answers[-1]['result'].diagnostics['retrieval_coverage']['subquestion_count']==2
    assert any('未分配到检索资料' in warning for warning in c._answers[-1]['result'].warnings)


def test_question_mark_confirmation_is_not_silently_removed():
    from zoom_copilot.app import CopilotController
    assert not CopilotController._acknowledgment_only('Right?')
    assert not CopilotController._acknowledgment_only('Okay. What did you measure?')


@pytest.mark.parametrize('utterance', [
    'Walk us through the calibration procedure.',
    'Please talk us through the sensing operator.',
    'Discuss your validation strategy.',
    'Clarify the assumptions on illumination.',
    'I would like to understand your contribution.',
    "I'd like to hear your validation plan.",
])
def test_addressed_request_recovers_after_candidate_answer(controller, utterance):
    c=controller
    c.change_interview_phase('qa');c.listening=True
    c.on_transcript('final','How did you calibrate the instrument?',
                    {'speaker_id':'prof','speaker_label':'Professor','turn_id':'first'})
    c.on_transcript('final','I used controlled reference targets for calibration.',
                    {'speaker_id':'candidate','speaker_label':'Candidate','turn_id':'answer'})
    assert c.intent._response_active
    c.on_transcript('final',utterance,
                    {'speaker_id':'prof','speaker_label':'Professor','turn_id':'second'})
    assert not c.intent._response_active
    assert c.assembler.text==utterance
    finish_pause(c)
    assert not c._cancel.is_set() and len(c._requests)==1
    assert c._requests[0][1]==utterance


def test_who_request_is_recognized_without_asr_question_mark():
    question='Who performed the instrument calibration.'
    assert is_interview_request(question)
    assert question_parts(question)==[question]


def test_confirmed_continuation_immediately_invalidates_old_stream_before_endpoint(controller):
    from zoom_copilot.answers import BilingualAnswer
    c=controller;qa(c)
    c.on_transcript('final','How would you estimate material reflectance?')
    finish_pause(c)
    old=c._active
    # The stable reader publishes complete sentences, not incomplete tokens.
    c.post('answer_delta',(old,'answer_en','Old suggestion.'))
    c.pump()
    assert c.window.answer_english.text()=='Old suggestion.'
    # No endpoint/auto_answer call here: the old result must already be stale.
    c.on_transcript('final','Especially with unknown illumination and strong scattering.')
    assert c._cancel.is_set() and c.window._answer_en==''
    c.post('answer_delta',(old,'answer_en','LATE DELTA'))
    c.post('answer',(old,'Old question',BilingualAnswer('', 'LATE RESULT', '', [])))
    c.pump()
    assert not c._answers and not c._history
    assert 'LATE' not in c.window._answer_en
    assert 'reflectance' in c.assembler.text and 'strong scattering' in c.assembler.text


def test_new_continuation_removes_obsolete_automatic_queue_before_worker_finishes(controller):
    c=controller;qa(c)
    c.on_transcript('final','How would you estimate reflectance?')
    finish_pause(c)
    c.on_transcript('final','Especially with unknown illumination.')
    finish_pause(c)
    assert len(c._requests)==1
    obsolete=c._requests[0][0]
    c.on_transcript('final','And compare the result against controlled reference targets.')
    assert not c._requests
    assert obsolete not in c._question_packets and obsolete not in c._automatic_ids
    assert obsolete not in c._answer_request_timing
    c.test_run_worker();c.test_run_callbacks()
    assert not c.busy and not c._answers
    finish_pause(c)
    c.test_run_worker()
    assert 'unknown illumination' in c.service.calls[-1]['question']
    assert 'controlled reference targets' in c.service.calls[-1]['question']
    assert len(c._answers)==1


def test_explicit_next_question_cancels_automatic_but_not_manual_snapshot(controller):
    c=controller;qa(c)
    c.on_transcript('final','How would you validate reconstruction?')
    finish_pause(c)
    c.new_question()
    assert c._cancel.is_set()
    c.test_run_worker();c.test_run_callbacks()
    c.request_answer('Manually selected question?')
    c.new_question()
    assert not c._cancel.is_set()


def test_complete_request_endpoint_has_no_extra_sentence_or_translation_wait():
    assembler=QuestionAssembler(silence_seconds=1.0)
    assembler.voice(at=100.0)
    assembler.add('Please describe your validation procedure.',at=100.4)
    assert not assembler.ready(now=100.9)
    # Default endpoint is 1 s from last voice, then at most one 250 ms timer
    # tick. A late final also requires 180 ms of stabilization. ASR/model time
    # is separate; this test measures only the deterministic controller gate.
    assert assembler.ready(now=101.0)
    assert not assembler.ready(now=101.0,pending=1)
    assembler.consume(retain=True)
    assert not assembler.ready(now=102.0)


def test_late_confirmed_final_adds_only_stabilization_not_another_silence_window():
    assembler=QuestionAssembler(silence_seconds=1.0)
    assembler.voice(at=100.0)
    assembler.add('How would you validate the reconstruction?',at=102.0)
    assert not assembler.ready(now=102.1)
    assert assembler.ready(now=102.2)


def test_two_questions_across_short_pause_preserve_both_in_single_automatic_request(controller):
    c=controller;qa(c)
    c.on_transcript('final','How would you calibrate the instrument?')
    c.assembler.voice()
    c.auto_answer()
    assert not c.busy
    c.on_transcript('final','And how would you validate uncertainty under scattering?')
    finish_pause(c)
    c.test_run_worker()
    call=c.service.calls[-1]
    assert 'calibrate the instrument' in call['raw_question']
    assert 'validate uncertainty' in call['raw_question']
    assert len(call['interview_packet']['subquestions_to_cover'])==2

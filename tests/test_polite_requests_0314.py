"""Confirmed-text request routing, not new ASR accuracy or audio measurements.

The screenshot crops below are copied as displayed, including wrong words;
other scientific examples are authored controls rather than invented quotes.
"""
from dataclasses import replace
import json
import time

import pytest

from test_controller_v2 import controller
from zoom_copilot.interview_context import is_interview_request, analyze_question, question_parts
from zoom_copilot.question_packet import QuestionContext
from zoom_copilot.turns import (QuestionAssembler, has_complete_direct_question,
    confirmed_question_boundary, explicit_question_focus, next_addressed_question_offset,
    is_non_request_context)


REQUESTS=(
    'Please explain how you recover spectral and polarization information under scattering',
    "I'd like to know whether your calibration separates illumination from material reflectance",
    'Tell me about your contribution to spectral polarimetric computational ghost imaging',
    "I'm interested in how your measurements constrain the inverse rendering problem",
    'The question is whether recovered Stokes components identify the material parameters',
    'I would like to hear your thoughts on the limitations of your reconstruction method',
    'I am curious about the uncertainty of your estimated material properties',
    "I'd be interested in your interpretation of the wavelength dependent attenuation",
    "I'd like your thoughts on the identifiability of reconstructed material parameters",
    "I'd appreciate your comments on the calibration procedure under unknown illumination",
    'Your thoughts on the limitations of single pixel measurements',
    'Please explain polarization',
    'Tell me about calibration',
)


def qa(c):
    c.pref.caption_speaker_mode='single_remote';c.pref.auto_answer=True
    c._listen_pref=replace(c.pref);c.change_interview_phase('qa');c.listening=True


def speaking(c,text):
    c.assembler.voice();c.on_transcript('final',text);c.assembler.voice();c.auto_answer()


@pytest.mark.parametrize('question',REQUESTS)
@pytest.mark.parametrize('stop',['','.','?'])
def test_complete_polite_requests_need_no_question_mark(question,stop):
    source=question+stop
    assert has_complete_direct_question(source,require_terminal=False)
    assert is_interview_request(source)
    assert not QuestionAssembler.incomplete_text(source)
    assert not is_non_request_context(source)
    packet=QuestionContext().build(source)
    assert packet.raw_question_en==source
    assert packet.answer_question_en==source
    assert packet.subquestions==(source,)


@pytest.mark.parametrize('question',REQUESTS)
def test_unpunctuated_request_dispatches_at_real_pause_and_retains_source(controller,question):
    c=controller;qa(c);speaking(c,question)
    assert not c.busy
    c.assembler.last_voice=c.assembler.last_text=time.monotonic()-2
    c.auto_answer()
    assert c.busy and c._question_raw_by_request[c._active]==question
    assert c._question_packets[c._active].answer_question_en.rstrip('.?!')==question


@pytest.mark.parametrize('prefix',[
    'Please explain how', "I'd like to know whether",'Tell me about',
    "I'm interested in how you",'The question is',
    'I would like to hear your thoughts on', 'I am curious about',
    "I'd be interested in your interpretation of",
    "I'd like your thoughts on", "I'd appreciate your comments on", 'Your thoughts on',
    'I am curious whether it', 'I was wondering whether it', "I'd like to know if they",
])
def test_cropped_request_prefix_never_becomes_complete_from_guessed_punctuation(prefix):
    for stop in ('','.','?'):
        assert QuestionAssembler.incomplete_text(prefix+stop)
        assert not has_complete_direct_question(prefix+stop,require_terminal=False)
        assert confirmed_question_boundary(prefix+stop+' Sure.') is None


def test_split_request_keeps_all_conditions_and_does_not_fire_at_intro(controller):
    c=controller;qa(c)
    fragments=["I'd be interested in your interpretation of",'the wavelength dependent attenuation',
        'when illumination is unknown and the scene is dynamic']
    for fragment in fragments:
        speaking(c,fragment)
        assert not c.busy
    speaking(c,'Sure.')
    expected=' '.join(fragments)
    assert c.busy and c._question_raw_by_request[c._active]==expected
    assert 'illumination is unknown' in c._question_packets[c._active].answer_question_en


@pytest.mark.parametrize('question',REQUESTS)
def test_unpunctuated_complete_request_seals_on_independent_sure(controller,question):
    c=controller;qa(c);speaking(c,question)
    assert not c.busy
    speaking(c,'Sure.')
    assert c.busy and c._question_raw_by_request[c._active]==question


def test_embedded_polite_request_preserves_scientific_setup_and_exact_source():
    source=('You mentioned a calibrated sensing matrix I would like to know '
        'whether it remains valid when the illumination changes')
    question='I would like to know whether it remains valid when the illumination changes'
    focus,background=explicit_question_focus(source)
    assert focus==question
    assert background=='You mentioned a calibrated sensing matrix'
    assert is_interview_request(source)
    packet=QuestionContext().build(source)
    assert packet.raw_question_en==source and packet.answer_question_en==question
    assert packet.background_en==background and packet.subquestions==(question,)


@pytest.mark.parametrize('source',[
    'How can we acquire spectral and polarization information in complex environments? '
        'This question motivated my PhD research.',
    'How does the calibrated operator work? The answer is a linear forward model.',
    'Why do we need multiple spectral bands? I will now explain the acquisition method.',
    'The slide quotes "Please explain the calibration. Tell me about material reflectance?"',
    'For example, the professor said: I would like to know how your algorithm works.',
])
def test_narrated_self_questions_and_quoted_requests_are_background(source):
    assert is_non_request_context(source) or not is_interview_request(source)
    assert not has_complete_direct_question(source,require_terminal=False)
    assert not is_interview_request(source)
    assert explicit_question_focus(source)[0]==''
    assert question_parts(source)==[]
    assert analyze_question(source).interaction_kind=='setup' or not is_non_request_context(source)


def test_real_request_after_narrated_self_question_is_not_globally_suppressed():
    source=('How can we acquire spectral information? This question motivated my PhD research. '
        'Please explain how your method recovers polarization under scattering.')
    assert not is_non_request_context(source)
    assert has_complete_direct_question(source)
    focus,background=explicit_question_focus(source)
    assert focus=='Please explain how your method recovers polarization under scattering.'
    assert 'motivated my PhD' in background


SCREENSHOT_CROPS=(
    "Okay, that's up for the question. And my previous work with many folks on recovering spectrum "
    'and porous information. Information, but for... So, the, so... what I so this is me '
    "it's also list me a with a question, new question for me. How can beyond the recurring "
    'spectral and power change information?',
    'Or convey',
    'How can we use this measurements to reliably infer her material. Physical property of all',
)


@pytest.mark.parametrize('crop',SCREENSHOT_CROPS)
def test_actual_displayed_broken_screenshot_crops_are_not_guessed_into_complete_asks(crop):
    assembler=QuestionAssembler();assembler.add(crop,at=1);assembler.voice(at=1)
    assert assembler.text==crop
    assert assembler.incomplete and not assembler.ready(now=5)
    assert not has_complete_direct_question(crop,require_terminal=False)


@pytest.mark.parametrize('statement',[
    'We recover spectral and polarization information from calibrated measurements.',
    'I am interested in computational imaging as a research area.',
    'This question is important for inverse rendering.',
    'The question will be discussed after the presentation.',
    'You mentioned the optical system in your previous slide.',
    'Your method uses calibrated spectral measurements and a physical forward model.',
    'We should discuss this later after the seminar.',
    'You can explain how the model works using the next slide.',
])
def test_ordinary_research_statements_are_not_new_requests(statement):
    assert not has_complete_direct_question(statement,require_terminal=False)
    assert not is_interview_request(statement)
    assert next_addressed_question_offset(statement) is None


def test_thirteen_source_rows_cannot_crowd_complete_focus_out_of_4200_char_budget():
    context=QuestionContext()
    background=[f'This documented condition {index} concerns calibrated spectral measurements.'
        for index in range(12)]
    question=('How would you distinguish 95 percent confidence from identifiable material parameters '
        'without claiming that we measure S3?')
    fragments=background+[question]
    for index,fragment in enumerate(fragments):
        context.record(f'11111111-2222-3333-4444-555555555555:{index}',fragment,
            speaker_label='Zoom远端（单一远端模式）',speaker_status='source',accepted=True)
    raw=' '.join(fragments)
    packet=context.build(raw,raw_question=raw)
    coverage={'subquestion_count':1,'covered_subquestion_indices':[1],
        'uncovered_subquestion_indices':[],'coverage_is_retrieval_only':True}
    packet=replace(packet,retrieval_coverage=coverage)
    assert len(packet.turns)==13
    data=packet.to_prompt_data(max_chars=4200)
    assert len(json.dumps(data,ensure_ascii=False,separators=(',',':')))<=4200
    assert data['answer_question_en']==question
    assert data['subquestions_to_cover']==[question]
    assert data['retrieval_coverage']==coverage
    assert 0<len(data['speaker_attributed_question_turns'])<13
    assert all(row['identity_status']=='source' for row in data['speaker_attributed_question_turns'])
    assert packet.raw_question_en==raw and len(packet.turns)==13
    assert packet.to_ui_data()['turns'][-1]['raw_en']==question


BARE_INTRODUCTIONS=(
    'I have a question?', "I've got one quick question", 'One question?',
    'I have another brief question!', 'My next question.', 'Alex, one question?',
    'I got a question for you.', 'I have a quick question to ask.',
    'I have a question, if I may.',
)


@pytest.mark.parametrize('intro',BARE_INTRODUCTIONS)
def test_question_introduction_is_incomplete_even_with_a_guessed_question_mark(intro):
    assert QuestionAssembler.incomplete_text(intro)
    assert not has_complete_direct_question(intro,require_terminal=False)
    assert confirmed_question_boundary(intro+' Sure.') is None


@pytest.mark.parametrize('intro',BARE_INTRODUCTIONS)
def test_question_introduction_never_dispatches_on_silence_or_backchannel(controller,intro):
    c=controller;qa(c);speaking(c,intro)
    c.assembler.last_voice=c.assembler.last_text=time.monotonic()-2
    c.auto_answer()
    assert not c.busy and not c._requests
    speaking(c,'Sure.')
    assert not c.busy and not c._requests


def test_substantive_request_after_bare_introduction_remains_answerable(controller):
    c=controller;qa(c);speaking(c,'I have a question?')
    question='Please explain how you calibrate the sensing operator without measuring S3.'
    speaking(c,question);speaking(c,'Sure.')
    assert c.busy
    assert c._question_packets[c._active].answer_question_en==question
    assert c._question_raw_by_request[c._active]=='I have a question? '+question

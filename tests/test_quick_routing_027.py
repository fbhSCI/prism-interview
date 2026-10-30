"""Question routing must save time without swallowing technical questions."""
import json
import pytest

from test_controller_v2 import controller
from zoom_copilot.answers import build_prompt
from zoom_copilot.config import Preferences
from zoom_copilot.question_packet import QuestionContext
from zoom_copilot.quick_reply import quick_coordination_reply
from zoom_copilot.turns import QuestionAssembler


def test_start_order_uses_suggestion_without_answer_provider_or_paper_search(controller):
    ctrl=controller
    question=('We will each say a model name. The person who cannot name another model loses. '
              'So, do you want to start, or do you want me to start?')
    ctrl.request_answer(question)
    ctrl.test_run_worker()
    assert not ctrl.service.calls
    assert not ctrl.kb.queries
    result=ctrl._answers[-1]['result']
    assert result.provider=='local-template'
    assert result.answer_en=="Please go ahead. I'll follow."
    assert result.answer_zh and not result.citations
    assert result.diagnostics['cloud_answer_calls']==0
    assert not ctrl.busy


def test_false_period_between_start_order_alternatives_is_one_coordination_ask():
    question='We will take turns. So, do you want to start? Or do you want me to start?'
    result=quick_coordination_reply(question)
    assert result is not None and result.provider=='local-template'


def test_similar_scientific_and_multipart_questions_still_use_selected_provider(controller):
    ctrl=controller
    for question in ('Where do you start when calibrating an inverse-rendering system?',
                     'Would you start with the forward model, or the inverse algorithm?',
                     'Do you want to start, or should I start? How do you estimate Stokes S3?'):
        assert quick_coordination_reply(question) is None
    ctrl.request_answer('Where do you start when calibrating an inverse-rendering system?')
    ctrl.test_run_worker()
    assert len(ctrl.service.calls)==1
    assert ctrl.kb.queries


def test_automatic_setup_waits_for_actual_ask_without_cloud_or_search(controller):
    ctrl=controller
    ctrl.window.set_settings({'interview_phase':'qa','auto_answer':True})
    setup='We work with a lot of models, right? Let us play a small game, okay?'
    ctrl.request_answer(setup,automatic=True)
    assert not ctrl.busy and not ctrl.test_work
    assert not ctrl.service.calls and not ctrl.kb.queries
    combined=setup+' Do you want to start, or do you want me to start?'
    ctrl.request_answer(combined,automatic=True)
    ctrl.test_run_worker()
    assert ctrl._answers[-1]['result'].provider=='local-template'
    assert not ctrl.service.calls


def test_overlap_flush_does_not_block_complete_question_or_clear_prior_rejection(controller):
    ctrl=controller
    ctrl.window.set_settings({'interview_phase':'qa','auto_answer':True})
    ctrl.save_settings(ctrl.window.get_settings())
    ctrl.listening=True
    ctrl.on_transcript('final','Do you want to start, or do you want me to start?')
    ctrl.on_transcript('final','',{'empty_commit':True,'word_timed':True})
    assert not ctrl._question_uncertain
    ctrl.assembler.last_text=ctrl.assembler.last_voice=0
    ctrl.auto_answer()
    assert ctrl.busy
    ctrl.test_run_worker()
    assert ctrl._answers[-1]['result'].provider=='local-template'
    ctrl._question_uncertain=True
    ctrl.on_transcript('final','',{'empty_commit':True,'word_timed':True})
    assert ctrl._question_uncertain


def test_scientific_setup_and_all_actual_asks_reach_prompt_with_original_speech():
    raw=('We have four linear analyzer channels. How would you calibrate M? '
         'Can these channels recover S3?')
    ctx=QuestionContext()
    ctx.record('q',raw,speaker_id=2,speaker_label='说话人 2',accepted=True)
    packet=ctx.build(raw,raw_question=raw)
    _,text=build_prompt(raw,[],Preferences(provider='codex',share_evidence=True),
                        raw_question=raw,interview_packet=packet.to_prompt_data())
    data=json.loads(text)
    assert data['interview_task']['interaction_kind']=='research'
    assert 'How would you calibrate M?' in data['interview_task']['answer_focus']
    assert 'Can these channels recover S3?' in data['interview_task']['answer_focus']
    assert 'four linear analyzer channels' in data['interview_input']['background_en']
    assert data['current_question']==raw


@pytest.mark.parametrize('prefix,tail',[
    ('So, what would','be your metric for object detection?'),
    ('Which assumptions fail in your','forward sensing model?'),
    ('How would you compare their.','noise sensitivity?'),
    ('Okay, so do you want to start or do.','you want me to start?'),
    ('How do you calibrate, and can.','the instrument recover S3?'),
])
def test_open_audio_cut_does_not_trigger_answer_before_continuation(prefix,tail):
    assembler=QuestionAssembler()
    assembler.add(prefix,at=10)
    assert assembler.incomplete
    assert not assembler.ready(now=14)
    assembler.add(tail,at=15)
    assert not assembler.incomplete
    assert assembler.ready(now=19)
    assert assembler.text.startswith(prefix)

"""The visible source-attributed question is the frozen input to the answer."""
from test_controller_v2 import controller
from zoom_copilot.answers import build_prompt,estimate_prompt_tokens,LOCAL_MAX_OUTPUT_TOKENS,LOCAL_CONTEXT_MARGIN
from zoom_copilot.config import Preferences
from zoom_copilot.knowledge import Passage
from zoom_copilot.question_packet import QuestionContext
from types import SimpleNamespace
import json
import pytest


def say(c,text,who='prof',turn='a'):
    c.on_transcript('final',text,dict(phase_epoch=c._phase_epoch,word_timed=True,
        speaker_id=who,speaker_label='面试官' if who=='prof' else '候选人',turn_id=turn))


def test_question_preview_and_dispatched_packet_keep_sources_and_paused_clauses(controller):
    c=controller
    say(c,'How do you distinguish the forward instrument matrix')
    say(c,'from the Stokes demodulation matrix?')
    say(c,'And does it also include the spatial spectral sensing operator?')
    view=c.window.question_packet_view.toPlainText()
    assert '面试官' in view and '1.' not in view
    assert 'forward instrument matrix' in view and 'spatial spectral' in view
    c.request_answer(c.window.current_question())
    c.test_run_worker()
    call=c.service.calls[0]
    assert 'forward instrument matrix' in call['raw_question'] and 'spatial spectral' in call['raw_question']
    turns=call['interview_packet']['speaker_attributed_question_turns']
    assert turns and all(t['speaker_label']=='面试官' for t in turns)
    assert all(t['role']=='interviewer' for t in turns)


def test_packet_snapshot_is_not_changed_by_later_question_or_speaker_rename(controller):
    c=controller
    say(c,'How would you calibrate the instrument?')
    c.request_answer(c.window.current_question())
    old=c._active
    c.assign_speaker('a','prof','老师新称呼')
    c.new_question()
    say(c,'Which measurements constrain the geometry?',turn='b')
    c.test_run_worker()
    call=c.service.calls[0]
    assert call['request_id']==old
    assert 'calibrate' in call['question'] and 'geometry' not in call['question']
    assert call['interview_packet']['speaker_attributed_question_turns'][0]['speaker_label']=='面试官'


def test_previous_voices_statement_is_context_for_the_new_question(controller):
    c=controller
    say(c,'The model is good at detecting a person.','candidate','prior')
    say(c,'So what would you use to evaluate its accuracy','prof','question')
    assert c.assembler.text=='So what would you use to evaluate its accuracy'
    packet=c._make_question_packet()
    assert packet.turns[0].speaker_id=='prof'
    assert any('detecting a person' in t.raw_en for t in packet.recent_context)
    c.request_answer(c.window.current_question())
    c.test_run_worker()
    data=c.service.calls[0]['interview_packet']
    assert data['recent_speech_for_followup_resolution_only']
    assert 'detecting a person' not in c.service.calls[0]['question']


def test_explicit_next_question_retires_prior_answer_mode_but_keeps_manual_roles(controller):
    c=controller
    say(c,'How do you evaluate accuracy?')
    say(c,"I'm not sure which metric was used.",'candidate','answer')
    c.change_speaker_role('candidate','ignore')
    c.new_question()
    say(c,'What about the uncertainty?','prof','new')
    assert c.assembler.text=='What about the uncertainty?'
    say(c,'Some further background from the candidate.','candidate','a2')
    assert 'further background' not in c.assembler.text


def test_nonquestion_setup_is_not_assigned_an_interviewer_role_without_evidence(controller):
    c=controller
    say(c,'Consider the current scattering environment.','voice1','setup')
    c.on_transcript('final','So what would be the metric',dict(word_timed=True,turn_id='unknown',phase_epoch=c._phase_epoch))
    packet=c._make_question_packet()
    assert packet.turns[0].role=='unknown'
    assert packet.turns[-1].role=='interviewer'
    assert 'scattering environment' in packet.question_en


def test_manual_candidate_role_removes_only_their_fragment_from_current_question(controller):
    c=controller
    say(c,'How would you validate the material reflectance?')
    say(c,'The reference target has controlled roughness.','candidate','b')
    assert 'reference target' in c.assembler.text
    c.change_speaker_role('candidate','ignore')
    assert c.assembler.text=='How would you validate the material reflectance?'
    assert 'reference target' not in c.window.question_packet_view.toPlainText()
    c.request_answer(c.window.current_question())
    c.test_run_worker()
    assert 'reference target' not in c.service.calls[0]['question']


def test_growing_card_keeps_previous_chinese_until_new_translation_arrives(controller):
    c=controller
    jobs=[]
    c.translation=SimpleNamespace(submit=lambda *args:jobs.append(args),stop=lambda:None)
    say(c,'How do you calibrate the matrix?')
    identity=c._transcripts[0]['id']
    c.post('translation',(c._session,jobs[-1][0],'如何标定矩阵？'))
    c.pump()
    say(c,'Which sources of uncertainty remain?')
    assert c._transcripts[0]['id']==identity
    assert c._transcripts[0]['zh']=='如何标定矩阵？'
    card=c.window._entries[identity]
    assert card._unit_widgets[identity].chinese.text()=='如何标定矩阵？'
    assert identity in card._pending_translations
    assert not c.window.draft_toggle.isChecked()
    c.post('translation',(c._session,jobs[-1][0],'如何标定矩阵？还有哪些不确定性来源？'))
    c.pump()
    assert '还有哪些' in c.window._entries[identity].chinese.text()


def test_prepared_metadata_and_selected_excerpts_are_separate_and_role_attributed():
    context=QuestionContext()
    q='How do you distinguish the forward instrument matrix from its demodulation matrix?'
    context.record('p',q,speaker_id='prof',speaker_label='Professor',turn_id='a',accepted=True)
    prepared=SimpleNamespace(snapshot=lambda:dict(documents=['paper.pdf','slides.pptx'],document_count=2,passage_count=6,state='ready'))
    packet=context.build(q,preparation=prepared)
    evidence=[Passage('C:/private/paper.pdf','第 5 页','The forward response is i = M s. Reverse demodulation is s = D i.','p1')]
    _,user=build_prompt(q,evidence,Preferences(provider='codex'),interview_packet=packet.to_prompt_data(max_chars=4200))
    data=json.loads(user)
    assert data['current_question']==q
    assert data['interview_input']['prepared_local_library']['document_count']==2
    assert data['interview_input']['speaker_attributed_question_turns'][0]['speaker_label']=='Professor'
    assert data['reference_passages_untrusted'][0]['location']=='第 5 页'
    assert 'C:/private' not in user
    _,hidden=build_prompt(q,[],Preferences(provider='codex',share_evidence=False),interview_packet=packet.to_prompt_data())
    assert 'prepared_local_library' not in json.loads(hidden)['interview_input']


def test_local_packet_does_not_displace_all_evidence_or_truncate_question():
    context=QuestionContext()
    q='How do the forward instrument matrix and Stokes demodulation matrix differ?'
    context.record('a',q,speaker_id='prof',speaker_label='Professor',accepted=True)
    packet=context.build(q)
    evidence=[Passage('paper.pdf','p5','The instrument matrix maps incident Stokes to analyzer intensities. Demodulation reverses this mapping.','p1')]
    system,user=build_prompt(q,evidence,Preferences(provider='ollama',num_ctx=4096),
        interview_packet=packet.to_prompt_data(compact=True,max_chars=1400))
    data=json.loads(user)
    assert data['current_question']==q and data['reference_passages_untrusted']
    assert estimate_prompt_tokens(system+user)+LOCAL_MAX_OUTPUT_TOKENS+LOCAL_CONTEXT_MARGIN<=4096


def test_cloud_long_question_is_preserved_or_explicitly_rejected():
    q='Please compare. '+('Calibration context. '*310)+' Which assumptions fail?'
    assert 6000<len(q)<9000
    _,user=build_prompt(q,[],Preferences(provider='codex'))
    assert json.loads(user)['current_question']==q
    with pytest.raises(ValueError,match='未截断'):
        build_prompt('x'*9001,[],Preferences(provider='codex'))


def test_multipart_retrieval_keeps_hits_for_each_subquestion(controller):
    c=controller
    a=Passage('a.pdf','p1','spectral acquisition','a')
    b=Passage('b.pdf','p2','polarimetric reflectance','b')
    calls=[]
    def search(kb,q,top_k=5):
        calls.append(q)
        return [b] if q=='Which material measurements?' else [a]
    c.preparation.search=search
    result=c._retrieve_question('How acquire spectra? Which material measurements?',SimpleNamespace(subquestions=('Readable combined UI paragraph',)),2)
    assert result==[a,b]
    assert 'Which material measurements?' in calls


def test_generic_subquestion_does_not_displace_full_question_definition(controller):
    c=controller
    full=[Passage('paper.pdf',f'p{i}',f'Full definition {i}',f'f{i}') for i in range(5)]
    child=[Passage('generic.pdf',f'p{i}',f'Generic related content {i}',f'g{i}') for i in range(5)]
    c.preparation.search=lambda kb,q,top_k:full if q=='Full question' else child
    result=c._retrieve_question('Full question',SimpleNamespace(subquestions=('Generic subquestion?',)),5)
    assert result[:4]==full[:4]
    assert result==full

"""Live revisions, raw evidence and interview routing across asynchronous stages."""
from types import SimpleNamespace
import json
from test_controller_v2 import controller
from zoom_copilot.interview_context import InterviewPreparation,question_parts
from zoom_copilot.answers import build_prompt
from zoom_copilot.config import Preferences
import threading


def clock(c,end,lexical=None):
    return dict(phase_epoch=c._phase_epoch,stream_id='stream',word_timed=True,
                end_sample=end,text_end_sample=end if lexical is None else lexical)


def test_growing_preview_updates_chinese_without_reverting_newer_words(controller):
    c=controller
    c.pref.live_translation=True
    c.on_transcript('partial','How would you model')
    key=('live',c._phase_epoch,c._live_token)
    c.on_transcript('partial','How would you model scattering')
    c.post('translation_preview',(c._session,key,'How would you model','如何建模'))
    c.pump()
    assert c._live_data['en'].endswith('scattering')
    assert c._live_data['zh']=='如何建模'
    c.on_transcript('partial','Why is that assumption valid')
    c.post('translation_preview',(c._session,key,'How would you model','过时中文'))
    c.pump()
    assert c._live_data['zh']==''


def test_final_commit_does_not_remove_newer_streaming_tail(controller):
    c=controller
    c.on_transcript('partial','How do you calibrate and validate',clock(c,12000))
    c.on_transcript('final','How do you calibrate?',clock(c,8000))
    assert c._live_data['en'].endswith('validate')
    assert c._partial_id is not None
    c.on_transcript('partial','and validate',clock(c,13000))
    assert 'calibrate' in c._live_data['en'] and 'validate' in c._live_data['en']
    c.on_transcript('final','and validate?',clock(c,13000))
    c.on_transcript('partial','',clock(c,15000,lexical=13000))
    assert c._partial_id is None
    assert c._live_data['en']==c._transcripts[0]['en']


def test_old_phase_final_is_history_only_and_stale_partial_cannot_reappear(controller):
    c=controller
    old=clock(c,5000)
    c.new_question()
    c.on_transcript('partial','Current new words',clock(c,9000))
    current=c._live_data['en']
    c.on_transcript('final','Old phase words.',old)
    assert c._live_data['en']==current
    assert c.window.live_caption.toPlainText()==current
    c.on_transcript('final','Current new words.',clock(c,9000))
    c.on_transcript('partial','Old obsolete words',clock(c,8000))
    assert 'obsolete' not in c._live_data['en']


def test_raw_question_preserved_and_candidate_response_excluded(controller):
    c=controller
    def meta(who,turn):
        return dict(phase_epoch=c._phase_epoch,speaker_id=who,speaker_label=f'Speaker {who}',turn_id=turn)
    c.on_transcript('final','How do you model',meta(1,'a'))
    c.on_transcript('final','scattering and polarization?',meta(1,'a'))
    assert c.assembler.text=='How do you model scattering and polarization?'
    c.on_transcript('final',"I think we can estimate the confidence of the probabilities.",meta(2,'b'))
    assert 'probabilities' not in c.assembler.text
    c.on_transcript('final','How would you validate the result?',meta(1,'c'))
    assert c.assembler.text=='How would you validate the result?'
    c.window.set_settings({'auto_answer':True,'interview_phase':'qa'})
    c.request_answer(c.assembler.text,automatic=True)
    c.test_run_worker()
    assert c.service.calls[0]['raw_question']=='How would you validate the result?'


def test_manual_speaker_assignment_survives_later_acoustic_revision(controller):
    c=controller
    c.on_transcript('final','Please explain your method.',dict(turn_id='a',speaker_id=None))
    c.assign_speaker('a','manual:prof','老师')
    c._apply_speaker_revision(dict(turn_id='a',speaker_id=2,speaker_label='Speaker 2',speaker_status='confirmed'))
    assert c._transcripts[0]['speaker_label']=='老师'
    assert c._transcripts[0]['speaker_status']=='manual'


def test_probability_self_questions_do_not_promote_candidate_to_interviewer(controller):
    c=controller
    def say(text,who,turn):
        c.on_transcript('final',text,dict(speaker_id=who,turn_id=turn,phase_epoch=c._phase_epoch))
    say('How would you measure confidence?',1,'q')
    say("I'm not sure if I use any metric, but I think we can look at the probabilities.",2,'a')
    say("What's the confidence? What's the probability of him being a person?",2,'a')
    say('This model is predicting him as a person with 95% probability and 5% as other objects.',2,'a')
    assert c.assembler.text=='How would you measure confidence?'
    assert len(c._transcripts)>=2
    c.intent.set_role(2,'interviewer')
    say('Which metric should we use?',2,'new-prof')
    assert c.assembler.text=='Which metric should we use?'


def test_unknown_and_colliding_speaker_answer_fragments_cannot_replace_interviewer_question(controller):
    c=controller
    c.window.set_settings({'auto_answer':True,'interview_phase':'qa'})
    c.save_settings(c.window.get_settings())
    def say(text,who,turn):
        c.on_transcript('final',text,dict(speaker_id=who,turn_id=turn,phase_epoch=c._phase_epoch))
    say('Which metrics have you used to assess model performance?',2,'teacher')
    say("I'm not sure.",2,'candidate-misclustered')
    say('If I use any metric, but I think we can look at the probabilities.',None,'candidate-unknown')
    say("If we are detecting a person, then what's the confidence?",1,'candidate-newcluster')
    say("What's the probability of him being a person?",1,'candidate-newcluster')
    assert c.assembler.text=='Which metrics have you used to assess model performance?'
    assert c.busy and not c._requests
    c.test_run_worker()
    assert len(c.service.calls)==1
    assert c.service.calls[0]['question']=='Which metrics have you used to assess model performance?'


def test_unpunctuated_setup_still_detects_addressed_question_before_candidate_reply(controller):
    c=controller
    c.window.set_settings({'auto_answer':True,'interview_phase':'qa'})
    c.save_settings(c.window.get_settings())
    def say(text,who,turn):
        c.on_transcript('final',text,dict(speaker_id=who,turn_id=turn,phase_epoch=c._phase_epoch))
    say('this is good at object detection',1,'prior')
    say('so what would be my metric to see if my model is performing well so do you know any metrics which you have used',2,'teacher')
    say("of use the mad I'm not sure if I use",None,'boundary')
    say('any metric but I think we can look at the probabilities',None,'boundary2')
    say("what's the probability of him being a person",1,'candidate')
    say('is okay this model',1,'candidate2')
    assert c.busy and not c._requests
    assert 'do you know any metrics' in c.assembler.text
    assert 'probability of him' not in c.assembler.text
    c.test_run_worker()
    assert len(c.service.calls)==1 and 'do you know any metrics' in c.service.calls[0]['raw_question']


def test_unknown_speaker_does_not_drop_middle_of_question_and_split_response_cue(controller):
    c=controller
    c.window.set_settings({'auto_answer':True,'interview_phase':'qa'})
    c.save_settings(c.window.get_settings())
    def say(text,who,turn):
        c.on_transcript('final',text,dict(speaker_id=who,turn_id=turn,phase_epoch=c._phase_epoch))
    say('How would you determine whether my',2,'teacher')
    say('model is performing well for object detection',None,'uncertain-boundary')
    say('so do you know any metrics you have used',3,'temporary-newcluster')
    say("in your case I'm not",None,'mixed-boundary')
    say('sure if I use any metric but I think we can look at probabilities',None,'candidate')
    say("what's the probability of this person",1,'candidate')
    assert c.busy and not c._requests
    c.test_run_worker()
    q=c.service.calls[0]['raw_question']
    assert 'model is performing well for object detection' in q
    assert 'do you know any metrics' in q
    assert "I'm not" not in q and 'probabilities' not in q


def test_connection_prewarm_deduplicates_and_never_sends_an_answer(controller):
    c=controller
    seen=[]
    c.service.available_models=lambda prefs:seen.append(prefs.provider) or [{'model':'a'}]
    c.check_models();c.check_models()
    assert len(c.test_work)==1
    c.test_run_worker()
    assert seen==['codex'] and not c._model_check_running and not c.service.calls
    c.window.set_settings({'cloud_consent':False})
    c.check_models()
    assert not c.test_work and seen==['codex']


def test_listening_ui_waits_until_audio_recorder_is_open(controller,monkeypatch):
    import zoom_copilot.app as module
    c=controller
    ready=threading.Event();ready.set()
    class Capture:
        def __init__(self,*args,**kwargs):self.callback=kwargs['on_ready']
        def start(self):pass
        def stop(self):pass
    monkeypatch.setattr(module,'AudioCapture',Capture)
    c.preparing=True;c._translation_ready=True;c._restoration_ready=True
    c.speech=SimpleNamespace(ready_event=ready,submit=lambda *args:None,stop=lambda *args,**kwargs:None)
    c._start_capture(c._session)
    assert c.preparing and not c.listening
    c.audio.callback();c.pump()
    assert c.listening and not c.preparing


def test_preparation_invalidates_after_import_without_cloud():
    class KB:
        document_names=['paper.pdf'];passages=[1];_revision=1
        def __init__(self):self.calls=0
        def search(self,question,top_k=5):self.calls+=1;return [self._revision]
    kb=KB();p=InterviewPreparation();p.prepare(kb)
    first=kb.calls
    assert p.search(kb,'Current question')==[1]
    assert p.search(kb,'Current question')==[1] and kb.calls==first+1
    kb._revision=2
    assert p.search(kb,'Current question')==[2] and kb.calls==first+2


def test_prompt_has_current_question_checklist_and_raw_english():
    question='How do you calibrate? Which assumptions fail?'
    raw='How do you uh calibrate? Which assumptions fail?'
    _,encoded=build_prompt(question,[],Preferences(provider='codex'),raw_question=raw)
    data=json.loads(encoded)
    assert data['recognized_english_before_cleanup']==raw
    assert data['interview_task']['question_checklist']==question_parts(question)
    assert data['current_question']==question

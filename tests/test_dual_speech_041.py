"""Two input devices share model weights, never recognition/commit state."""
from types import SimpleNamespace
import threading
import time

import numpy as np

from zoom_copilot.audio import AudioChunk
from zoom_copilot.speech import SpeechWorker
from zoom_copilot.streaming_asr import StreamingDraftWorker


def until(predicate):
    end = time.perf_counter() + 3
    while not predicate() and time.perf_counter() < end:
        time.sleep(.005)
    assert predicate()


def chunk(source, value=1, start=0, size=1600):
    return AudioChunk(np.full(size, value, dtype=np.float32),
        start_sample=start, commit_start_sample=start, commit_end_sample=start+size,
        stream_id=source+'-stream', capture_context=1, input_source=source)


def test_both_sources_keep_words_at_the_same_source_time_with_one_whisper_model():
    models, texts = [], []

    class Model:
        def __init__(self, *a, **kw):
            models.append(self)
        def transcribe(self, audio, **kw):
            text = ' Remote' if audio[0] == 1 else ' Mine'
            word = SimpleNamespace(word=text, start=.01, end=.08)
            return iter([SimpleNamespace(text=text, words=[word])]), None

    worker = SpeechWorker('test', model_factory=Model, on_text=lambda *args:texts.append(args))
    worker.submit('final', chunk('loopback', 1))
    worker.submit('final', chunk('microphone', 2))
    assert worker.pending_for_source('loopback') == 1
    assert worker.pending_for_source('microphone') == 1
    worker.start()
    until(lambda:len(texts) == 2)
    worker.stop();worker.thread.join(2)
    assert len(models) == 1
    assert [(v[1], v[2]['input_source']) for v in texts] == [('Remote','loopback'),('Mine','microphone')]
    assert all(v[2]['recognition_gap'] is False for v in texts)
    assert worker.pending == 0


def test_mic_final_does_not_invalidate_a_remote_partial_in_progress():
    entered, release = threading.Event(), threading.Event()
    texts = []

    class Model:
        def __init__(self,*a,**kw):pass
        def transcribe(self, audio, **kw):
            if audio[0] == 1:
                entered.set();assert release.wait(2)
            return iter([SimpleNamespace(text='Remote draft' if audio[0] == 1 else 'My response')]),None

    worker = SpeechWorker('test', model_factory=Model, on_text=lambda *args:texts.append(args))
    worker.start();assert worker.ready_event.wait(2)
    worker.submit('partial',chunk('loopback',1))
    assert entered.wait(2)
    worker.submit('final',chunk('microphone',2))
    release.set()
    until(lambda:len(texts)==2)
    worker.stop();worker.thread.join(2)
    assert [(v[0],v[1],v[2]['input_source']) for v in texts] == [
        ('partial','Remote draft','loopback'),('final','My response','microphone')]


def test_mic_rejection_keeps_source_and_does_not_enroll_a_remote_voiceprint():
    texts, annotations = [], []
    class Diarizer:
        last_metrics = {}
        def prepare(self):return True
        def annotate(self,*args):
            annotations.append(True)
            return [{'text':'Remote words'}]
    class Model:
        def __init__(self,*a,**kw):pass
        def transcribe(self,audio,**kw):
            return iter([SimpleNamespace(text='Remote words',avg_logprob=-2 if audio[0]==2 else 0)]),None
    worker=SpeechWorker('test',model_factory=Model,diarizer=Diarizer(),on_text=lambda *a:texts.append(a))
    worker.submit('final',chunk('microphone',2))
    worker.submit('final',chunk('loopback',1))
    worker.start();until(lambda:len(texts)==2)
    worker.stop();worker.thread.join(2)
    assert texts[0][0]=='rejected' and texts[0][2]['input_source']=='microphone'
    assert texts[1][2]['input_source']=='loopback' and not texts[1][2]['recognition_gap']
    assert len(annotations)==1


def test_remote_pending_gate_does_not_count_my_queued_audio():
    worker=SpeechWorker('test',model_factory=lambda *a,**k:None)
    worker.submit('partial',chunk('loopback'))
    worker.submit('final',chunk('microphone'))
    assert worker.pending_for_source('loopback')==1
    assert worker.pending_for_source('microphone')==1
    worker.stop(False)
    assert worker.pending_for_source('loopback')==worker.pending_for_source('microphone')==0


def test_interleaved_streaming_blocks_keep_both_sentences_and_independent_commits():
    class Stream:
        def __init__(self):self.accepted=self.decoded=0;self.source=None
        def accept_waveform(self,rate,audio):
            self.source=self.source or ('remote' if audio[0]==1 else 'mine')
            self.accepted+=len(audio)
    class Recognizer:
        def __init__(self):self.recognizer=self;self.streams=[]
        def create_stream(self):
            v=Stream();self.streams.append(v);return v
        def is_ready(self,v):return v.decoded<v.accepted
        def decode_stream(self,v):v.decoded=v.accepted
        def get_result(self,v):
            tokens=[' '+v.source,' second',' third'][:v.decoded//1600]
            return SimpleNamespace(tokens=tokens,timestamps=[i*.1 for i in range(len(tokens))])
    recognizer=Recognizer();texts=[]
    worker=StreamingDraftWorker(None,lambda *v:texts.append(v),recognizer_factory=lambda:recognizer)
    worker.start();assert worker.ready_event.wait(2)
    def submit(source,value,start):
        worker.submit(np.full(1600,value),{'stream_id':source,'capture_context':1,
            'input_source':source,'start_sample':start,'speech_active':True})
    submit('loopback',1,0);until(lambda:len(texts)==1)
    submit('microphone',2,0);until(lambda:len(texts)==2)
    worker.commit({'stream_id':'microphone','capture_context':1,'commit_end_sample':1600})
    submit('loopback',1,1600);until(lambda:len(texts)==3)
    submit('microphone',2,1600);until(lambda:len(texts)==4)
    worker.stop(False);worker.thread.join(2)
    assert len(recognizer.streams)==2
    assert [(v[1],v[2]['input_source']) for v in texts] == [
        ('remote','loopback'),('mine','microphone'),
        ('remote second','loopback'),('second','microphone')]

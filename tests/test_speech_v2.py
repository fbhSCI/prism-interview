from types import SimpleNamespace
import threading
import time
import numpy as np

from zoom_copilot.speech import SpeechWorker
from zoom_copilot.turns import QuestionAssembler, join_fragments


def until(predicate, timeout=3):
    deadline = time.monotonic()+timeout
    while time.monotonic()<deadline:
        if predicate():
            return
        time.sleep(.01)
    assert predicate()


def test_finals_are_retained_during_loading_and_previews_do_not_evict_them():
    release = threading.Event()
    texts = []
    class Model:
        def __init__(self, *args, **kwargs):
            release.wait(2)
        def transcribe(self, audio, **kwargs):
            return iter([SimpleNamespace(text=f"Question number {int(audio[0])}?")]), None
    worker = SpeechWorker("test", on_text=lambda kind,en,zh: texts.append((kind,en)), model_factory=Model)
    worker.start()
    for number in range(12):
        assert worker.submit("final", np.full(1000, number))
        worker.submit("partial", np.ones(1000))
    release.set()
    until(lambda: len(texts)==12)
    worker.stop()
    assert texts == [("final", f"Question number {n}?") for n in range(12)]


def test_short_followup_is_preserved_and_translation_is_not_on_asr_thread():
    class Model:
        def __init__(self,*args,**kwargs): pass
        def transcribe(self,*args,**kwargs):
            return iter([SimpleNamespace(text="Why?")]),None
    class SlowTranslator:
        def translate(self,text):
            raise AssertionError("ASR must not wait for translation")
    output=[]
    worker=SpeechWorker("test",SlowTranslator(),lambda *args:output.append(args),model_factory=Model)
    worker.start()
    assert worker.ready_event.wait(2)
    worker.submit("final",np.ones(2000))
    until(lambda:bool(output))
    worker.stop()
    assert output==[("final","Why?","")]


def test_model_load_error_is_explicit_and_new_instance_can_recover():
    def bad(*args,**kwargs): raise RuntimeError("bad weights")
    broken=SpeechWorker("bad",model_factory=bad)
    broken.start()
    assert broken.failed_event.wait(2)
    assert not broken.ready_event.is_set()
    assert not broken.submit("final",np.ones(1000))
    class Model:
        def __init__(self,*args,**kwargs): pass
    recovered=SpeechWorker("good",model_factory=Model)
    recovered.start()
    assert recovered.ready_event.wait(2)
    recovered.stop()


def test_backlog_overflow_is_explicit_not_silent_eviction():
    worker=SpeechWorker("test",model_factory=lambda *a,**k:None)
    assert worker.submit("final",np.ones(120*16000,dtype=np.float32))
    assert not worker.submit("final",np.ones(16000,dtype=np.float32))
    assert worker.failed_event.is_set()
    assert worker.pending==1


def test_question_assembly_waits_for_real_silence_and_processing():
    turn=QuestionAssembler(silence_seconds=1.6)
    turn.voice(10)
    turn.add("How do you separate the sensing operator",10.3)
    turn.voice(11.5)
    turn.add("from the Stokes calibration matrix?",12)
    assert not turn.ready(12.5)
    assert not turn.ready(14,pending=1)
    assert turn.ready(14,pending=0)
    question,version=turn.consume()
    assert question=="How do you separate the sensing operator from the Stokes calibration matrix?"
    assert version==2 and not turn.ready(20)
    turn.add("Why?",20)
    assert turn.ready(22)
    assert turn.consume()[0]=="Why?"


def test_overlap_removes_repeated_words_without_losing_fragments():
    assert join_fragments("What is the sensing operator", "sensing operator used for?")=="What is the sensing operator used for?"
    assert join_fragments("Why?", "Why?")=="Why?"
    assert join_fragments("How does it work?", "And what are the limitations?")=="How does it work? And what are the limitations?"

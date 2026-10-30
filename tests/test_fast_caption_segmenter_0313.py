"""Stable full-context commits and one shorter first window; no model/device."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from zoom_copilot import audio as audio_module
from zoom_copilot.audio import AudioCapture,Segmenter,RATE
from zoom_copilot.speech import SpeechWorker


def feed(segmenter,audio):
    final=[]
    for offset in range(0,len(audio),1600):
        final.extend(c for kind,c in segmenter.push(audio[offset:offset+1600]) if kind=='final')
    return final


def fast():
    return Segmenter(vad=False,maximum_sec=3.6,partial_sec=1.8,
        commit_interval_sec=1.6,initial_maximum_sec=3.0)


def test_legacy_default_final_windows_and_commits_stay_exactly_the_same():
    seg=Segmenter(vad=False,maximum_sec=3.6,partial_sec=1.8)
    frames=feed(seg,np.ones(12*RATE,np.float32));frames.append(seg.flush())
    assert [(c.start_sample,c.commit_start_sample,c.commit_end_sample,len(c),c.hard_cut) for c in frames]==[
        (0,0,48000,57600,True),
        (38400,48000,86400,57600,True),
        (76800,86400,124800,57600,True),
        (115200,124800,163200,57600,True),
        (153600,163200,192000,38400,False),
    ]


def test_fast_windows_keep_full_context_after_only_first_three_seconds():
    seg=fast()
    frames=feed(seg,np.ones(10*RATE,np.float32));frames.append(seg.flush())
    assert [len(c)/RATE for c in frames]==[3.0,3.6,3.6,3.6,3.6,2.6]
    assert [c.commit_end_sample/RATE for c in frames]==[2.4,4.0,5.6,7.2,8.8,10.0]
    assert seg.overlap_samples==2*RATE and seg.holdback_samples==int(.6*RATE)
    assert frames[-1].hard_cut is False and seg.flush() is None


@pytest.mark.parametrize('extra_samples',[0,1,1599])
def test_fast_commit_intervals_cover_every_source_sample_and_last_tail(extra_samples):
    audio=(1+np.arange(10*RATE+extra_samples,dtype=np.float32)/RATE)
    seg=fast();frames=feed(seg,audio)
    tail=seg.flush()
    if tail is not None:frames.append(tail)
    assert frames[0].commit_start_sample==0
    assert frames[-1].commit_end_sample==len(audio)
    assert all(a.commit_end_sample==b.commit_start_sample for a,b in zip(frames,frames[1:]))
    assert all(c.commit_start_sample<c.commit_end_sample for c in frames)
    for c in frames:
        np.testing.assert_array_equal(c,audio[c.start_sample:c.start_sample+len(c)])


def test_soft_first_final_consumes_short_window_without_resetting_at_next_pause():
    seg=fast()
    first=feed(seg,np.concatenate([np.ones(5*1600,np.float32),np.zeros(12*1600,np.float32)]))
    assert len(first)==1 and not first[0].hard_cut
    seg.reset()  # An utterance reset is deliberately not a capture-session reset.
    later=feed(seg,np.ones(4*RATE,np.float32))
    assert later and len(later[0])==int(3.6*RATE)


def test_valid_flush_consumes_first_window_but_rejected_short_noise_does_not():
    seg=fast();feed(seg,np.ones(8000,np.float32))
    assert seg.flush() is not None
    frames=feed(seg,np.ones(4*RATE,np.float32))
    assert frames and len(frames[0])==int(3.6*RATE)
    fresh=fast();feed(fresh,np.ones(1600,np.float32))
    assert fresh.flush() is None
    frames=feed(fresh,np.ones(4*RATE,np.float32))
    assert frames and len(frames[0])==3*RATE


def test_optional_cadence_without_short_first_window_keeps_first_full_context():
    seg=Segmenter(vad=False,maximum_sec=3.6,commit_interval_sec=1.6)
    frames=feed(seg,np.ones(7*RATE,np.float32))
    assert all(len(c)==int(3.6*RATE) for c in frames)
    assert [c.commit_end_sample/RATE for c in frames]==[3.0,4.6,6.2]


def test_word_timed_commit_filter_keeps_boundary_words_and_final_tail_exactly_once():
    seg=fast();frames=feed(seg,np.ones(10*RATE,np.float32));frames.append(seg.flush())
    source=[(f' w{i}',int((.1+i*.4)*RATE),int((.3+i*.4)*RATE)) for i in range(25)]
    worker=SpeechWorker('no-model');emitted=[]
    for c in frames:
        words=[SimpleNamespace(word=text,start=(lo-c.start_sample)/RATE,end=(hi-c.start_sample)/RATE)
            for text,lo,hi in source if c.start_sample<=lo and hi<=c.start_sample+len(c)]
        part=SimpleNamespace(text=''.join(w.word for w in words),words=words,avg_logprob=0.)
        meta={key:getattr(c,key) for key in ('start_sample','commit_start_sample','commit_end_sample','stream_id','hard_cut')}
        text,rejections,timed=worker._decode_text([part],'final',meta)
        assert timed and not rejections
        emitted.extend(text.split())
    assert emitted==[text.strip() for text,_,_ in source]


def test_capture_forwards_fast_policy_and_epoch_flush_keeps_only_new_source(monkeypatch):
    emitted=[];constructed=[]
    capture=AudioCapture('fake',.005,lambda kind,c:emitted.append((kind,c)),lambda value:None,
        lambda message:None,context=41,maximum_sec=3.6,streaming=True,
        commit_interval_sec=1.6,initial_maximum_sec=3.0)
    class Recorder:
        index=0
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def record(self,numframes):
            self.index+=1
            if self.index==7:capture.set_context(42)
            if self.index==50:capture.stop()
            return np.full((numframes,1),1. if self.index<7 else 2.,np.float32)
    device=SimpleNamespace(isloopback=True,name='fake',recorder=lambda **kwargs:Recorder())
    monkeypatch.setitem(sys.modules,'soundcard',SimpleNamespace(get_microphone=lambda *args,**kwargs:device))
    def segmenter(**kwargs):
        constructed.append(kwargs)
        return Segmenter(vad=False,**kwargs)
    monkeypatch.setattr(audio_module,'Segmenter',segmenter)
    import scipy.signal
    monkeypatch.setattr(scipy.signal,'resample_poly',lambda values,up,down:values[::down])
    capture._run()
    assert constructed[0]['commit_interval_sec']==1.6
    assert constructed[0]['initial_maximum_sec']==3.0
    frames=[c for kind,c in emitted if kind=='final']
    assert frames[0].capture_context==41 and len(frames[0])==6*1600
    assert np.all(frames[0]==1.)
    assert frames[1].capture_context==42 and len(frames[1])==int(3.6*RATE)
    assert all(np.all(c==2.) for c in frames[1:])
    assert frames[1].start_sample==6*1600
    assert frames[-1].commit_end_sample==50*1600
    assert not capture.error_event.is_set()


@pytest.mark.parametrize('settings',[
    {'commit_interval_sec':0},{'commit_interval_sec':3.6},{'commit_interval_sec':float('nan')},
    {'initial_maximum_sec':0},{'initial_maximum_sec':4.0},{'initial_maximum_sec':float('inf')},
    {'commit_interval_sec':1.6,'initial_maximum_sec':.5},
])
def test_invalid_explicit_timing_is_rejected_before_streaming(settings):
    with pytest.raises(ValueError):Segmenter(vad=False,maximum_sec=3.6,**settings)

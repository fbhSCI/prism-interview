"""Source selection captures the requested endpoint, never a fuzzy substitute."""
import json
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from zoom_copilot import audio
from zoom_copilot.config import Preferences


def device(identity, *, loopback=False, name="Headset"):
    return SimpleNamespace(id=identity, name=name, isloopback=loopback)


def soundcard(monkeypatch, devices):
    calls = []
    def enumerate_devices(**kwargs):
        calls.append(kwargs)
        return devices
    # No get_microphone stub: production must not use fuzzy name lookup.
    monkeypatch.setitem(sys.modules, "soundcard", SimpleNamespace(all_microphones=enumerate_devices))
    return calls


def test_source_enumeration_distinguishes_headset_playback_and_microphone(monkeypatch):
    calls = soundcard(monkeypatch, [device("headset-out", loopback=True), device("headset-mic"),
                                  device("monitor-out", loopback=True, name="Monitor")])
    assert audio.speaker_loopbacks() == [("headset-out", "Headset"), ("monitor-out", "Monitor")]
    assert audio.microphone_inputs() == [("headset-mic", "Headset")]
    assert audio.input_devices("microphone") == [("headset-mic", "Headset")]
    assert calls == [{"include_loopback": True}] * 3


@pytest.mark.parametrize("source,selected", [("microphone", "headset-out"), ("loopback", "headset-mic"),
                                           ("microphone", "Headset"), ("loopback", "headset")])
def test_wrong_class_missing_device_or_fuzzy_name_does_not_start_capture(monkeypatch, source, selected):
    soundcard(monkeypatch, [device("headset-out", loopback=True), device("headset-mic")])
    statuses, ready = [], []
    capture = audio.AudioCapture(selected, .005, lambda *args: pytest.fail("Unexpected audio"),
                                 lambda _: None, statuses.append, input_source=source,
                                 on_ready=lambda: ready.append(True))
    capture._run()
    assert capture.error_event.is_set() and not capture.ready_event.is_set() and ready == []
    assert statuses and statuses[-1].startswith("音频设备错误：")
    assert "所选" in statuses[-1]


def test_duplicate_device_id_is_an_explicit_error(monkeypatch):
    soundcard(monkeypatch, [device("ambiguous"), device("ambiguous", name="Other")])
    with pytest.raises(RuntimeError, match="麦克风"):
        audio._selected_input("ambiguous", "microphone")


@pytest.mark.parametrize("source", ["unknown", "both", "", None])
def test_invalid_source_is_rejected_before_any_device_read(monkeypatch, source):
    calls = soundcard(monkeypatch, [])
    with pytest.raises(ValueError):
        audio.AudioCapture("x", .005, lambda *args: None, lambda _: None, lambda _: None,
                           input_source=source)
    with pytest.raises(ValueError):
        audio.input_devices(source)
    assert calls == []


@pytest.mark.parametrize("source,loopback", [("microphone", False), ("loopback", True)])
def test_selected_source_recorder_streams_downmixed_audio_and_flushes_tail(monkeypatch, source, loopback):
    events, statuses, recorder_options = [], [], []
    capture = audio.AudioCapture("chosen", .005, lambda kind, chunk: events.append((kind, chunk)),
                                 lambda _: None, statuses.append, input_source=source,
                                 streaming=True, context=19)
    class Recorder:
        blocks = 0
        def __enter__(self):
            assert not capture.ready_event.is_set()
            return self
        def __exit__(self, *args):
            statuses.append("closed")
        def record(self, numframes):
            self.blocks += 1
            if self.blocks == 3:
                capture.stop()
            return np.tile(np.array([.2, .6], dtype=np.float32), (numframes, 1))
    def recorder(**kwargs):
        recorder_options.append(kwargs)
        return Recorder()
    chosen = device("chosen", loopback=loopback)
    chosen.recorder = recorder
    other = device("wrong", loopback=not loopback)
    other.recorder = lambda **kwargs: pytest.fail("Opened the wrong source")
    soundcard(monkeypatch, [other, chosen])
    original_segmenter = audio.Segmenter
    monkeypatch.setattr(audio, "Segmenter", lambda **kwargs: original_segmenter(vad=False, **kwargs))
    import scipy.signal
    monkeypatch.setattr(scipy.signal, "resample_poly", lambda values, up, down: values[::down])
    capture._run()
    assert capture.ready_event.is_set() and not capture.error_event.is_set()
    assert recorder_options == [{"samplerate": 48000, "blocksize": 4096}]
    chunks = [chunk for kind, chunk in events if kind == "stream"]
    assert len(chunks) == 3
    assert [c.start_sample for c in chunks] == [0, 1600, 3200]
    assert all(c.capture_context == 19 and np.allclose(c, .4) for c in chunks)
    final = [chunk for kind, chunk in events if kind == "final"]
    assert len(final) == 1 and len(final[0]) == 4800 and final[0].commit_end_sample == 4800
    assert final[0].capture_context == 19 and np.allclose(final[0], .4)
    assert statuses[-2:] == ["closed", "音频监听已停止"]
    assert ("麦克风" if source == "microphone" else "会议声音") in statuses[0]


def test_microphone_open_failure_never_announces_ready(monkeypatch):
    selected = device("chosen")
    class BrokenRecorder:
        def __enter__(self): raise RuntimeError("device busy")
        def __exit__(self, *args): pass
    selected.recorder = lambda **kwargs: BrokenRecorder()
    soundcard(monkeypatch, [selected])
    monkeypatch.setattr(audio, "Segmenter", lambda **kwargs: None)
    capture = audio.AudioCapture("chosen", .005, lambda *args: None, lambda _: None, lambda _: None,
                                 input_source="microphone")
    capture._run()
    assert capture.error_event.is_set() and not capture.ready_event.is_set()


def test_source_preferences_roundtrip_and_keep_each_device_choice(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISM_INTERVIEW_HOME", str(tmp_path))
    pref = Preferences(input_source="microphone", microphone_id="headset-mic", speaker_id="monitor-out")
    pref.save()
    loaded = Preferences.load()
    assert (loaded.input_source, loaded.microphone_id, loaded.speaker_id) == ("microphone", "headset-mic", "monitor-out")
    assert not loaded.cloud_consent and not loaded.translation_cloud_consent
    payload = json.loads((tmp_path / "preferences.json").read_text())
    assert "api_key" not in payload and "session_api_keys" not in payload


def test_invalid_source_preferences_are_normalized_without_enabling_cloud():
    pref = Preferences(input_source="not-a-source", microphone_id=None, speaker_id=17)
    assert pref.input_source == "loopback" and pref.microphone_id == pref.speaker_id == ""
    assert not pref.cloud_consent and not pref.translation_cloud_consent


def test_fresh_default_recommends_same_qwen_flash_model_without_consent():
    pref = Preferences()
    assert pref.provider == "qwen" and pref.answer_model == "qwen3.8-flash"
    assert pref.translation_engine == "qwen" and pref.translation_qwen_model == "qwen3.8-flash"
    assert pref.api_base_url == pref.translation_qwen_base_url == "https://dashscope.aliyuncs.com/compatible-mode/v1"
    assert not pref.cloud_consent and not pref.translation_cloud_consent and not pref.api_key


@pytest.mark.parametrize("provider", ["codex", "deepseek", "ollama", "openai_compatible"])
def test_qwen_default_model_and_endpoint_do_not_override_other_explicit_providers(provider):
    pref = Preferences(provider=provider)
    assert pref.provider == provider and pref.answer_model == pref.api_base_url == ""


def test_qwen_region_remains_an_explicit_endpoint_choice():
    pref = Preferences(qwen_region="ap-southeast-1")
    assert pref.api_base_url == "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def test_existing_local_translation_and_model_choices_are_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISM_INTERVIEW_HOME", str(tmp_path))
    (tmp_path / "preferences.json").write_text(json.dumps({"provider": "ollama", "answer_model": "local-choice",
        "translation_engine": "argos", "speaker_id": "old-output"}), encoding="utf8")
    pref = Preferences.load()
    assert pref.provider == "ollama" and pref.answer_model == "local-choice" and pref.translation_engine == "argos"
    assert pref.speaker_id == "old-output" and pref.input_source == "loopback"


def test_source_tag_survives_audio_views_and_independent_recorder_clocks():
    chunk = audio.AudioChunk(np.ones(1600, dtype=np.float32), start_sample=0,
        commit_start_sample=0, commit_end_sample=1600, stream_id="remote-stream", input_source="loopback")
    assert chunk.copy().input_source == chunk[:800].input_source == "loopback"
    local = audio.AudioChunk(np.ones(1600, dtype=np.float32), start_sample=0,
        commit_start_sample=0, commit_end_sample=1600, stream_id="own-stream", input_source="microphone")
    assert local.start_sample == chunk.start_sample and local.stream_id != chunk.stream_id


def eventually(predicate):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline and not predicate():
        time.sleep(.005)
    assert predicate()


def fake_group_capture(monkeypatch, *, fail=None, delayed_microphone=False):
    instances = {}
    permit_microphone = threading.Event()
    class FakeCapture:
        def __init__(self, identity, *, input_source, on_audio, on_level, on_status,
                     on_activity, context, _start_gate, **kwargs):
            self.input_source = input_source
            self.identity, self.context = identity, context
            self.ready_event, self.error_event = threading.Event(), threading.Event()
            self.stop_requested = threading.Event()
            self.thread = None
            self.on_audio, self.on_level, self.on_status = on_audio, on_level, on_status
            self.on_activity, self.gate = on_activity, _start_gate
            self.closed = False
            self.recorded = False
            instances[input_source] = self
        def start(self):
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
        def run(self):
            try:
                if self.input_source == fail:
                    self.error_event.set()
                    self.on_status("音频设备错误：test busy")
                    return
                if delayed_microphone and self.input_source == "microphone":
                    while not self.stop_requested.is_set() and not permit_microphone.wait(.005):
                        pass
                if self.stop_requested.is_set(): return
                self.ready_event.set()
                self.on_status("监听中")
                while not self.stop_requested.is_set() and not self.gate.wait(.005): pass
                if self.stop_requested.is_set(): return
                self.recorded = True
                self.on_level(.2 if self.input_source == "loopback" else .4)
                if self.on_activity:
                    self.on_activity(True)
                self.on_audio("stream", self.chunk())
                self.stop_requested.wait(3)
                self.on_audio("final", self.chunk())
            finally:
                self.closed = True
        def chunk(self):
            return audio.AudioChunk(np.full(1600, 1 if self.input_source == "loopback" else 2, dtype=np.float32),
                start_sample=0, commit_start_sample=0, commit_end_sample=1600,
                stream_id=self.input_source + "-clock", capture_context=self.context,
                input_source=self.input_source)
        def stop(self): self.stop_requested.set()
        def set_context(self, value): self.context = value
    monkeypatch.setattr(audio, "AudioCapture", FakeCapture)
    return instances, permit_microphone


def test_dual_capture_waits_for_both_sources_and_drains_both_without_mixing(monkeypatch):
    children, permit = fake_group_capture(monkeypatch, delayed_microphone=True)
    emitted, ready, levels, activity, statuses = [], [], [], [], []
    group = audio.DualAudioCapture("output", "own-mic", .005,
        lambda kind, chunk: emitted.append((kind, chunk)), levels.append, statuses.append,
        activity.append, on_ready=lambda: ready.append(True), context=7)
    group.start()
    eventually(lambda: "loopback" in children and children["loopback"].ready_event.is_set())
    assert not group.ready_event.is_set() and not children["loopback"].recorded
    assert not emitted and not ready
    permit.set()
    eventually(lambda: len(emitted) == 2)
    assert group.ready_event.is_set() and ready == [True]
    assert {chunk.input_source for _, chunk in emitted} == {"loopback", "microphone"}
    assert {chunk.stream_id for _, chunk in emitted} == {"loopback-clock", "microphone-clock"}
    assert all(chunk.capture_context == 7 for _, chunk in emitted)
    assert activity == [True]  # microphone speech never resets remote question timing
    assert max(levels) == .4
    group.set_context(8)
    group.stop()
    group.thread.join(2)
    assert not group.thread.is_alive() and all(child.closed for child in children.values())
    finals = [chunk for kind, chunk in emitted if kind == "final"]
    assert len(finals) == 2 and all(chunk.capture_context == 8 for chunk in finals)
    assert not group.error_event.is_set()


@pytest.mark.parametrize("failure", ["loopback", "microphone"])
def test_dual_source_failure_stops_the_other_without_claiming_partial_ready(monkeypatch, failure):
    children, _ = fake_group_capture(monkeypatch, fail=failure)
    ready, emitted, statuses = [], [], []
    group = audio.DualAudioCapture("output", "own-mic", .005,
        lambda *args: emitted.append(args), lambda _: None, statuses.append,
        on_ready=lambda: ready.append(True))
    group.start()
    group.thread.join(2)
    assert not group.thread.is_alive() and group.error_event.is_set()
    assert not group.ready_event.is_set() and not ready and not emitted
    assert all(child.closed for child in children.values())
    assert any("双方监听已停止" in status for status in statuses)


def test_dual_preferences_preserve_each_explicit_device():
    pref = Preferences(input_source="both", speaker_id="meeting-headset", microphone_id="own-headset")
    assert pref.input_source == "both"
    assert pref.speaker_id == "meeting-headset" and pref.microphone_id == "own-headset"

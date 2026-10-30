"""Video-demo launch contracts; no audio capture, cloud call or model inference."""
from __future__ import annotations

from dataclasses import replace
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from zoom_copilot import app as controller_module
from zoom_copilot import instance_guard as instance_module
from zoom_copilot.config import Preferences
from zoom_copilot.ui import CopilotWindow


@pytest.fixture
def launch_harness(monkeypatch, tmp_path):
    state = SimpleNamespace(tasks=[], calls=[], controllers=[], windows=[],
                            speaker_reads=0, images=[], application=None,
                            default_output=SimpleNamespace(id="default-output"),
                            combo_id="default-output", empty_device_reads=0,
                            device_reads=0, statuses=[], listen_times=[], now=0, serial=0,
                            guard_calls=[], instance_primary=True, guard_error=None)
    preferences = Preferences(auto_answer=True, speaker_id="previous-output",
                              provider="codex", cloud_consent=True)
    monkeypatch.setattr(controller_module, "load_runtime", lambda: None)
    monkeypatch.setattr(controller_module, "app_dir", lambda: tmp_path / "isolated-launch")
    monkeypatch.setattr(Preferences, "load", classmethod(lambda cls: replace(preferences)))

    # This fixture replaces QApplication and its event loop. Keep IPC at the
    # same dependency boundary; real native/process lock behavior is exercised
    # in test_instance_guard_030, not inside this scheduling-only fake app.
    class InstanceGuard:
        def __init__(self, directory, parent):
            state.guard_calls.append(("create", directory))
        def acquire_or_notify(self, command):
            state.guard_calls.append(("acquire", command))
            if state.guard_error:
                raise RuntimeError(state.guard_error)
            return state.instance_primary
        def bind_window(self, window, choose_model):
            state.guard_calls.append(("bind", window))
        def close(self):
            state.guard_calls.append(("close",))
    monkeypatch.setattr(instance_module, "InstanceGuard", InstanceGuard)

    def default_speaker():
        state.speaker_reads += 1
        return state.default_output
    monkeypatch.setitem(sys.modules, "soundcard", SimpleNamespace(default_speaker=default_speaker))

    class Application:
        @staticmethod
        def instance():
            return None
        def __init__(self, argv):
            state.application = self
            self.aboutToQuit = SimpleNamespace(connect=lambda callback: None)
        def setApplicationName(self, name):
            pass
        def setFont(self, font):
            pass
        def quit(self):
            state.calls.append("quit")
        def exec(self):
            count = 0
            while state.tasks:
                state.tasks.sort(key=lambda row: (row[0], row[1]))
                deadline, _, callback = state.tasks.pop(0)
                state.now = deadline
                callback()
                count += 1
                assert count < 200, "Unbounded launch retry"
            return 17

    class Window:
        def __init__(self):
            state.windows.append(self)
        def setWindowTitle(self, title):
            self.title = title
        def show(self):
            state.calls.append("show")
        def get_settings(self):
            state.device_reads += 1
            return {"speaker_id": "" if state.device_reads <= state.empty_device_reads else state.combo_id}
        def set_status(self, message, level="info"):
            state.statuses.append((message, level))
        def grab(self):
            return SimpleNamespace(save=state.images.append)

    class Controller:
        def __init__(self, window, prefs=None):
            self.received_preferences = prefs
            self.pref = prefs or Preferences.load()
            state.controllers.append(self)
        def start_listening(self):
            state.calls.append("start-local-listening")
            state.listen_times.append(state.now)
        def check_models(self):
            state.calls.append("check-catalog")
        def choose_answer_model(self):
            state.calls.append("choose-model")
        def request_answer(self, *args):
            raise AssertionError("Video-demo startup must not request an answer")
        def close(self):
            pass

    monkeypatch.setattr(controller_module, "QApplication", Application)
    monkeypatch.setattr(controller_module, "QFont", lambda *args: args)
    monkeypatch.setattr(controller_module, "CopilotWindow", Window)
    monkeypatch.setattr(controller_module, "CopilotController", Controller)
    def schedule(delay, callback):
        state.serial += 1
        state.tasks.append((state.now + delay, state.serial, callback))
    monkeypatch.setattr(controller_module.QTimer, "singleShot", schedule)
    return state, preferences


def test_normal_launch_prewarms_authorized_codex_without_recording(launch_harness, monkeypatch):
    state, saved = launch_harness
    saved.provider, saved.cloud_consent = "codex", True
    monkeypatch.setattr(sys, "argv", ["PrismInterview"])
    assert controller_module.main() == 17
    assert state.calls == ["show", "check-catalog"]
    assert state.speaker_reads == 0
    assert [entry[0] for entry in state.guard_calls] == ["create", "acquire", "bind", "close"]


@pytest.mark.parametrize("provider,consent,expected_catalog", [
    ("codex", True, True), ("codex", False, False),
    ("ollama", True, False), ("openai_compatible", True, False),
])
def test_video_demo_overrides_saved_auto_mode_and_only_schedules_listen_catalog(
        launch_harness, monkeypatch, provider, consent, expected_catalog):
    state, saved = launch_harness
    saved.provider, saved.cloud_consent = provider, consent
    monkeypatch.setattr(sys, "argv", ["PrismInterview", "--video-demo"])
    assert controller_module.main() == 17
    prefs = state.controllers[0].received_preferences
    assert prefs.auto_answer is False
    assert prefs.speaker_id == "default-output"
    assert prefs.provider == provider and prefs.cloud_consent == consent
    assert state.speaker_reads == 1
    assert state.calls.count("start-local-listening") == 1
    assert state.calls.count("check-catalog") == int(expected_catalog)
    assert set(state.calls) <= {"show", "start-local-listening", "check-catalog"}
    assert "视频听译体验" in state.windows[0].title


def test_smoke_combined_with_demo_never_reads_speaker_or_schedules_capture(
        launch_harness, monkeypatch, tmp_path):
    state, _ = launch_harness
    output = tmp_path / "smoke.png"
    monkeypatch.setattr(sys, "argv", ["PrismInterview", "--video-demo", "--smoke", str(output)])
    assert controller_module.main() == 17
    assert state.controllers[0].received_preferences is None
    assert state.speaker_reads == 0
    assert "start-local-listening" not in state.calls
    assert "check-catalog" not in state.calls
    assert state.images == [str(output)]
    assert state.guard_calls == []


@pytest.mark.parametrize("args,command", [([], "activate"), (["--choose-model"], "choose-model"),
                                         (["--video-demo"], "activate")])
def test_duplicate_launch_routes_without_constructing_or_mutating_session(launch_harness, monkeypatch, args, command):
    state, _ = launch_harness
    state.instance_primary = False
    monkeypatch.setattr(sys, "argv", ["PrismInterview", *args])
    assert controller_module.main() == 0
    assert state.guard_calls[1] == ("acquire", command)
    assert state.guard_calls[-1] == ("close",)
    assert state.windows == [] and state.controllers == [] and state.tasks == []
    assert state.speaker_reads == 0 and state.calls == []


def test_window_initialization_failure_releases_guard_before_outer_error_dialog(launch_harness, monkeypatch):
    state, _ = launch_harness
    def fail_window():
        raise ValueError("simulated construction failure")
    monkeypatch.setattr(controller_module, "CopilotWindow", fail_window)
    monkeypatch.setattr(sys, "argv", ["PrismInterview"])
    with pytest.raises(ValueError, match="simulated construction failure"):
        controller_module.main()
    assert state.guard_calls[-1] == ("close",)
    assert state.controllers == []


def test_unresponsive_existing_instance_returns_failure_without_starting_controller(launch_harness, monkeypatch):
    state, _ = launch_harness
    state.guard_error = "unresponsive owner"
    messages = []
    monkeypatch.setattr(controller_module.QMessageBox, "warning", lambda *args: messages.append(args[-1]))
    monkeypatch.setattr(sys, "argv", ["PrismInterview"])
    assert controller_module.main() == 1
    assert messages == ["unresponsive owner"]
    assert state.guard_calls[-1] == ("close",)
    assert state.windows == [] and state.controllers == []


def test_demo_with_no_default_speaker_stays_open_and_bounds_device_retries(
        launch_harness, monkeypatch):
    state, _ = launch_harness
    state.default_output = None
    state.combo_id = ""
    monkeypatch.setattr(sys, "argv", ["PrismInterview", "--video-demo"])
    assert controller_module.main() == 17
    assert state.controllers[0].pref.auto_answer is False
    assert state.speaker_reads == 1
    assert state.device_reads == 41
    assert state.now == 10400
    assert "start-local-listening" not in state.calls
    assert len(state.statuses) == 1
    assert "点击开始监听" in state.statuses[0][0]
    assert state.statuses[0][1] == "warning"


def test_demo_waits_for_async_device_selection_before_starting(
        launch_harness, monkeypatch):
    state, _ = launch_harness
    state.empty_device_reads = 3
    monkeypatch.setattr(sys, "argv", ["PrismInterview", "--video-demo"])
    assert controller_module.main() == 17
    assert state.device_reads == 4
    assert state.listen_times == [1150]
    assert state.calls.count("start-local-listening") == 1
    assert state.statuses == []


def test_manual_demo_preferences_cannot_trigger_answer_after_recognized_question(monkeypatch, tmp_path):
    qt = QApplication.instance() or QApplication([])
    monkeypatch.setenv("PRISM_INTERVIEW_HOME", str(tmp_path / "data"))
    queued_work = []
    monkeypatch.setattr(controller_module.QTimer, "singleShot", lambda *args: None)
    monkeypatch.setattr(controller_module.CopilotController, "background",
                        lambda self, operation, name="PrismWorker": queued_work.append(name))

    class ForbiddenAnswerService:
        def __init__(self, preferences):
            self.preferences = preferences
        def answer(self, *args, **kwargs):
            raise AssertionError("Manual listening must not run answer inference")
        def close(self):
            pass
    monkeypatch.setattr(controller_module, "AnswerService", ForbiddenAnswerService)
    monkeypatch.setattr(controller_module, "KnowledgeBase", lambda *args: SimpleNamespace(
        document_names=[], document_records={}, passages=[]))
    window = CopilotWindow()
    prefs = Preferences(auto_answer=False, provider="codex", cloud_consent=True,
                        model_cache_dir=str(tmp_path / "models"))
    ctrl = controller_module.CopilotController(window, prefs)
    ctrl.timer.stop()
    ctrl.auto_timer.stop()
    try:
        ctrl.listening = True
        ctrl.on_transcript("final", "How would you estimate calibration uncertainty?")
        ctrl.assembler.last_text = ctrl.assembler.last_voice = 0
        ctrl.auto_answer()
        assert not ctrl.busy and not ctrl._requests and queued_work == []
        assert window.current_question() == "How would you estimate calibration uncertainty?"
    finally:
        ctrl.close()
        window.close()
        window.deleteLater()
        qt.processEvents()

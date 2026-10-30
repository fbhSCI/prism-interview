"""Real local IPC / process lifecycle regressions, without audio or credentials."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from PySide6.QtCore import QCoreApplication
from PySide6.QtNetwork import QLocalSocket
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QWidget

from zoom_copilot.instance_guard import (
    InstanceGuard, activate_window, instance_identity, interactive_instance_required,
)


WORKER = r'''
import json, sys, time
from pathlib import Path
from PySide6.QtCore import QCoreApplication, QTimer
from zoom_copilot.instance_guard import InstanceGuard
app = QCoreApplication([])
folder, state, command, lifetime, blocked = sys.argv[1:]
guard = InstanceGuard(Path(folder))
owned = guard.acquire_or_notify(command)
Path(state).write_text(json.dumps({'primary':owned, 'socket':guard.server_name}))
if not owned:
    raise SystemExit(0)
def record(command):
    with Path(state + '.commands').open('a') as f:
        f.write(command + '\n')
guard.command_received.connect(record)
if blocked == 'yes':
    time.sleep(int(lifetime)/1000)
else:
    if blocked == 'delay':
        time.sleep(.7)
    QTimer.singleShot(int(lifetime), app.quit)
    app.exec()
guard.close()
'''


@pytest.fixture
def processes(tmp_path):
    worker = tmp_path / "guard_worker.py"
    worker.write_text(WORKER, encoding="utf-8")
    started = []
    env = dict(os.environ, PYTHONUTF8="1")
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])

    def start(folder, state_name, command="activate", lifetime=12000, blocked=False):
        state = tmp_path / state_name
        proc = subprocess.Popen([sys.executable, str(worker), str(folder), str(state), command,
                                 str(lifetime), "delay" if blocked == "delay" else ("yes" if blocked else "no")],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
        started.append(proc)
        return proc, state

    yield start
    for proc in started:
        if proc.poll() is None:
            proc.terminate()
        proc.communicate(timeout=6)


def read_state(proc, path, timeout=6):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except json.JSONDecodeError:
                pass
        if proc.poll() is not None:
            out, err = proc.communicate()
            raise AssertionError((proc.returncode, out, err))
        time.sleep(.02)
    raise AssertionError("instance process did not become ready")


def test_duplicate_routes_activation_and_model_chooser_to_existing_owner(processes, tmp_path):
    folder = tmp_path / "session"
    owner, state = processes(folder, "owner.json")
    assert read_state(owner, state)["primary"]
    for command in ("activate", "choose-model"):
        duplicate, routed = processes(folder, command + ".json", command)
        assert read_state(duplicate, routed)["primary"] is False
        duplicate.communicate(timeout=3)
        assert duplicate.returncode == 0
    assert Path(str(state) + ".commands").read_text().splitlines() == ["activate", "choose-model"]
    assert owner.poll() is None


def test_concurrent_start_has_exactly_one_owner(processes, tmp_path):
    first, a = processes(tmp_path / "shared", "first.json")
    second, b = processes(tmp_path / "shared", "second.json")
    assert sorted([read_state(first, a)["primary"], read_state(second, b)["primary"]]) == [False, True]


def test_slow_startup_routes_model_chooser_once_without_duplicate_retries(processes, tmp_path):
    folder = tmp_path / "starting"
    owner, state = processes(folder, "slow.json", blocked="delay")
    assert read_state(owner, state)["primary"]
    duplicate, routed = processes(folder, "chooser.json", "choose-model")
    assert read_state(duplicate, routed)["primary"] is False
    assert Path(str(state) + ".commands").read_text().splitlines() == ["choose-model"]


def test_dead_owner_lock_is_recovered_and_data_directories_are_independent(processes, tmp_path):
    owner, state = processes(tmp_path / "one", "owner.json")
    assert read_state(owner, state)["primary"]
    other, other_state = processes(tmp_path / "two", "other.json")
    assert read_state(other, other_state)["primary"]
    owner.kill()
    owner.communicate(timeout=3)
    replacement, replaced = processes(tmp_path / "one", "replacement.json")
    assert read_state(replacement, replaced)["primary"]
    assert other.poll() is None


def test_live_unresponsive_owner_never_starts_second_session(processes, tmp_path):
    owner, state = processes(tmp_path / "blocked", "blocked.json", blocked=True)
    assert read_state(owner, state)["primary"]
    application = QApplication.instance() or QApplication([])
    contender = InstanceGuard(tmp_path / "blocked")
    try:
        with pytest.raises(RuntimeError, match="没有再启动"):
            contender.acquire_or_notify(timeout_ms=150)
        assert not contender._owned
        assert owner.poll() is None
    finally:
        contender.close()


def test_socket_ignores_unknown_oversized_and_multiple_commands(processes, tmp_path):
    owner, state = processes(tmp_path / "strict", "strict.json")
    endpoint = read_state(owner, state)["socket"]
    application = QApplication.instance() or QApplication([])
    for payload in (b"start-capture\n", b"activate\nchoose-model\n", b"x" * 100):
        socket = QLocalSocket()
        socket.connectToServer(endpoint)
        assert socket.waitForConnected(1000)
        socket.write(payload)
        if socket.bytesToWrite():
            socket.waitForBytesWritten(1000)
        socket.waitForReadyRead(300)
        assert bytes(socket.readAll()) != b"ok\n"
        socket.abort()
    assert not Path(str(state) + ".commands").exists()


def test_window_restore_keeps_existing_session_and_diagnostics_bypass_guard(tmp_path):
    application = QApplication.instance() or QApplication([])
    window = QWidget()
    window.setProperty("session-test", "untouched")
    try:
        window.showMinimized()
        QTest.qWait(10)
        activate_window(window)
        QTest.qWait(10)
        assert window.isVisible() and not window.isMinimized()
        assert window.property("session-test") == "untouched"
        window.showMaximized()
        activate_window(window)
        assert window.isMaximized()
    finally:
        window.close()
    options = dict(smoke=None, self_check=None, verify_runtime=None)
    assert interactive_instance_required(SimpleNamespace(**options))
    for mode in options:
        assert not interactive_instance_required(SimpleNamespace(**(options | {mode: "test"})))
    assert instance_identity(tmp_path) == instance_identity(tmp_path / ".")
    if os.name == "nt":
        assert instance_identity(tmp_path) == instance_identity(Path(str(tmp_path).upper()))

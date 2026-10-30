"""One interactive application per data directory, without sharing user content.

The lock decides ownership atomically. The user-local socket carries only a
fixed activation command; it cannot request capture, inference or file access.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import time
from typing import Callable

from PySide6.QtCore import QObject, QLockFile, QTimer, Signal
from PySide6.QtNetwork import QLocalServer, QLocalSocket


_COMMANDS = {b"activate", b"choose-model"}
_MAX_REQUEST = 32
_UNRESPONSIVE = "此资料库的助手已在运行，但窗口暂未响应。请等待当前窗口恢复后重试；为避免重复占用显卡，本次没有再启动助手。"


def instance_identity(data_dir: Path) -> str:
    """Aliases resolving to the same directory share the same IPC endpoint."""
    canonical = os.path.normcase(str(Path(data_dir).expanduser().resolve()))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def interactive_instance_required(args) -> bool:
    """Diagnostic runs must remain usable alongside an interactive session."""
    return not (args.smoke or args.self_check or args.verify_runtime)


def activate_window(window) -> None:
    """Restore an existing Qt window without changing its session/settings."""
    if window.isMinimized():
        window.showNormal()
    else:
        window.show()
    window.raise_()
    window.activateWindow()


class InstanceGuard(QObject):
    command_received = Signal(str)

    def __init__(self, data_dir: Path, parent: QObject | None = None):
        super().__init__(parent)
        directory = Path(data_dir).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        self.server_name = "PrismInterview-" + instance_identity(directory)
        self._lock = QLockFile(str(directory / ".prism-instance.lock"))
        # A long-running healthy interview must never become stale by age.
        # Qt still detects a dead owner PID and recovers its lock automatically.
        self._lock.setStaleLockTime(0)
        self._server = QLocalServer(self)
        self._server.setSocketOptions(QLocalServer.SocketOption.UserAccessOption)
        self._server.newConnection.connect(self._accept_connections)
        self._owned = False
        self._connections = set()

    def acquire_or_notify(self, command: str = "activate", timeout_ms: int = 4000) -> bool:
        """Return True for owner, False after acknowledged existing-window request.

        If an alive owner does not respond, fail closed: do not create another
        controller, reload settings or compete for the GPU.
        """
        payload = command.encode("ascii")
        if payload not in _COMMANDS:
            raise ValueError("Unsupported instance request")
        if self._owned:
            return True
        deadline = time.monotonic() + max(0, timeout_ms) / 1000
        while True:
            if self._lock.tryLock(0):
                self._owned = True
                # Only the lock owner may remove an abandoned socket endpoint.
                QLocalServer.removeServer(self.server_name)
                if not self._server.listen(self.server_name):
                    message = self._server.errorString()
                    self.close()
                    raise RuntimeError("无法建立助手窗口通信：" + message)
                return True
            if self._lock.error() == QLockFile.LockError.PermissionError:
                raise RuntimeError("无法在助手数据目录创建运行锁，请检查该目录的写入权限。")
            remaining = max(0, int((deadline - time.monotonic()) * 1000))
            notified = self._notify(payload, remaining)
            if notified is True:
                return False
            if notified is False or remaining <= 0:
                # Once sent, never re-send a chooser request on an ACK timeout:
                # a busy owner may still dispatch the first request later.
                raise RuntimeError(_UNRESPONSIVE)
            # Covers the short interval after the owner obtains its lock and
            # before it starts listening, and an owner dying during startup.
            time.sleep(min(.04, remaining / 1000))

    def _notify(self, payload: bytes, timeout_ms: int) -> bool | None:
        socket = QLocalSocket()
        deadline = time.monotonic() + timeout_ms / 1000
        def remaining():
            return max(0, int((deadline - time.monotonic()) * 1000))
        try:
            socket.connectToServer(self.server_name)
            if not socket.waitForConnected(min(300, remaining())):
                return None
            socket.write(payload + b"\n")
            if socket.bytesToWrite() and not socket.waitForBytesWritten(remaining()):
                return False
            reply = b""
            while len(reply) < 3:
                if not socket.bytesAvailable() and not socket.waitForReadyRead(remaining()):
                    return False
                reply += bytes(socket.readAll())
            return reply == b"ok\n"
        finally:
            socket.abort()

    def _accept_connections(self):
        while self._server.hasPendingConnections():
            socket = self._server.nextPendingConnection()
            self._connections.add(socket)
            socket.setReadBufferSize(_MAX_REQUEST + 1)
            timer = QTimer(socket)
            timer.setSingleShot(True)
            timer.timeout.connect(socket.abort)
            timer.start(1500)
            socket.readyRead.connect(lambda s=socket: self._read_command(s))
            socket.disconnected.connect(lambda s=socket: self._discard(s))
            if socket.bytesAvailable():
                self._read_command(socket)

    def _read_command(self, socket):
        if socket not in self._connections:
            return
        # Exactly one tiny ASCII line is accepted. Unknown commands never emit.
        available = socket.bytesAvailable()
        if available > _MAX_REQUEST:
            socket.abort()
            return
        if not socket.canReadLine():
            return
        payload = bytes(socket.readAll())
        if payload.endswith(b"\n") and payload[:-1] in _COMMANDS:
            self.command_received.emit(payload[:-1].decode("ascii"))
            socket.write(b"ok\n")
            socket.flush()
            socket.disconnectFromServer()
        else:
            socket.abort()

    def _discard(self, socket):
        self._connections.discard(socket)
        socket.deleteLater()

    def bind_window(self, window, choose_model: Callable[[], None]):
        def dispatch(command):
            activate_window(window)
            if command == "choose-model":
                # Do not enter a modal chooser before acknowledging the sender.
                QTimer.singleShot(0, choose_model)
        self.command_received.connect(dispatch)

    def close(self):
        for socket in tuple(self._connections):
            socket.abort()
        self._connections.clear()
        self._server.close()
        if self._owned:
            self._lock.unlock()
            self._owned = False

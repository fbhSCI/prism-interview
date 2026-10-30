"""Explicit public speech-model preparation, independent of capture or AI APIs."""
from __future__ import annotations

from pathlib import Path
import threading

from PySide6.QtCore import QObject, Signal, Slot, Qt, QTimer
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QPlainTextEdit, QPushButton, QProgressBar)

from .runtime import default_model_cache
from .setup_models import prepare


class _Signals(QObject):
    progress = Signal(int, str)
    complete = Signal(int, bool)


class ModelSetupDialog(QDialog):
    """One nonmodal task per window; closing an active task only hides it."""
    prepared = Signal(str)

    def __init__(self, parent, preferences):
        super().__init__(parent)
        self.setWindowTitle("准备本机语音模型 · Prism Interview")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)
        self.setMinimumSize(620, 390)
        self._active = False
        self._generation = 0
        self.thread: threading.Thread | None = None
        self._signals = _Signals(self)
        self._signals.progress.connect(self._progress, Qt.ConnectionType.QueuedConnection)
        self._signals.complete.connect(self._complete, Qt.ConnectionType.QueuedConnection)
        layout = QVBoxLayout(self)
        self.description = QLabel("准备本机英语识别、实时英文草稿和标点模型。\n只下载公开模型，不会监听声音、上传资料或调用付费 AI。")
        self.description.setWordWrap(True)
        layout.addWidget(self.description)
        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        self.path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.path_label)
        self.status_label = QLabel("点击开始准备，首次下载可能需要一些时间。")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(200)
        layout.addWidget(self.log, 1)
        buttons = QHBoxLayout()
        self.start_button = QPushButton("开始准备")
        self.start_button.clicked.connect(self.start)
        buttons.addWidget(self.start_button)
        buttons.addStretch()
        self.close_button = QPushButton("关闭")
        self.close_button.clicked.connect(self.close)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)
        self.configure(preferences)

    @property
    def active(self):
        return self._active

    def configure(self, preferences):
        if self._active:
            return
        # Read only model choices. Do not retain/repr a preference object that
        # can contain an in-memory provider credential.
        selected = getattr(preferences, "model_cache_dir", "")
        self.model_root = Path(selected).expanduser().resolve() if selected else default_model_cache()
        self.asr_model = str(getattr(preferences, "whisper_size", "large-v3-turbo"))
        self.streaming = getattr(preferences, "streaming_captions", True) is True
        self.path_label.setText("模型目录：" + str(self.model_root))

    @Slot()
    def start(self):
        if self._active:
            return
        self._generation += 1
        generation = self._generation
        self._active = True
        self.start_button.setEnabled(False)
        self.close_button.setText("后台继续")
        self.status_label.setText("正在检查／下载本机语音模型…")
        self.progress_bar.setRange(0, 0)
        self.log.clear()
        root, model, streaming = self.model_root, self.asr_model, self.streaming
        signals = self._signals

        def send(name, *values):
            try:
                getattr(signals, name).emit(*values)
            except RuntimeError:
                # Main-window destruction may delete the C++ signal object;
                # public-file work must never access a deleted widget.
                pass

        def worker():
            try:
                prepare(root, asr=model, translation=False, streaming=streaming,
                        punctuation=True,
                        on_status=lambda message: send("progress", generation, str(message)[:1000]))
            except Exception:
                # Network errors can contain request text/headers. A fixed
                # message is enough to retry; never display arbitrary bodies.
                send("complete", generation, False)
            else:
                send("complete", generation, True)

        self.thread = threading.Thread(target=worker, name="PrismPrepareSpeechModels", daemon=True)
        self.thread.start()

    @Slot(int, str)
    def _progress(self, generation, message):
        if generation != self._generation or not self._active:
            return
        self.log.appendPlainText(message)

    @Slot(int, bool)
    def _complete(self, generation, success):
        if generation != self._generation or not self._active:
            return
        self._active = False
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1 if success else 0)
        self.start_button.setEnabled(True)
        self.start_button.setText("重新检查" if success else "重试")
        self.close_button.setText("关闭")
        message = ("本机语音模型已准备好。返回工作台选择输入设备，再开始监听。" if success else
                   "准备失败。请检查网络、模型目录和可用空间，然后点击重试。")
        self.status_label.setText(message)
        self.log.appendPlainText(message)
        if success:
            self.prepared.emit(str(self.model_root))

    def closeEvent(self, event):
        if self._active:
            event.ignore()
            self.hide()
        else:
            super().closeEvent(event)


def show_model_setup(window, preferences, *, auto_start=True):
    """Explicit button handler. Reopen an existing task without starting a second."""
    dialog = getattr(window, "_model_setup_dialog", None)
    try:
        if dialog is not None:
            dialog.configure(preferences)
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
    except RuntimeError:
        dialog = None
    if dialog is None:
        dialog = ModelSetupDialog(window, preferences)
        window._model_setup_dialog = dialog
        dialog.show()
    if auto_start and not dialog.active:
        QTimer.singleShot(0, dialog.start)
    return dialog

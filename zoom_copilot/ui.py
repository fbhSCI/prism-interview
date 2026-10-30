"""Prism's Qt presentation layer. It performs no audio, model, or network work.

Controller contract: all public methods are called on the Qt main thread. Streamed
answers are *cumulative*, not deltas. API keys returned by ``get_settings`` are
session-only; the controller must exclude them when persisting preferences.

Run ``python -m zoom_copilot.ui --smoke path.png`` for a deterministic, clearly
labelled demonstration screenshot, or omit --smoke for the empty application.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
import re
from typing import Any

from PySide6.QtCore import Qt, Signal, QSize, QTimer, QRectF, QPointF, QPoint, QSignalBlocker, QEvent
from PySide6.QtGui import QColor, QFont, QFontDatabase, QIcon, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QDockWidget, QDoubleSpinBox, QFileDialog,
    QFormLayout, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMainWindow, QMenu, QMessageBox, QDialogButtonBox, QPlainTextEdit, QProgressBar, QPushButton,
    QScrollArea, QSizePolicy, QSpinBox, QSplitter, QStackedWidget, QTableWidget,
    QTableWidgetItem, QToolButton, QVBoxLayout, QWidget,
)


NAVY = "#0B1220"
INK = "#E8EFF8"
MUTED = "#8EA2BA"
TEAL = "#65DCC5"
FONT = "Microsoft YaHei UI" if sys.platform == "win32" else "DejaVu Sans"

# Shared product palette: deep navy foundations, quiet layered surfaces and
# high-contrast text. Rendering and scroll logic remain independent of theme.
STYLE = """
QMainWindow, QWidget#canvas { background: #0B1220; color: #E8EFF8; }
QWidget { font-family: "Segoe UI", "Microsoft YaHei UI"; font-size: 13px; color: #DDE7F4; }
QWidget#sidebar { background: #080E19; border-right: 1px solid #1C293B; }
QWidget#sidebar QLabel { color: #98AAC0; }
QWidget#sidebar QLabel#brand { color: #F4F8FF; }
QWidget#sidebar QLabel#brandSub { color: #65DCC5; }
QLabel#brand { color: #F4F8FF; font-size: 27px; font-weight: 700; letter-spacing: 1px; }
QLabel#brandSub { color: #65DCC5; font-size: 10px; letter-spacing: 3px; }
QLabel#navSection { color: #7589A2; font-size: 10px; font-weight: 600; letter-spacing: 2px; }
QLabel#navFoot { color: #7D91AA; font-size: 11px; line-height: 1.6; }
QFrame#sideContext { background: #101C2D; border: 1px solid #21334A; border-radius: 12px; }
QPushButton#navButton { color: #9AAEC8; background: transparent; border: 1px solid transparent; border-radius: 9px;
    text-align: left; padding: 13px 13px; font-size: 13px; }
QPushButton#navButton:hover { background: #131F30; color: #F4F8FF; }
QPushButton#navButton:checked { background: #123436; border: 1px solid #23524F; color: #93F3DE; font-weight: 600; }
QLabel#eyebrow { color: #788DA8; font-size: 10px; font-weight: 600; letter-spacing: 2px; }
QLabel#pageTitle { font-size: 27px; font-weight: 650; color: #F2F6FE; }
QLabel#subtitle, QLabel#muted { color: #90A4BE; font-size: 12px; }
QLabel#cardTitle { color: #E5EFFB; font-size: 15px; font-weight: 650; }
QLabel#sectionEyebrow { color: #65DCC5; font-size: 10px; font-weight: 700; letter-spacing: 1.3px; }
QLabel#badge { background: #113632; color: #8AF1D8; border: 1px solid #1E5149; padding: 5px 10px; border-radius: 10px; font-size: 11px; }
QLabel#neutralBadge { background: #1C2A3F; color: #9CB4D2; padding: 4px 8px; border-radius: 8px; font-size: 10px; }
QFrame#card { background: #111D2E; border: 1px solid #26374F; border-radius: 13px; }
QFrame#captureBar { background: #111D2E; border: 1px solid #26374F; border-radius: 12px; }
QFrame#divider { background: #26364C; border: 0; max-height: 1px; min-height: 1px; }
QFrame#transcriptItem { background: #162337; border: 1px solid #293B54; border-radius: 10px; }
QFrame#partialItem { background: #112D32; border: 1px dashed #39756D; border-radius: 10px; }
QLabel#transcriptEn { font-size: 17px; color: #E8F0FC; line-height: 1.5; }
QLabel#transcriptZh { font-size: 16px; color: #A9C3D8; line-height: 1.5; }
QLabel#time { color: #7D93AE; font-size: 10px; }
QLabel#questionZh { color: #A0CECB; font-size: 13px; padding: 8px 10px;
    background: #153131; border-radius: 7px; }
QFrame#answerQuestionCard { background: #132D32; border: 1px solid #28504E; border-radius: 9px; }
QFrame#answerQuestionBackgroundCard { background: #142134; border: 1px solid #26374F; border-radius: 9px; }
QLabel#answerQuestionEn { color: #E3F8F5; font-size: 15px; font-weight: 600; padding: 0; }
QLabel#answerQuestionZh { color: #A3D3CB; font-size: 14px; padding: 0; }
QLabel#emptyTitle { font-size: 18px; font-weight: 600; color: #D5E3F5; }
QLabel#emptyBody { color: #90A4BE; font-size: 13px; }
QLabel#note { background: #142337; color: #9DB4CE; border: 1px solid #26374F; border-radius: 9px; padding: 11px 13px; font-size: 12px; }
QPushButton, QToolButton { background: #1A2B42; border: 1px solid #334862;
    border-radius: 7px; padding: 8px 12px; color: #C2D4EA; font-weight: 500; }
QPushButton:hover, QToolButton:hover { background: #223851; border-color: #567894; color: #F4F8FF; }
QPushButton:pressed, QToolButton:pressed { background: #2A455B; }
QPushButton:disabled, QToolButton:disabled { background: #152031; color: #5F748F; border-color: #26364B; }
QPushButton#primary { color: #072321; background: #65DCC5; border: 1px solid #65DCC5; padding: 10px 17px; font-weight: 700; }
QPushButton#primary:hover { background: #8BEBD7; border-color: #8BEBD7; }
QPushButton#primary:disabled { color: #6B9A92; background: #234941; border-color: #234941; }
QPushButton#quiet, QToolButton#quiet { background: transparent; border-color: transparent; color: #9BB2CC; padding: 5px 7px; }
QPushButton#quiet:hover, QToolButton#quiet:hover { background: #1C3543; color: #8AF1D8; }
QPushButton#danger { color: #F0A5A8; border-color: #70414F; }
QPushButton#phaseButton { padding: 5px 10px; color: #9DB2CD; background: #18263B; }
QPushButton#phaseButton:checked { color: #96F0DB; background: #153D38; border-color: #377A6D; font-weight: 600; }
QLineEdit, QPlainTextEdit, QComboBox, QDoubleSpinBox, QSpinBox { background: #0E1929; border: 1px solid #30445E;
    border-radius: 7px; padding: 8px 10px; color: #DCE9F8; selection-background-color: #285E65; }
QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QDoubleSpinBox:focus, QSpinBox:focus { border-color: #65DCC5; background: #132236; }
QLineEdit:disabled, QComboBox:disabled, QDoubleSpinBox:disabled { color: #6D819C; background: #111B2B; border-color: #25344A; }
QComboBox { padding-right: 24px; min-height: 18px; }
QComboBox::drop-down { border: 0; width: 23px; }
QComboBox QAbstractItemView { background: #16273C; color: #DFEAF7; selection-background-color: #28544F; selection-color: #E8FFF7; border: 1px solid #405775; padding: 4px; }
QCheckBox { color: #A3B6CD; spacing: 7px; font-size: 12px; }
QCheckBox::indicator { width: 15px; height: 15px; }
QScrollArea { background: transparent; border: 0; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 2px; }
QScrollBar::handle:vertical { background: #405672; border-radius: 3px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: #65829E; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
QScrollBar:horizontal { height: 8px; background: transparent; }
QScrollBar::handle:horizontal { background: #405672; border-radius: 3px; min-width: 24px; }
QSplitter::handle { background: transparent; width: 14px; }
QMainWindow#readingWorkspace { background: #0B1220; }
QMainWindow#readingWorkspace::separator { background: #21354A; width: 8px; height: 8px; }
QMainWindow#readingWorkspace::separator:hover { background: #438E86; }
QDockWidget { color: #94ADC8; font-weight: 600; }
QDockWidget::title { background: #142337; padding: 7px 10px; border: 1px solid #293D56; }
QTableWidget { background: #111D2E; alternate-background-color: #16243A; border: 0;
    gridline-color: #293B53; selection-background-color: #214B4A; selection-color: #D9FFF4; }
QHeaderView::section { background: #17273D; color: #9EB5D1; font-size: 11px; padding: 12px;
    border: 0; border-bottom: 1px solid #30445E; font-weight: 600; text-align: left; }
QTableWidget::item { padding: 10px; border: 0; }
QListWidget { background: #111D2E; border: 1px solid #30445E; border-radius: 8px; padding: 5px; color: #DDE7F4; }
QListWidget::item { padding: 11px; border-radius: 5px; }
QListWidget::item:selected { background: #214B4A; color: #D9FFF4; }
QListWidget::item:hover { background: #1D3349; }
QProgressBar { background: #23364C; border: 0; border-radius: 2px; max-height: 4px; }
QProgressBar::chunk { background: #65DCC5; border-radius: 2px; }
QDialog { background: #101C2E; }
QMenu { background: #142337; color: #DDE7F4; border: 1px solid #405775; padding: 5px; }
QMenu::item { padding: 7px 18px; border-radius: 4px; }
QMenu::item:selected { background: #28544F; color: #E8FFF7; }
QToolTip { background: #21354A; color: #F4F8FF; border: 1px solid #405775; padding: 8px; }
"""


def _paint_brand(painter: QPainter) -> None:
    """The same original vector geometry as docs/assets/brand.svg (80 x 80)."""
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#102431"))
    painter.drawRoundedRect(QRectF(0, 0, 80, 80), 23, 23)
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(QColor("#66E4CA"), 3, Qt.PenStyle.SolidLine,
                        Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    triangle = QPainterPath(QPointF(40, 16))
    triangle.lineTo(65, 61)
    triangle.lineTo(15, 61)
    triangle.closeSubpath()
    painter.drawPath(triangle)
    painter.setPen(QPen(QColor("#88B5FF"), 2.5, Qt.PenStyle.SolidLine,
                        Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    painter.drawLine(QPointF(40, 16), QPointF(48, 61))
    inner = QPainterPath(QPointF(15, 61))
    inner.lineTo(44, 39)
    inner.lineTo(65, 61)
    painter.drawPath(inner)
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#E1FFF5"))
    painter.drawEllipse(QPointF(44, 39), 4, 4)


def _brand_icon() -> QIcon:
    """Native title/task-switcher icon; requires no external runtime asset."""
    pixmap = QPixmap(128, 128)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.scale(1.6, 1.6)
    _paint_brand(painter)
    painter.end()
    return QIcon(pixmap)


def _icon(kind: str, color: str = MUTED, size: int = 18) -> QIcon:
    """Small vector icons rendered without SVG or any external assets."""
    pix = QPixmap(size * 2, size * 2)
    pix.fill(Qt.GlobalColor.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    p.scale(size / 12, size / 12)
    p.setPen(QPen(QColor(color), 1.15, Qt.PenStyle.SolidLine,
                 Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
    if kind == "wave":
        for x, length in [(3, 4), (6, 9), (9, 15), (12, 10), (15, 5), (18, 12), (21, 4)]:
            p.drawLine(QPointF(x, 12 - length / 2), QPointF(x, 12 + length / 2))
    elif kind == "library":
        p.drawRoundedRect(QRectF(3, 4, 5, 16), 1, 1)
        p.drawRoundedRect(QRectF(10, 4, 5, 16), 1, 1)
        p.drawLine(18, 4, 21, 20)
    elif kind == "settings":
        for y, x in [(5, 8), (12, 16), (19, 10)]:
            p.drawLine(3, y, 21, y)
            p.setBrush(QColor(NAVY if color == "#A4BCC5" else "#FFFFFF"))
            p.drawEllipse(QPointF(x, y), 2.2, 2.2)
    elif kind == "play":
        p.setBrush(QColor(color))
        path = QPainterPath(QPointF(8, 5))
        path.lineTo(19, 12)
        path.lineTo(8, 19)
        path.closeSubpath()
        p.drawPath(path)
    elif kind == "stop":
        p.setBrush(QColor(color))
        p.drawRoundedRect(QRectF(6, 6, 12, 12), 2, 2)
    elif kind == "copy":
        p.drawRoundedRect(QRectF(8, 8, 12, 13), 2, 2)
        p.drawLine(5, 16, 4, 16)
        p.drawLine(4, 16, 4, 3)
        p.drawLine(4, 3, 16, 3)
    elif kind == "upload":
        p.drawLine(12, 17, 12, 3)
        p.drawLine(7, 8, 12, 3)
        p.drawLine(17, 8, 12, 3)
        p.drawLine(4, 15, 4, 21)
        p.drawLine(4, 21, 20, 21)
        p.drawLine(20, 21, 20, 15)
    elif kind == "export":
        p.drawLine(12, 3, 12, 17)
        p.drawLine(7, 12, 12, 17)
        p.drawLine(17, 12, 12, 17)
        p.drawLine(4, 17, 4, 21)
        p.drawLine(4, 21, 20, 21)
        p.drawLine(20, 21, 20, 17)
    elif kind == "refresh":
        p.drawArc(QRectF(4, 4, 16, 16), 30 * 16, 290 * 16)
        p.drawLine(20, 4, 20, 10)
        p.drawLine(20, 10, 14, 10)
    elif kind == "source":
        p.drawRoundedRect(QRectF(5, 3, 14, 18), 2, 2)
        for y in (8, 12, 16):
            p.drawLine(9, y, 15, y)
    elif kind == "spark":
        path = QPainterPath(QPointF(12, 2))
        for x, y in [(14.5, 9.5), (22, 12), (14.5, 14.5), (12, 22), (9.5, 14.5), (2, 12), (9.5, 9.5)]:
            path.lineTo(x, y)
        path.closeSubpath()
        p.drawPath(path)
    else:
        p.drawEllipse(QRectF(4, 4, 16, 16))
    p.end()
    pix.setDevicePixelRatio(2)
    return QIcon(pix)


def _label(text: str = "", name: str = "", wrap: bool = False) -> QLabel:
    widget = QLabel(text)
    if name:
        widget.setObjectName(name)
    widget.setWordWrap(wrap)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    return widget


def _button(text: str, kind: str = "", primary: bool = False) -> QPushButton:
    widget = QPushButton(text)
    widget.setCursor(Qt.CursorShape.PointingHandCursor)
    widget.setMinimumHeight(34)
    if primary:
        widget.setObjectName("primary")
    if kind:
        widget.setIcon(_icon(kind, "#072321" if primary else MUTED))
        widget.setIconSize(QSize(16, 16))
    return widget


def _frame(name: str = "card") -> QFrame:
    widget = QFrame()
    widget.setObjectName(name)
    return widget


def _vbox(widget: QWidget, margin: int = 20, spacing: int = 12) -> QVBoxLayout:
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(margin, margin, margin, margin)
    layout.setSpacing(spacing)
    return layout


def _divider() -> QFrame:
    return _frame("divider")


def _combo(items: list[tuple[str, str]], editable: bool = False) -> QComboBox:
    control = QComboBox()
    control.setEditable(editable)
    for title, value in items:
        control.addItem(title, value)
    control.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return control


def _select_value(control: QComboBox, value: Any) -> None:
    index = control.findData(value)
    if index < 0:
        index = control.findText(str(value))
    if index >= 0:
        control.setCurrentIndex(index)
    elif control.isEditable():
        control.setEditText(str(value or ""))


def _combo_value(control: QComboBox) -> str:
    # An editable combobox can retain the old item's userData after editing.
    if control.isEditable():
        return control.currentText().strip()
    return str(control.currentData() or "")


class QuestionPreview(QLabel):
    """A naturally wrapping question inside the continuous answer article."""
    def __init__(self, name: str, lines: int = 2):
        super().__init__()
        self.setObjectName(name)
        self.full_text = ""
        self.lines = None
        self.setTextFormat(Qt.TextFormat.PlainText)
        self.setWordWrap(True)
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

    def set_preview(self, text: str) -> None:
        if text != self.full_text:
            self.full_text = text
            self._render_preview()

    def _render_preview(self) -> None:
        if self.text() != self.full_text:
            self.setText(self.full_text)
        self.updateGeometry()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render_preview()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in (QEvent.Type.FontChange, QEvent.Type.StyleChange):
            self._render_preview()


class InterviewModeDropdown(QFrame):
    """A small keyboard-accessible popup which does not reflow the workspace."""
    def __init__(self, parent: QWidget):
        super().__init__(parent, Qt.WindowType.Popup | Qt.WindowType.FramelessWindowHint)
        self.setObjectName("interviewModeDropdown")
        self.setAccessibleName("面试模式")
        self.setAttribute(Qt.WidgetAttribute.WA_NoMouseReplay, True)
        self.setStyleSheet("QFrame#interviewModeDropdown { background:#142337; border:1px solid #405775; border-radius:10px; }")

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self.hide()
            event.accept()
            return
        super().keyPressEvent(event)


class PrismMark(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFixedSize(38, 38)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.scale(self.width() / 80, self.height() / 80)
        _paint_brand(painter)



class EmptyGlyph(QWidget):
    """Quiet editorial empty-state illustration, rather than a fake waveform."""
    def __init__(self, answer: bool = False):
        super().__init__()
        self.answer = answer
        self.setFixedSize(136, 110)

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setPen(QPen(QColor("#2C445C"), 1))
        p.setBrush(QColor("#142337"))
        p.drawEllipse(QRectF(23, 10, 90, 90))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QRectF(34, 21, 68, 68))
        p.setPen(QPen(QColor("#65DCC5"), 2, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        if self.answer:
            p.drawRoundedRect(QRectF(48, 33, 41, 43), 6, 6)
            for y, length in [(45, 23), (54, 23), (63, 14)]:
                p.drawLine(57, y, 57 + length, y)
        else:
            for x, length in [(47, 9), (57, 24), (67, 39), (77, 24), (87, 9)]:
                p.drawLine(QPointF(x, 55 - length / 2), QPointF(x, 55 + length / 2))
        p.setBrush(QColor("#264A4C"))
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QRectF(18, 78, 7, 7))
        p.drawEllipse(QRectF(111, 26, 5, 5))


class BilingualParagraph(QWidget):
    """One stable source/translation unit; never infer alignment from sentences."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = _vbox(self, 0, 5)
        self.english = _label("", "transcriptEn", True)
        self.chinese = _label("", "transcriptZh", True)
        self.pending = _label("", "muted", True)
        pending_policy = self.pending.sizePolicy()
        pending_policy.setRetainSizeWhenHidden(True)
        self.pending.setSizePolicy(pending_policy)
        self.pending.setFixedHeight(18)
        for reader in (self.english, self.chinese):
            reader.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            reader.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            layout.addWidget(reader)
        self.chinese.setToolTip("对应上方同一段英文的译文；识别原话始终保留，可用于核对术语与细节。")
        layout.addWidget(self.pending)
        self.pending.hide()

    def set_text(self, en: str, zh: str, pending: bool = False) -> bool:
        # A translation belongs to the stable source ID, not a guessed nth
        # sentence. Translators may merge sentences or use different endings.
        # Restored speech may carry a newline for every guessed sentence. The
        # caption paragraph should wrap at its available width, not promote
        # those provisional punctuation guesses into separate display rows.
        from .caption_display import caption_english
        display_en = caption_english(en)
        changed = False
        if self.english.text() != display_en:
            self.english.setText(display_en)
            changed = True
        if self.chinese.text() != zh:
            self.chinese.setText(zh)
            changed = True
        if self.chinese.isHidden() == bool(zh):
            self.chinese.setVisible(bool(zh))
            changed = True
        message = "本段补充内容正在翻译…" if pending and zh else "正在翻译本段…" if pending else ""
        if self.pending.text() != message:
            self.pending.setText(message)
            changed = True
        if self.pending.isHidden() == bool(message):
            self.pending.setVisible(bool(message))
            changed = True
        return changed

    def minimumSizeHint(self):
        # QLabel's wrapped minimum height is measured at its narrowest word.
        # Propagating that as a nested paragraph minimum produces a phantom
        # scroll range even when the wide card already fits the entire source.
        hint = super().minimumSizeHint()
        return QSize(hint.width(), 0)


class TranscriptItem(QFrame):
    assign_requested = Signal()

    def __init__(self, en: str, zh: str, partial: bool, speaker_id=None, speaker_label=None):
        super().__init__()
        self.setObjectName("partialItem" if partial else "transcriptItem")
        layout = _vbox(self, 14, 8)
        header = QHBoxLayout()
        self.speaker_id = speaker_id
        self.speaker_status = None
        self.speaker_label = speaker_label or ("说话人未确认" if not speaker_id else str(speaker_id))
        self._segments: dict[str, tuple[str, str, bool]] = {}
        self._unit_widgets: dict[str, BilingualParagraph] = {}
        self._raw_segments: dict[str, str] = {}
        self.turn_ids: set[str] = set()
        self._pending_translations: set[str] = set()
        self.speaker = _label("识别中 · 尚未确认" if partial else self.speaker_label, "sectionEyebrow")
        header.addWidget(self.speaker)
        assign = QToolButton()
        assign.setText("更正")
        assign.setObjectName("quiet")
        assign.setToolTip("手动指定这段话的说话人；声纹标签可能有误。")
        assign.clicked.connect(self.assign_requested.emit)
        header.addWidget(assign)
        header.addStretch()
        self.timestamp = _label(datetime.now().strftime("%H:%M:%S"), "time")
        header.addWidget(self.timestamp)
        layout.addLayout(header)
        self.english = _label(en, "transcriptEn", True)
        self.english.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.english.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.chinese = _label(zh or "中文译文待生成…", "transcriptZh", True)
        self.chinese.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.chinese.setToolTip("本地实时译文可能误译或漏译，请结合上方英语原话核对。回答模型的问题释义是另一输出，不会自动校正这条译文。")
        self.chinese.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        # Aggregate readers retain the controller/export compatibility contract;
        # actual display is interleaved by the stable translation source IDs.
        self.english.setParent(self)
        self.chinese.setParent(self)
        self.english.hide()
        self.chinese.hide()
        self.paragraphs = QWidget()
        self.paragraphs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.paragraph_layout = _vbox(self.paragraphs, 0, 15)
        layout.addWidget(self.paragraphs)
        self.raw_toggle = QToolButton()
        self.raw_toggle.setText("展开识别原话")
        self.raw_toggle.setCheckable(True)
        self.raw_toggle.setObjectName("quiet")
        header.insertWidget(2, self.raw_toggle)
        self.raw_text = _label("", "muted", True)
        self.raw_text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.raw_text.hide()
        self.raw_toggle.toggled.connect(self.raw_text.setVisible)
        layout.addWidget(self.raw_text)

    def update_text(self, en: str, zh: str, partial: bool) -> bool:
        if self.english.text() != en:
            self.english.setText(en)
        if zh and self.chinese.text() != zh:
            self.chinese.setText(zh)
        changed = False
        if self.speaker.text() != self.speaker_label:
            self.speaker.setText(self.speaker_label)
            changed = True
        desired = "partialItem" if partial else "transcriptItem"
        if self.objectName() != desired:
            self.setObjectName(desired)
            self.style().unpolish(self)
            self.style().polish(self)
            changed = True
        return changed

    def update_segment(self, identifier: str, en: str, zh: str, partial: bool,
                       speaker_id=None, speaker_label=None, raw_en=None, speaker_status=None) -> bool:
        if speaker_id:
            self.speaker_id = speaker_id
        if speaker_label:
            self.speaker_label = str(speaker_label)
        if speaker_status is not None:
            self.speaker_status = str(speaker_status)
        self._segments[str(identifier)] = (str(en), str(zh), bool(partial))
        self._raw_segments[str(identifier)] = str(raw_en if raw_en is not None else
            self._raw_segments.get(str(identifier), en))
        from .transcript_text import split_sentences
        # Segment boundaries belong to ASR, not necessarily to sentences. Join
        # fragments first, then render complete sentences on separate lines.
        english = "\n".join(split_sentences(" ".join(
            part[0].strip() for part in self._segments.values() if part[0].strip())))
        chinese = "\n".join(self._translation_lines(part[1].strip())
            for part in self._segments.values() if part[1].strip())
        pending = bool(self._pending_translations) or any(not part[1].strip() for part in self._segments.values())
        if pending and chinese:
            chinese += "\n正在翻译最新一句…"
        changed = self.update_text(english, chinese, any(part[2] for part in self._segments.values()))
        for unit_id in list(self._unit_widgets):
            if unit_id not in self._segments:
                unit = self._unit_widgets.pop(unit_id)
                self.paragraph_layout.removeWidget(unit)
                unit.deleteLater()
                changed = True
        for unit_id, (unit_en, unit_zh, _unit_partial) in self._segments.items():
            if unit_id not in self._unit_widgets:
                unit = BilingualParagraph(self.paragraphs)
                self._unit_widgets[unit_id] = unit
                self.paragraph_layout.addWidget(unit)
                changed = True
            unit_changed = self._unit_widgets[unit_id].set_text(unit_en.strip(), unit_zh.strip(),
                unit_id in self._pending_translations or not unit_zh.strip())
            changed = unit_changed or changed
        raw_text = "\n".join(part for part in self._raw_segments.values() if part)
        if self.raw_text.text() != raw_text:
            self.raw_text.setText(raw_text)
            changed = not self.raw_text.isHidden() or changed
        if not chinese:
            self.chinese.setText("中文译文待生成…")
        self.speaker.setToolTip(self.speaker_status_tooltip(self.speaker_status))
        palette = ("#65DCC5", "#92B7FF", "#C4ABFF", "#EFC381")
        color = palette[sum(map(ord, str(self.speaker_id))) % len(palette)] if self.speaker_id else "#9CA9AF"
        card_style = f"QFrame#{self.objectName()} {{ border-left:3px solid {color}; }}"
        speaker_style = f"color:{color}; font-size:11px; font-weight:600;"
        if self.styleSheet() != card_style:
            self.setStyleSheet(card_style)
            changed = True
        if self.speaker.styleSheet() != speaker_style:
            self.speaker.setStyleSheet(speaker_style)
            changed = True
        return changed

    @staticmethod
    def speaker_status_tooltip(status) -> str:
        if status == "source":
            return "远端音频 · 来源标签，不是声纹身份确认。采集当前扬声器回环；其他应用声音也会进入，不含本机麦克风。"
        if status == "manual":
            return "人工指定"
        return "匿名声纹分组可能识别有误；不代表已确认参会者身份。"

    @staticmethod
    def _translation_lines(text: str) -> str:
        """Display sentence boundaries without altering words/numeric values."""
        from .transcript_text import split_sentences
        # Local translators can mix Chinese prose with ASCII sentence endings.
        # Decimal points are not endings; attached Chinese after a period needs
        # a boundary space before the shared abbreviation-aware splitter.
        text = re.sub(r"([。！？?!]+)[ \t]*(?=\S)", r"\1\n", str(text))
        text = re.sub(r"\.(?=[\u3400-\u9fff])", ". ", text)
        lines = []
        abbreviation = re.compile(r"(?i)(?<![a-z])(?:e\.g|i\.e|etc|dr|prof|mr|mrs|ms|vs|fig|eq|approx|al)\.$")
        for line in text.splitlines():
            pieces = []
            for sentence in split_sentences(line):
                # Chinese may touch the abbreviation without a separating
                # space, so the shared English tokenizer cannot see its start.
                if pieces and abbreviation.search(pieces[-1]):
                    pieces[-1] += " " + sentence
                elif sentence:
                    pieces.append(sentence)
            lines.extend(pieces)
        return "\n".join(lines)


class ReadingDock(QDockWidget):
    """Closing a floating reader returns it to the workspace, retaining text."""
    def closeEvent(self, event):
        if self.isFloating():
            event.ignore()
            self.setFloating(False)
            self.show()
        else:
            super().closeEvent(event)


class ModelChoice(QComboBox):
    """Editable genuine catalog choices, retaining the old text-field contract."""
    def __init__(self):
        super().__init__()
        self.setEditable(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def text(self) -> str:
        index = self.currentIndex()
        if index >= 0 and self.currentText() == self.itemText(index):
            return str(self.itemData(index) or "")
        return self.currentText().strip()

    def setText(self, text: str) -> None:
        index = self.findData(text)
        self.setCurrentIndex(index)
        if index < 0:
            self.setEditText(text)

    def setPlaceholderText(self, text: str) -> None:
        self.lineEdit().setPlaceholderText(text)


class CopilotWindow(QMainWindow):
    start_requested = Signal()
    stop_requested = Signal()
    answer_requested = Signal(str)
    cancel_requested = Signal()
    import_requested = Signal(list)
    clear_requested = Signal()
    refresh_devices_requested = Signal()
    check_models_requested = Signal()
    install_translation_requested = Signal()
    install_speaker_requested = Signal()
    export_requested = Signal()
    save_session_requested = Signal()
    history_requested = Signal()
    auto_save_changed = Signal(bool)
    settings_saved = Signal(dict)
    interview_phase_changed = Signal(str)
    new_question_requested = Signal()
    model_picker_requested = Signal()
    speaker_assigned = Signal(str, str, str)
    speaker_role_changed = Signal(str, str)
    question_packet_changed = Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Prism Interview · 0.4.0")
        self.setWindowIcon(_brand_icon())
        self.setMinimumSize(1120, 720)
        self.resize(1500, 980)
        available = QApplication.primaryScreen().availableGeometry() if QApplication.primaryScreen() else None
        if available:
            self.resize(min(1500, max(1120, available.width() - 40)),
                        min(980, max(720, available.height() - 45)))
        self._settings: dict[str, Any] = {}
        self._session_api_keys: dict[str, str] = {}
        self._session_api_bases: dict[str, str] = {}
        self._session_answer_models: dict[str, str] = {}
        self._last_provider = "ollama"
        self._api_endpoint_identity = ""
        self._interview_phase = "qa"
        self._listening = False
        self._busy = False
        self._started_at = 0.0
        self._entries: dict[str, TranscriptItem] = {}
        self._turn_cards: dict[str, TranscriptItem] = {}
        self._focus_mode = "split"
        self._question_panel_expanded = True
        self._reading_layout_checked = False
        self._reading_size_timer = QTimer(self)
        self._reading_size_timer.setSingleShot(True)
        self._reading_size_timer.timeout.connect(self._size_reading_docks)
        self._timeline_scroll_generation = 0
        self._timeline_follow_pending = False
        self._timeline_applying_position = False
        self._timeline_anchor = None
        self._timeline_anchor_generation = 0
        self._timeline_position_timer = QTimer(self)
        self._timeline_position_timer.setSingleShot(True)
        self._timeline_position_timer.timeout.connect(self._apply_timeline_position)
        self._timeline_finish_timer = QTimer(self)
        self._timeline_finish_timer.setSingleShot(True)
        self._timeline_finish_timer.timeout.connect(self._finish_timeline_position)
        self._timeline_finish_generation = 0
        self._answer_scroll_generation = 0
        self._answer_pairs: list[QWidget] = []
        self._answer_pair_widgets: dict[tuple, QWidget] = {}
        self._answer_visible_pairs: list[dict] | None = None
        self._hidden_floating_docks: set[str] = set()
        self._sources: list = []
        self._answer_en = ""
        self._answer_zh = ""
        self._answer_question_en = ""
        self._answer_question_preview_en = ""
        self._answer_question_raw_en = ""
        self._answer_question_zh = ""
        self._answer_question_dialog: QDialog | None = None
        self._answer_history: dict[str, dict] = {}
        self._answer_active_id: str | None = None
        self._answer_displayed_id: str | None = None
        self._answer_model_name = ""
        self._sources_dialog: QDialog | None = None
        self._overlay = None
        self._live_en = ""
        self._live_zh = ""
        self._live_units = []
        self._latest_caption_id = ""
        self._live_turn_id = ""
        self._live_speaker_id = None
        self._live_speaker_status = None
        self._live_speaker_label = "说话人待补认"
        self._live_state = "等待语音"
        self._speaker_roles = {}
        self._question_packet = {}
        self._background_question_packet = {}
        self._question_edit_base = ""
        self._manual_question_active = False
        self._make_ui()
        self.setStyleSheet(STYLE)
        self._clock = QTimer(self)
        self._clock.timeout.connect(self._update_clock)
        self._clock.start(1000)
        self.set_settings({})
        self.set_listening(False)
        self.set_busy(False)

    def _make_ui(self) -> None:
        canvas = QWidget()
        canvas.setObjectName("canvas")
        self.setCentralWidget(canvas)
        root = QHBoxLayout(canvas)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        self.sidebar = self._make_sidebar()
        root.addWidget(self.sidebar)
        workspace = QWidget()
        layout = _vbox(workspace, 25, 17)
        layout.setContentsMargins(26, 24, 26, 13)
        layout.addLayout(self._make_header())
        self.pages = QStackedWidget()
        self.pages.addWidget(self._make_session())
        self.pages.addWidget(self._make_library())
        self.pages.addWidget(self._make_settings())
        layout.addWidget(self.pages, 1)
        layout.addLayout(self._make_footer())
        root.addWidget(workspace, 1)
        self._set_reading_mode()

    def _make_sidebar(self) -> QWidget:
        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(192)
        layout = _vbox(side, 16, 9)
        layout.setContentsMargins(16, 29, 16, 21)
        brand = QHBoxLayout()
        brand.setSpacing(9)
        brand.addWidget(PrismMark())
        brand.addWidget(_label("Prism", "brand"))
        brand.addStretch()
        layout.addLayout(brand)
        layout.addWidget(_label("INTERVIEW", "brandSub"))
        layout.addSpacing(36)
        layout.addWidget(_label("WORKSPACE", "navSection"))
        layout.addSpacing(6)
        self.nav_buttons: list[QPushButton] = []
        for i, (title, icon) in enumerate([("面试工作台", "wave"), ("面试资料库", "library"), ("模型与设置", "settings")]):
            btn = _button(title)
            btn.setIcon(_icon(icon, "#A4BCC5", 19))
            btn.setIconSize(QSize(19, 19))
            btn.setObjectName("navButton")
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked=False, index=i: self._navigate(index))
            self.nav_buttons.append(btn)
            layout.addWidget(btn)
        self.nav_buttons[0].setChecked(True)
        layout.addStretch()
        context = _frame("sideContext")
        box = _vbox(context, 13, 8)
        box.addWidget(_label("YOUR CONTEXT", "navSection"))
        self.context_label = _label("候选人背景与目标岗位\n在设置中自定义", wrap=True)
        self.context_label.setStyleSheet("font-size:12px; color:#BAD0E7; line-height:1.5;")
        box.addWidget(self.context_label)
        layout.addWidget(context)
        layout.addSpacing(13)
        layout.addWidget(_label("资料本地索引\n云端发送范围由你控制", "navFoot", True))
        return side

    def _make_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.setSpacing(5)
        titles.addWidget(_label("PRISM INTERVIEW  /  WORKSPACE", "eyebrow"))
        self.page_title = _label("面试工作台", "pageTitle")
        titles.addWidget(self.page_title)
        self.page_subtitle = _label("听懂问题 · 结合资料 · 自信表达", "subtitle")
        titles.addWidget(self.page_subtitle)
        self.preparation_status = _label("资料准备状态待检查", "muted", True)
        self.preparation_status.setToolTip("资料在本地提取和索引；回答时检索相关片段，这不代表重新训练模型或保证事实正确。")
        titles.addWidget(self.preparation_status)
        header.addLayout(titles, 1)
        self.reading_mode = _combo([("面试阅读布局", "split"), ("专注 · 听译", "transcript"),
                                    ("专注 · 回答", "answer"), ("专注 · 当前问题", "question")])
        self.reading_mode.setMaximumWidth(160)
        self.reading_mode.setToolTip("面板可拖动、调整宽高或浮出到其他屏幕；专注模式仍可只看一个面板。")
        self.reading_mode.currentIndexChanged.connect(self._set_reading_mode)
        header.addWidget(self.reading_mode)
        self.focus_question_button = _button("恢复布局")
        self.focus_question_button.clicked.connect(self.restore_reading_layout)
        self.focus_question_button.setToolTip("将三个阅读面板收回主窗口，并恢复回答优先的两列布局。")
        header.addWidget(self.focus_question_button)
        self.focus_controls_button = _button("面试控制")
        # A painted Qt widget popup stays anchored to this button without
        # inserting another toolbar row or depending on native menu rendering.
        self.focus_controls_button.setAccessibleName("选择面试模式")
        self.focus_controls_button.setToolTip("在此按钮下方切换汇报／问答模式及自动回答；Esc 关闭。")
        self.focus_controls_button.clicked.connect(self._toggle_interview_controls)
        self.focus_controls_button.hide()
        header.addWidget(self.focus_controls_button)
        self.model_picker_button = _button("选择回答模型", "settings")
        self.model_picker_button.clicked.connect(self.model_picker_requested.emit)
        header.addWidget(self.model_picker_button)
        right = QVBoxLayout()
        right.setAlignment(Qt.AlignmentFlag.AlignRight)
        self.privacy_chip = _label("本地模式", "badge")
        self.privacy_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right.addWidget(self.privacy_chip)
        self.session_clock = _label("尚未开始监听", "muted")
        self.session_clock.setAlignment(Qt.AlignmentFlag.AlignRight)
        right.addWidget(self.session_clock)
        header.addLayout(right)
        return header

    def _make_session(self) -> QWidget:
        page = QWidget()
        layout = _vbox(page, 0, 14)
        self.interview_controls_panel = InterviewModeDropdown(self)
        self.interview_controls_panel.setFixedWidth(280)
        controls = QVBoxLayout(self.interview_controls_panel)
        controls.setContentsMargins(12, 10, 12, 10)
        controls.setSpacing(7)
        controls.addWidget(_label("面试模式", "cardTitle"))
        self.inline_presentation_button = _button("汇报 · 仅听译")
        self.inline_qa_button = _button("问答 · 自动辅助")
        for phase, button in (("presentation", self.inline_presentation_button), ("qa", self.inline_qa_button)):
            button.setObjectName("phaseButton")
            button.setCheckable(True)
            button.setAccessibleName("汇报模式，仅听译" if phase == "presentation" else "问答模式")
            button.setMinimumHeight(34)
            button.clicked.connect(lambda checked=False, value=phase: self._select_inline_phase(value))
            controls.addWidget(button)
        self.dropdown_auto_checkbox = QCheckBox("自动整理问题并生成回答")
        self.dropdown_auto_checkbox.setToolTip("仅在问答模式生效；可随时关闭自动回答，继续保留实时听译。")
        self.dropdown_auto_checkbox.toggled.connect(lambda checked: self.auto_checkbox.setChecked(checked))
        controls.addWidget(self.dropdown_auto_checkbox)
        footer = QHBoxLayout()
        footer.setSpacing(6)
        generate = _button("生成当前回答")
        generate.clicked.connect(self._answer_from_mode_dropdown)
        footer.addWidget(generate)
        next_question = _button("下一题")
        next_question.clicked.connect(self._next_from_mode_dropdown)
        footer.addWidget(next_question)
        controls.addLayout(footer)
        self.interview_controls_panel.hide()
        capture = _frame("captureBar")
        row = QHBoxLayout(capture)
        row.setContentsMargins(16, 13, 14, 13)
        row.setSpacing(10)
        audio_group = QVBoxLayout()
        audio_group.setSpacing(2)
        audio_group.addWidget(_label("音频输入", "sectionEyebrow"))
        audio_group.addWidget(_label("系统输出回环", "muted"))
        row.addLayout(audio_group)
        self.device_select = QComboBox()
        self.device_select.setMinimumWidth(180)
        self.device_select.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.device_select.setToolTip("选择 Zoom 的扬声器／耳机输出。不会采集自己的麦克风。")
        self.device_select.currentIndexChanged.connect(self._device_changed)
        row.addWidget(self.device_select, 1)
        refresh = _button("", "refresh")
        refresh.setFixedWidth(34)
        refresh.setToolTip("刷新音频输出设备")
        refresh.setAccessibleName("刷新音频输出设备")
        refresh.clicked.connect(self.refresh_devices_requested.emit)
        row.addWidget(refresh)
        self.start_button = _button("开始监听", "play", True)
        self.start_button.clicked.connect(self._start)
        row.addWidget(self.start_button)
        self.stop_button = _button("停止", "stop")
        self.stop_button.clicked.connect(self.stop_requested.emit)
        row.addWidget(self.stop_button)
        self.overlay_button = _button("浮动字幕")
        self.overlay_button.clicked.connect(self.show_subtitle_overlay)
        row.addWidget(self.overlay_button)
        self.capture_mode_select = _combo([("对方 1 人", "single_remote"), ("对方多人", "acoustic")])
        self.capture_mode_select.setMaximumWidth(130)
        self.capture_mode_select.setToolTip("仅听 Zoom 远端，不采集本机麦克风；改变人数后重新开始监听。多人采用匿名声纹分组。")
        self.capture_mode_select.currentIndexChanged.connect(self._quick_caption_mode_changed)
        row.addWidget(self.capture_mode_select)
        layout.addWidget(capture)

        self.dock_workspace = QMainWindow(page)
        self.dock_workspace.setObjectName("readingWorkspace")
        self.dock_workspace.setWindowFlags(Qt.WindowType.Widget)
        self.dock_workspace.setDockOptions(QMainWindow.DockOption.AllowNestedDocks | QMainWindow.DockOption.AllowTabbedDocks)
        self.dock_workspace.setMinimumSize(0, 0)
        layout.addWidget(self.dock_workspace, 1)
        timeline = _frame()
        self.timeline_panel = timeline
        timeline.setMinimumWidth(270)
        column = _vbox(timeline, 10, 6)
        head = QHBoxLayout()
        title = QVBoxLayout()
        title.setSpacing(3)
        title.addWidget(_label("BILINGUAL CONVERSATION", "sectionEyebrow"))
        self.conversation_title = _label("说话人 · 双语对话", "cardTitle")
        self.conversation_title.setToolTip("显示最近已校订语句的说话人；完整轮次标签保留在卡片中。")
        title.addWidget(self.conversation_title)
        head.addLayout(title)
        head.addStretch()
        self.expand_transcript_button = _button("浮出")
        self.expand_transcript_button.setToolTip("将对话作为独立阅读窗口，可拖到扩展屏并调整宽高。")
        self.expand_transcript_button.clicked.connect(lambda: self._float_reading_panel("transcript"))
        head.addWidget(self.expand_transcript_button)
        self.live_chip = _label("未监听", "neutralBadge")
        head.addWidget(self.live_chip)
        column.addLayout(head)
        column.addWidget(_divider())
        draft_row = QHBoxLayout()
        self.draft_toggle = QToolButton()
        self.draft_toggle.setText("查看草稿 ▸")
        self.draft_toggle.setFixedWidth(84)
        self.draft_toggle.setCheckable(True)
        self.draft_toggle.setObjectName("quiet")
        self.draft_toggle.setToolTip("草稿可能反复修订。正式语句和译文保留在下方卡片，默认收起草稿。")
        draft_row.addWidget(self.draft_toggle)
        draft_row.addStretch()
        column.addLayout(draft_row)
        self.draft_panel = QWidget()
        draft_column = _vbox(self.draft_panel, 0, 5)
        self.draft_panel.setMaximumHeight(205)
        speaker_row = QHBoxLayout()
        self.live_speaker = _label("说话人待补认", "sectionEyebrow")
        self.live_speaker.setToolTip("声音积累后会补认说话人；不能推断参会者姓名。仅包含所选系统输出中的声音。")
        speaker_row.addWidget(self.live_speaker)
        self.assign_live_speaker = QToolButton()
        self.assign_live_speaker.setText("更正说话人")
        self.assign_live_speaker.setObjectName("quiet")
        self.assign_live_speaker.clicked.connect(lambda: self._show_speaker_assignment(self._live_turn_id))
        speaker_row.addWidget(self.assign_live_speaker)
        speaker_row.addStretch()
        self.live_sentence_state = _label("等待语音", "muted")
        speaker_row.addWidget(self.live_sentence_state)
        draft_column.addLayout(speaker_row)
        self.live_readers = QSplitter(Qt.Orientation.Vertical)
        self.live_readers.setChildrenCollapsible(False)
        self.live_readers.setHandleWidth(8)
        self.live_caption = QPlainTextEdit()
        self.live_caption.setReadOnly(True)
        self.live_caption.setMinimumHeight(40)
        self.live_caption.setPlaceholderText("英文草稿 · 可能继续修订")
        self.live_caption.setStyleSheet("font-size:14px; background:#132236; border:0; padding:5px;")
        self.live_readers.addWidget(self.live_caption)
        self.live_translation = QPlainTextEdit()
        self.live_translation.setReadOnly(True)
        self.live_translation.setMinimumHeight(40)
        self.live_translation.setPlaceholderText("中文草稿 · 完整译文保留在下方卡片")
        self.live_translation.setStyleSheet("font-size:14px; color:#ABC3DB; background:#132236; border:0; padding:5px;")
        self.live_readers.addWidget(self.live_translation)
        self.live_readers.setSizes([55, 50])
        draft_column.addWidget(self.live_readers, 1)
        raw_row = QHBoxLayout()
        self.live_raw_toggle = QToolButton()
        self.live_raw_toggle.setText("展开识别原话")
        self.live_raw_toggle.setCheckable(True)
        self.live_raw_toggle.setObjectName("quiet")
        raw_row.addWidget(self.live_raw_toggle)
        raw_row.addStretch()
        self.history_toggle = QToolButton()
        # Kept as a controller/test compatibility handle. Conversation cards
        # are always the primary reader; they no longer live behind a toggle.
        self.history_toggle.setText("说话人对话")
        self.history_toggle.setCheckable(True)
        self.history_toggle.setChecked(True)
        self.history_toggle.setObjectName("quiet")
        self.history_toggle.toggled.connect(self._toggle_history)
        self.history_toggle.hide()
        draft_column.addLayout(raw_row)
        self.live_raw = QPlainTextEdit()
        self.live_raw.setReadOnly(True)
        self.live_raw.setMaximumHeight(55)
        self.live_raw.setPlaceholderText("识别原话保留用于核对；整理后的英文显示在上方。")
        self.live_raw.hide()
        self.live_raw_toggle.toggled.connect(self.live_raw.setVisible)
        draft_column.addWidget(self.live_raw)
        column.addWidget(self.draft_panel)
        self.draft_panel.hide()
        self.draft_toggle.toggled.connect(self._toggle_draft)
        self.history_controls_widget = QWidget()
        history_controls = QHBoxLayout()
        history_controls.setContentsMargins(0, 0, 0, 0)
        self.history_controls_widget.setLayout(history_controls)
        self.follow_transcript = QCheckBox("跟随最新对话")
        self.follow_transcript.setChecked(True)
        self.follow_transcript.setToolTip("勾选后自动滚到最新译文；取消后可以稳定回看历史，新增内容不会把阅读位置拉走。")
        self.follow_transcript.toggled.connect(self._follow_latest_changed)
        history_controls.addWidget(self.follow_transcript)
        history_controls.addStretch()
        self.latest_button = _button("已在最新位置")
        self.latest_button.setFixedWidth(112)
        self.latest_button.clicked.connect(self._resume_timeline_follow)
        history_controls.addWidget(self.latest_button)
        draft_row.addWidget(self.history_controls_widget)
        self.transcript_stack = QStackedWidget()
        self.transcript_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.transcript_stack.setMinimumHeight(70)
        empty = QWidget()
        empty_layout = _vbox(empty, 12, 11)
        empty_layout.addStretch()
        empty_layout.addWidget(EmptyGlyph(), 0, Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(_label("从听清每一个问题开始", "emptyTitle"), 0, Qt.AlignmentFlag.AlignCenter)
        body = _label("按说话轮次显示英语原话与中文译文。\n声纹分组用匿名标签，并可能有误。", "emptyBody", True)
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(body)
        empty_layout.addSpacing(10)
        setup = _button("检查模型与音频")
        setup.clicked.connect(self.check_models_requested.emit)
        empty_layout.addWidget(setup, 0, Qt.AlignmentFlag.AlignCenter)
        empty_layout.addStretch()
        self.transcript_stack.addWidget(empty)
        self.transcript_scroll = QScrollArea()
        self.transcript_scroll.setWidgetResizable(True)
        # Reserve scrollbar width from the first card; its appearance must not
        # rewrap every previous sentence while following newly added speech.
        self.transcript_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.transcript_scroll.viewport().installEventFilter(self)
        self.transcript_scroll.verticalScrollBar().valueChanged.connect(self._timeline_scroll_changed)
        # Follow late word-wrap/layout changes too, without posting another
        # layout pass or replacing any card widgets.
        self.transcript_scroll.verticalScrollBar().rangeChanged.connect(self._timeline_range_changed)
        items = QWidget()
        self.transcript_layout = _vbox(items, 0, 10)
        # Conversation order stays chronological. Stretch only unused space;
        # do not reserve a screenful below the newest message to pin it on top.
        self.transcript_layout.addStretch()
        self.transcript_scroll.setWidget(items)
        self.transcript_stack.addWidget(self.transcript_scroll)
        column.addWidget(self.transcript_stack, 1)
        column.addWidget(_divider())
        row = QHBoxLayout()
        row.addWidget(_label("音量", "muted"))
        self.level_meter = QProgressBar()
        self.level_meter.setRange(0, 100)
        self.level_meter.setTextVisible(False)
        self.level_meter.setFixedWidth(85)
        row.addWidget(self.level_meter)
        row.addStretch()
        self.asr_metric = _label("识别引擎待检查", "muted")
        row.addWidget(self.asr_metric)
        column.addLayout(row)

        reply = _frame()
        self.reply_panel = reply
        reply.setMinimumWidth(380)
        column = _vbox(reply, 12, 6)
        head = QHBoxLayout()
        title = QVBoxLayout()
        title.setSpacing(3)
        self.answer_model_label = _label("模型 · 尚未确认", "sectionEyebrow")
        self.answer_model_label.setMaximumWidth(250)
        self.answer_model_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        title.addWidget(self.answer_model_label)
        title.addWidget(_label("回答建议", "cardTitle"))
        head.addLayout(title)
        head.addStretch()
        self.expand_answer_button = _button("浮出")
        self.expand_answer_button.setToolTip("将完整问题和回答作为独立阅读窗口，可拖到扩展屏并调整宽高。")
        self.expand_answer_button.clicked.connect(lambda: self._float_reading_panel("answer"))
        head.addWidget(self.expand_answer_button)
        self.source_button = _button("来源  0", "source")
        self.source_button.setObjectName("quiet")
        self.source_button.clicked.connect(self._show_sources)
        self.source_button.setEnabled(False)
        head.addWidget(self.source_button)
        self.copy_button = _button("复制英文", "copy")
        self.copy_button.setObjectName("quiet")
        self.copy_button.clicked.connect(self._copy_answer)
        self.copy_button.setEnabled(False)
        head.addWidget(self.copy_button)
        column.addLayout(head)
        column.addWidget(_divider())
        self.answer_history_bar = QWidget()
        history_row = QHBoxLayout(self.answer_history_bar)
        history_row.setContentsMargins(0, 0, 0, 0)
        history_row.setSpacing(5)
        self.answer_history_select = QComboBox()
        self.answer_history_select.setMinimumWidth(80)
        self.answer_history_select.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.answer_history_select.setToolTip("本次会话的回答记录。已完成答案保留在这里，选择即可回看。")
        self.answer_history_select.currentIndexChanged.connect(self._select_answer_history)
        history_row.addWidget(self.answer_history_select, 1)
        self.follow_answers_checkbox = QCheckBox("自动切换")
        self.follow_answers_checkbox.setChecked(True)
        self.follow_answers_checkbox.setToolTip("默认开启：新回答出现有效文字后自动切换，等待期间保留当前答案。手动回看历史会暂时关闭，可重新勾选。")
        self.follow_answers_checkbox.toggled.connect(self._follow_new_answers_changed)
        history_row.addWidget(self.follow_answers_checkbox)
        self.answer_new_button = QToolButton()
        self.answer_new_button.setObjectName("answerNewButton")
        self.answer_new_button.setStyleSheet(
            "QToolButton { background:#153D38; color:#96F0DB; border:1px solid #377A6D; "
            "border-radius:6px; padding:5px 8px; font-weight:600; } "
            "QToolButton:hover { background:#214D46; }")
        self.answer_new_button.clicked.connect(self._show_latest_answer)
        history_row.addWidget(self.answer_new_button)
        self.answer_history_bar.hide()
        column.addWidget(self.answer_history_bar)
        self.answer_stack = QStackedWidget()
        self.answer_stack.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        self.answer_stack.setMinimumHeight(100)
        empty = QWidget()
        empty_layout = _vbox(empty, 14, 12)
        empty_layout.addStretch()
        empty_layout.addWidget(EmptyGlyph(True), 0, Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(_label("让真实经历成为回答依据", "emptyTitle"), 0, Qt.AlignmentFlag.AlignCenter)
        body = _label("导入简历、项目经历或岗位说明，获得结合资料的\n英文表达建议，并逐段核对中文。", "emptyBody", True)
        body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(body)
        import_btn = _button("导入面试资料", "upload")
        import_btn.clicked.connect(self._choose_files)
        empty_layout.addSpacing(6)
        empty_layout.addWidget(import_btn, 0, Qt.AlignmentFlag.AlignCenter)
        empty_layout.addStretch()
        self.answer_stack.addWidget(empty)
        answer_page = QWidget()
        answer_layout = _vbox(answer_page, 0, 4)
        self.answer_article = QWidget()
        article_layout = _vbox(self.answer_article, 4, 12)
        article_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.answer_scroll = self._answer_reader(self.answer_article)
        # One continuous reader replaces the two independently scrolling panes.
        # Keep the old handle as an alias for history/controller integrations.
        self.answer_chinese_scroll = self.answer_scroll
        self.answer_splitter = self.answer_article
        answer_layout.addWidget(self.answer_scroll, 1)
        self.answer_question_card = _frame("answerQuestionCard")
        self.answer_question_card.setToolTip("优先显示这条回答需要覆盖的问句；完整提问背景在回答下方保留。")
        question_layout = _vbox(self.answer_question_card, 12, 8)
        question_header = QHBoxLayout()
        question_header.addWidget(_label("本题 · 需要回答", "sectionEyebrow"), 1)
        self.answer_question_expand = QToolButton()
        self.answer_question_expand.setObjectName("quiet")
        self.answer_question_expand.setStyleSheet("font-size:11px; padding:1px 3px;")
        self.answer_question_expand.setText("核对原话")
        self.answer_question_expand.setToolTip("可选：核对未经整理的识别原话。问句和背景都在当前页面保留，无需展开才能阅读。")
        self.answer_question_expand.clicked.connect(self._show_answer_question_reader)
        question_header.addWidget(self.answer_question_expand)
        question_layout.addLayout(question_header)
        self.question_original = QuestionPreview("answerQuestionEn")
        question_layout.addWidget(self.question_original)
        self.question_translation = QuestionPreview("answerQuestionZh")
        question_layout.addWidget(self.question_translation)
        self.answer_question_card.hide()
        self.answer_question_card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        article_layout.addWidget(self.answer_question_card)
        language_row = QHBoxLayout()
        language_row.addWidget(_label("回答建议 · 英文 / 中文", "sectionEyebrow"))
        language_row.addStretch()
        self.answer_state = _label("", "muted")
        language_row.addWidget(self.answer_state)
        self.chinese_toggle = QToolButton()
        self.chinese_toggle.setObjectName("quiet")
        self.chinese_toggle.setStyleSheet("font-size:11px; padding:0px 3px;")
        self.chinese_toggle.setText("中文对照")
        self.chinese_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.chinese_toggle.setCheckable(True)
        self.chinese_toggle.setChecked(True)
        self.chinese_toggle.setArrowType(Qt.ArrowType.DownArrow)
        self.chinese_toggle.toggled.connect(self._toggle_chinese)
        language_row.addWidget(self.chinese_toggle)
        article_layout.addLayout(language_row)
        # These hidden plain-text handles preserve the aggregate answer for
        # accessibility integrations and tests; visible paragraphs are stable
        # paired widgets in the single article below.
        self.answer_english = _label("", wrap=True)
        self.answer_english.setStyleSheet("font-size: 17px; color: #E8F0FC; padding: 2px 0 8px 0;")
        self.answer_english.setParent(answer_page)
        self.answer_english.hide()
        self.answer_chinese = _label("", wrap=True)
        self.answer_chinese.setStyleSheet("font-size: 16px; color: #ABC3DB; padding: 2px 0 8px 0;")
        self.answer_chinese.setParent(answer_page)
        self.answer_chinese.hide()
        self.answer_paragraphs = QWidget()
        self.answer_paragraphs_layout = _vbox(self.answer_paragraphs, 0, 16)
        article_layout.addWidget(self.answer_paragraphs)
        self.answer_context_note = _label("建议基于当前问题与所选资料生成，请核对事实与个人经历。", "muted", True)
        article_layout.addWidget(self.answer_context_note)
        self.answer_question_background_card = _frame("answerQuestionBackgroundCard")
        background_layout = _vbox(self.answer_question_background_card, 12, 8)
        background_layout.addWidget(_label("提问背景 · 英文原文保留", "sectionEyebrow"))
        self.answer_question_background = QuestionPreview("answerQuestionBackground")
        self.answer_question_background.setStyleSheet("font-size:13px; color:#ABC3DB;")
        background_layout.addWidget(self.answer_question_background)
        self.answer_question_background_card.hide()
        article_layout.addWidget(self.answer_question_background_card)
        self.answer_stack.addWidget(answer_page)
        column.addWidget(self.answer_stack, 1)

        composer = _frame()
        self.question_panel = composer
        composer.setMinimumWidth(270)
        box = _vbox(composer, 10, 6)
        row = QHBoxLayout()
        row.addWidget(_label("当前问题", "cardTitle"))
        row.addStretch()
        self.expand_question_button = _button("浮出")
        self.expand_question_button.setToolTip("将完整当前问题作为独立窗口，可自由调整宽高。")
        self.expand_question_button.clicked.connect(lambda: self._float_reading_panel("question"))
        row.addWidget(self.expand_question_button)
        self.question_reader_button = _button("独立阅读")
        self.question_reader_button.setObjectName("quiet")
        self.question_reader_button.clicked.connect(self._show_question_reader)
        self.question_reader_button.setParent(composer)
        self.question_reader_button.hide()
        self.next_question_button = _button("下一题")
        self.next_question_button.setToolTip("默认自动保留12秒内的补充；需要立即开始新题时点击。")
        self.next_question_button.clicked.connect(self.new_question_requested.emit)
        row.addWidget(self.next_question_button)
        question_header_row = row
        box.addLayout(row)
        row = QHBoxLayout()
        self.phase_buttons = QButtonGroup(self)
        self.phase_buttons.setExclusive(True)
        self.presentation_phase_button = _button("汇报中")
        self.qa_phase_button = _button("问答 · 自动辅助")
        for phase, button in (("presentation", self.presentation_phase_button), ("qa", self.qa_phase_button)):
            button.setObjectName("phaseButton")
            button.setCheckable(True)
            button.setFixedHeight(30)
            button.setToolTip("汇报期间仅听译；切入问答后自动整理问题并给出文字建议，不会发声。")
            button.clicked.connect(lambda checked=False, value=phase: self._choose_interview_phase(value))
            self.phase_buttons.addButton(button)
            button.setParent(composer)
            button.hide()
        self.phase_note = _label("仅听译 · 不自动回答", "muted")
        row.addWidget(self.phase_note)
        row.addStretch()
        self.auto_checkbox = QCheckBox("自动回答")
        self.auto_checkbox.setToolTip("由控制器在一个完整问题结束后触发。可随时关闭。")
        self.auto_checkbox.toggled.connect(self._auto_answer_changed)
        row.addWidget(self.auto_checkbox)
        box.addLayout(row)
        row = QHBoxLayout()
        row.setSpacing(12)
        self.question_input = QPlainTextEdit()
        self.question_input.setPlaceholderText("完整问题将自动填入；也可以输入英文问题进行演练…")
        self.question_input.setMinimumHeight(64)
        self.question_input.setStyleSheet("font-size:15px;")
        self.question_input.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.question_input.textChanged.connect(self._grow_question)
        self.question_input.setTabChangesFocus(True)
        self.question_reader_stack = QStackedWidget()
        self.question_reader_stack.setMinimumHeight(64)
        self.question_reader_stack.addWidget(self.question_input)
        self.question_packet_view = QPlainTextEdit()
        self.question_packet_view.setReadOnly(True)
        self.question_packet_view.setStyleSheet("font-size:15px;")
        self.question_packet_view.setPlaceholderText("按说话人、句子整理后的完整问题将在这里显示。")
        self.question_reader_stack.addWidget(self.question_packet_view)
        row.addWidget(self.question_reader_stack, 1)
        actions = QHBoxLayout()
        actions.setSpacing(6)
        self.answer_button = _button("生成双语回答", "spark", True)
        self.answer_button.clicked.connect(self._answer)
        actions.addWidget(self.answer_button)
        self.cancel_button = _button("停止生成")
        self.cancel_button.clicked.connect(self.cancel_requested.emit)
        self.cancel_button.hide()
        actions.addWidget(self.cancel_button)
        self.practice_button = _button("演练")
        self.practice_button.setToolTip("填入演练问题")
        self.practice_button.setObjectName("quiet")
        self.practice_button.clicked.connect(self._practice)
        question_header_row.addWidget(self.practice_button)
        self.edit_question_button = _button("编辑问题")
        self.edit_question_button.setObjectName("quiet")
        self.edit_question_button.setCheckable(True)
        self.edit_question_button.setToolTip("修改实际发给回答模型的英文问题；文献检索会使用编辑后的问题。")
        self.edit_question_button.toggled.connect(self._toggle_question_editor)
        actions.addWidget(self.edit_question_button)
        actions.addStretch()
        box.addLayout(row, 1)
        box.addLayout(actions)
        self.reading_docks = {}
        for name, title, panel in (("transcript", "双语对话 · 拖动标题可移动", timeline),
                                   ("answer", "回答建议 · 完整连续阅读", reply),
                                   ("question", "当前问题 · 原始输入保留", composer)):
            dock = ReadingDock(title, self.dock_workspace)
            dock.setObjectName("readingDock_" + name)
            dock.setAllowedAreas(Qt.DockWidgetArea.AllDockWidgetAreas)
            dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable |
                             QDockWidget.DockWidgetFeature.DockWidgetFloatable |
                             QDockWidget.DockWidgetFeature.DockWidgetClosable)
            dock.setWidget(panel)
            dock.topLevelChanged.connect(lambda floating, value=name: self._dock_floating_changed(value, floating))
            self.reading_docks[name] = dock
        self.restore_reading_layout()
        return page

    def _make_library(self) -> QWidget:
        page = QWidget()
        layout = _vbox(page, 0, 14)
        actions = QHBoxLayout()
        self.library_count = _label("0 份资料", "badge")
        actions.addWidget(self.library_count)
        actions.addWidget(_label("PDF · DOCX · PPTX · TXT · Markdown", "muted"))
        actions.addStretch()
        clear = _button("清空索引")
        clear.setObjectName("danger")
        clear.clicked.connect(self._clear_library)
        actions.addWidget(clear)
        add = _button("导入资料", "upload", True)
        add.clicked.connect(self._choose_files)
        actions.addWidget(add)
        layout.addLayout(actions)
        card = _frame()
        box = _vbox(card, 20, 15)
        box.addWidget(_label("你的资料，成为回答的依据", "cardTitle"))
        box.addWidget(_label("导入简历、作品集、项目报告、岗位说明或论文。索引存于本机；选择云模型时，只发送你允许的相关片段。", "muted", True))
        box.addWidget(_divider())
        self.library_stack = QStackedWidget()
        empty = QWidget()
        empty_layout = _vbox(empty, 22, 12)
        empty_layout.addStretch()
        empty_layout.addWidget(EmptyGlyph(True), 0, Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(_label("建立你的面试资料库", "emptyTitle"), 0, Qt.AlignmentFlag.AlignCenter)
        note = _label("优先放入能说明真实经历、工作成果和个人职责的材料。\n扫描版 PDF 或复杂公式的提取效果，请在导入后检查。", "emptyBody", True)
        note.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_layout.addWidget(note)
        empty_layout.addStretch()
        self.library_stack.addWidget(empty)
        self.document_table = QTableWidget(0, 4)
        self.document_table.setHorizontalHeaderLabels(["文档", "格式", "文本片段", "状态"])
        self.document_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.document_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.document_table.setAlternatingRowColors(True)
        self.document_table.setShowGrid(False)
        self.document_table.verticalHeader().hide()
        self.document_table.verticalHeader().setDefaultSectionSize(54)
        header = self.document_table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for index in (1, 2, 3):
            header.setSectionResizeMode(index, QHeaderView.ResizeMode.ResizeToContents)
        self.library_stack.addWidget(self.document_table)
        box.addWidget(self.library_stack, 1)
        layout.addWidget(card, 1)
        import_help = _label("导入完成后，下次提问会自动检索相关资料；修改文件后需重新导入。", "note", True)
        import_help.setToolTip("资料会建立本地检索索引，并在回答时选择相关片段；这不是重新训练模型。扫描 PDF 需要先做 OCR，导入后请检查提取文字。")
        layout.addWidget(import_help)
        layout.addWidget(_label("检索到资料不等于已证实结论。英文回答旁的「来源」可查看本次检索证据。", "note", True))
        return page

    def _make_settings(self) -> QWidget:
        page = QWidget()
        layout = _vbox(page, 0, 12)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        grid = QGridLayout(content)
        grid.setContentsMargins(0, 0, 6, 0)
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        asr, form = self._settings_card("语音识别", "本地 Whisper · 可使用 NVIDIA GPU")
        self.whisper_select = _combo([(v, v) for v in ["small.en", "medium.en", "large-v3-turbo", "large-v3", "base.en", "tiny.en"]], True)
        form.addRow("识别模型", self.whisper_select)
        self.asr_device_select = _combo([("自动选择", "auto"), ("NVIDIA CUDA", "cuda"), ("CPU", "cpu")])
        form.addRow("运行设备", self.asr_device_select)
        self.compute_select = _combo([("INT8 + FP16 · 显存优先", "int8_float16"), ("FP16 · GPU", "float16"), ("INT8 · CPU", "int8")])
        form.addRow("计算精度", self.compute_select)
        self.threshold_input = QDoubleSpinBox()
        self.threshold_input.setDecimals(4)
        self.threshold_input.setRange(0.0001, 0.2)
        self.threshold_input.setSingleStep(0.001)
        form.addRow("语音阈值", self.threshold_input)
        self.caption_window_select = _combo([("3.6 秒 · 推荐 · 兼顾速度与完整性", "3.6"), ("3.2 秒 · 更快确认", "3.2"), ("3 秒 · 最快确认，断句可能更碎", "3"), ("6 秒 · 更多校订上下文", "6"), ("12 秒 · 最长校订上下文", "12")])
        self.caption_window_select.setToolTip("连续语音确认的最长窗口。较短窗口让正式说话人卡片和翻译更早更新；暂停可提前确认。即时草稿仍可选。切换后重新开始监听。")
        form.addRow("Whisper 校订窗口", self.caption_window_select)
        self.caption_update_mode_select = _combo([
            ("更快确认（GPU·单人）", "responsive"), ("原版节奏", "original")])
        self.caption_update_mode_select.setToolTip(
            "保留 large-v3-turbo、INT8+FP16、beam 搜索精度和常规 3.6 秒上下文。仅实际使用 CUDA、单一远端且校订窗口为 3.6 秒时启用："
            "本次监听首窗 3.0 秒，之后使用 3.6 秒窗口；已确认内容约每 1.6 秒更新一次，原版约 2.4 秒。"
            "首窗预算每次监听只用一次，不会在每次停顿后重置。CPU、多人及其他窗口仍使用原流程。"
            "约增加 50% 的 Whisper 计算量，也可能增加千问翻译请求及费用；不保证识别准确性。保存后需重新开始监听。")
        form.addRow("确认节奏", self.caption_update_mode_select)
        self.streaming_captions_checkbox = QCheckBox("逐词字幕 · 流式显示并随后校订")
        self.streaming_captions_checkbox.setToolTip("当前词可能被后续语音改写；最终校订由 Whisper 完成。切换后重新开始监听。")
        form.addRow(self.streaming_captions_checkbox)
        self.live_translation_checkbox = QCheckBox("边说边译 · 允许随语句修订中文")
        form.addRow(self.live_translation_checkbox)
        self.clean_speech_checkbox = QCheckBox("整理口语与标点 · 保留可展开原话")
        self.clean_speech_checkbox.setToolTip("删除明显语气词和重复起句，整理句子；保留否定、数字、术语及不确定语气。")
        form.addRow(self.clean_speech_checkbox)
        self.grammar_refinement_checkbox = QCheckBox("AI 语法润色（可选）")
        self.grammar_refinement_checkbox.setToolTip("在后台用本地模型润色，仅更新整理后的显示，不阻塞实时字幕；原话始终保留。复杂改写可能因保真校验未通过而跳过。")
        form.addRow(self.grammar_refinement_checkbox)
        self.caption_speaker_mode_select = _combo([
            ("1 人（默认）", "single_remote"),
            ("多人（声纹分组）", "acoustic"),
        ])
        self.caption_speaker_mode_select.setToolTip("仅识别 Zoom 对方的声音，不采集本机麦克风。1 人使用远端来源标签，不代表确认身份；多人按声纹匿名分组，可能有误。回环中的其他应用声音也会进入。切换后重新开始监听。")
        self.caption_speaker_mode_select.currentIndexChanged.connect(self._sync_caption_mode)
        form.addRow("对方人数", self.caption_speaker_mode_select)
        self.speaker_diarization_checkbox = QCheckBox("本地声纹区分说话人（匿名分组）")
        self.speaker_diarization_checkbox.setToolTip("按声音特征分组，可能误认；不能识别 Zoom 昵称。只采集当前输出设备，不包括自己的麦克风。")
        form.addRow(self.speaker_diarization_checkbox)
        self.prepare_speaker_button = _button("准备说话人模型", "export")
        self.prepare_speaker_button.setToolTip("显式下载并校验本地声纹模型；仅首次准备需要联网。")
        self.prepare_speaker_button.clicked.connect(self.install_speaker_requested.emit)
        form.addRow(self.prepare_speaker_button)
        self.cache_input = QLineEdit()
        self.cache_input.setPlaceholderText("留空使用默认模型缓存")
        form.addRow("模型缓存", self.cache_input)
        self.translation_engine_select = _combo([("实时优先 · Argos · CPU", "argos"),
                                                 ("千问 Qwen · 连续字幕云翻译", "qwen"),
                                                 ("NLLB 实验模式（仅非商业）", "nllb"),
                                                 ("精细翻译 · TranslateGemma 4B · 更慢/占显卡", "gemma")])
        self.translation_engine_select.setToolTip("字幕翻译与回答模型独立。保存后重新开始监听才切换当前引擎。Argos 快但有专业误译；Qwen 需单独授权上传已确认英文；NLLB 仅非商业；本地大模型也可能漏译否定。")
        self.translation_engine_select.currentIndexChanged.connect(self._sync_translation_settings)
        form.addRow("翻译引擎", self.translation_engine_select)
        self.translation_active_label = _label("当前听译 · 尚未开始监听", "muted", True)
        self.translation_active_label.setToolTip("这里显示本次监听实际使用的翻译引擎。更改下方设置后，需要保存并重新开始监听。")
        form.addRow(self.translation_active_label)
        self.translation_cloud_checkbox = QCheckBox("允许持续英译中使用千问云服务（独立授权）")
        self.translation_cloud_checkbox.setToolTip("连续字幕将已确认的英文语音文本发送至下方千问接口，可能持续产生 API 费用。取消勾选立即停止当前云听译；启用需保存后重新开始监听。此授权独立于回答模型的问题/资料授权；不会发送音频或整份文献。")
        self.translation_cloud_checkbox.toggled.connect(self._translation_consent_toggled)
        form.addRow(self.translation_cloud_checkbox)
        self.translation_cloud_note = _label("千问字幕：发送已确认的英文片段，不发送音频或整份资料；可能持续计费。原英文保留，译文仍需核对。", "muted", True)
        form.addRow(self.translation_cloud_note)
        self.translation_qwen_model_input = QLineEdit("qwen-flash")
        self.translation_qwen_model_input.setPlaceholderText("例如 qwen-flash；需为该接口实际支持的模型")
        form.addRow("字幕千问模型", self.translation_qwen_model_input)
        self.translation_qwen_base_input = QLineEdit("https://dashscope.aliyuncs.com/compatible-mode/v1")
        self.translation_qwen_base_input.setToolTip("使用此 HTTPS 接口下为千问保存的密钥。此地址独立于回答提供方，切换回答模型不会更改字幕接口或字幕上传授权。")
        form.addRow("字幕千问接口", self.translation_qwen_base_input)
        self.translation_timeout_input = QDoubleSpinBox()
        self.translation_timeout_input.setDecimals(1)
        self.translation_timeout_input.setRange(3.0, 60.0)
        self.translation_timeout_input.setSuffix(" 秒")
        self.translation_timeout_input.setValue(20.0)
        form.addRow("字幕请求超时", self.translation_timeout_input)
        translation = _button("准备离线英译中模型", "export")
        self.translation_prepare_button = translation
        translation.setToolTip("由控制器下载安装免费的本地翻译模型；仅首次准备需要联网。")
        translation.clicked.connect(self._prepare_translation)
        form.addRow(translation)
        self._sync_translation_settings()
        grid.addWidget(asr, 0, 0)

        model, form = self._settings_card("回答模型", "先检查可用性，再进入正式会话")
        self.provider_select = _combo([("本地 Ollama · 实验", "ollama"), ("Codex · 现有登录", "codex"),
                                      ("DeepSeek · 独立API", "deepseek"), ("千问 Qwen · 独立API", "qwen"),
                                      ("兼容 OpenAI 的 API", "openai_compatible")])
        self.provider_select.currentIndexChanged.connect(self._provider_changed)
        form.addRow("提供方", self.provider_select)
        self.ollama_select = _combo([(v, v) for v in ["qwen3:8b", "qwen2.5:7b", "qwen2.5:3b"]], True)
        self.ollama_select.setToolTip("本地模型的能力与延迟取决于硬件、模型和问题。先用自己的面试资料演练，并检查事实与个人贡献；更大的模型不保证更准确。")
        form.addRow("本地模型", self.ollama_select)
        self.answer_model_input = ModelChoice()
        self.answer_model_input.setPlaceholderText("留空使用当前可用的默认 GPT 模型")
        form.addRow("云模型", self.answer_model_input)
        self.answer_speed_select = _combo([("标准", "standard"), ("Fast（订阅额度约 2.5×）", "fast")])
        self.answer_speed_select.setToolTip("仅 Codex：Fast 约消耗 2.5 倍订阅额度，不会提高本地 ASR 速度；只有官方目录支持时才请求，否则报错。")
        form.addRow("回答服务档", self.answer_speed_select)
        form.addRow(_label("Fast 仅用于 Codex，约消耗 2.5× 订阅额度；不加快本地识别。目录不支持时会报错。", "muted", True))
        self.api_base_input = QLineEdit()
        self.api_base_input.setPlaceholderText("例如服务商的 https://…/v1")
        self.api_base_input.textEdited.connect(self._api_endpoint_edited)
        form.addRow("API 地址", self.api_base_input)
        self.api_key_input = QLineEdit()
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.setPlaceholderText("留空使用本机此地址的默认密钥；新密钥仅本次使用")
        self.api_key_input.setToolTip("在“选择回答模型”窗口保存或删除本机默认密钥；此输入框不会显示已保存的密钥。")
        form.addRow("API 密钥", self.api_key_input)
        self.free_quota_checkbox = QCheckBox("已在服务商设置免费额度及禁止超额计费")
        self.free_quota_checkbox.setToolTip("仅适用于独立 API。不能确认时请使用 Codex 订阅；程序不验证服务商的计费设置。")
        form.addRow(self.free_quota_checkbox)
        self.qwen_paid_checkbox = QCheckBox("允许千问使用付费余额")
        self.qwen_paid_checkbox.setToolTip("仅授权当前选择的千问官方 API 按量计费；不代表其他提供方获准付费。密钥仍只保留在本次运行。")
        form.addRow(self.qwen_paid_checkbox)
        self.brief_answer_checkbox = QCheckBox("简短模式 · 用于简单问题")
        self.brief_answer_checkbox.setToolTip("需要充分解释的问题建议关闭。回答仍先显示结论，再补充各子问的方法、依据、条件与限制；较长回答的完整中文需要更多时间。")
        form.addRow(self.brief_answer_checkbox)
        self.answer_words_input = QSpinBox()
        self.answer_words_input.setRange(60, 360)
        self.answer_words_input.setSingleStep(20)
        self.answer_words_input.setSuffix(" 英文词")
        self.answer_words_input.setToolTip("按问题复杂程度选择充分长度，不强制写满。中文回答与英文完整对应；数字、比较对象和实验条件不能为缩短回答而省略。")
        form.addRow("回答篇幅上限", self.answer_words_input)
        grid.addWidget(model, 0, 1)

        privacy, form = self._settings_card("数据与隐私", "资料库保留本地，云端发送需要明确开启")
        self.cloud_checkbox = QCheckBox("允许向所选云模型发送问题与面试背景")
        self.cloud_checkbox.setToolTip("Codex 和兼容 API 都属于云端提供方。")
        self.cloud_checkbox.toggled.connect(self._provider_changed)
        form.addRow(self.cloud_checkbox)
        self.evidence_checkbox = QCheckBox("同时发送检索到的相关资料片段")
        self.evidence_checkbox.toggled.connect(self._provider_changed)
        form.addRow(self.evidence_checkbox)
        self.cloud_note = _label("", "muted", True)
        form.addRow(self.cloud_note)
        form.addRow(_label("API 密钥仅在内存中使用。已有订阅不等于其他服务商的 API 余额；此界面不会购买额度。", "muted", True))
        grid.addWidget(privacy, 1, 0)

        research, form = self._settings_card("面试背景", "目标岗位、真实经历和表达重点，由你定义")
        self.profile_input = QPlainTextEdit()
        self.profile_input.setPlaceholderText("填写目标岗位、行业、面试类型和希望强调的能力。个人事实以已确认背景与资料为准。")
        self.profile_input.setMinimumHeight(106)
        form.addRow(self.profile_input)
        form.addRow(_label("候选人档案 · 请核对真实经历", "cardTitle", True))
        self.candidate_profile_input = QPlainTextEdit()
        self.candidate_profile_input.setPlaceholderText("填写已经确认的教育经历、工作职责、项目成果和个人贡献；导入一份资料不代表其中所有经历都属于你。")
        self.candidate_profile_input.setMinimumHeight(160)
        form.addRow(self.candidate_profile_input)
        self.prefer_prepared_answers_checkbox = QCheckBox("优先使用资料中的预备回答（高匹配时即时显示）")
        self.prefer_prepared_answers_checkbox.setToolTip("仅复用已导入资料中与当前问题高度匹配的原文回答，并显示来源；关闭后每题交给所选模型生成。")
        form.addRow(self.prefer_prepared_answers_checkbox)
        self.topmost_checkbox = QCheckBox("窗口始终置顶")
        form.addRow(self.topmost_checkbox)
        grid.addWidget(research, 1, 1)
        grid.setRowStretch(2, 1)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        actions = QHBoxLayout()
        self.model_status = _label("可用性以实际检查结果为准", "muted")
        self.model_status.setWordWrap(True)
        actions.addWidget(self.model_status, 1)
        check = _button("检查模型与设备", "refresh")
        check.clicked.connect(self._check_models)
        actions.addWidget(check)
        save = _button("保存设置", primary=True)
        save.clicked.connect(self._save_settings_clicked)
        actions.addWidget(save)
        layout.addLayout(actions)
        return page

    def _settings_card(self, title: str, subtitle: str) -> tuple[QFrame, QFormLayout]:
        card = _frame()
        card.setMinimumWidth(350)
        box = _vbox(card, 19, 8)
        box.addWidget(_label(title, "cardTitle"))
        box.addWidget(_label(subtitle, "muted", True))
        box.addSpacing(7)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.setVerticalSpacing(12)
        form.setHorizontalSpacing(13)
        box.addLayout(form)
        box.addStretch()
        return card, form

    def _make_footer(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        self.status_dot = _label("●")
        self.status_dot.setStyleSheet("color:#91A7AC;font-size:8px;")
        row.addWidget(self.status_dot)
        self.status_label = _label("准备就绪 · 请选择音频输出并检查模型", "muted")
        self.status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        row.addWidget(self.status_label, 1)
        self.engine_metric = _label("", "muted")
        row.addWidget(self.engine_metric)
        self.auto_save_checkbox = QCheckBox("自动保存")
        self.auto_save_checkbox.setChecked(False)
        self.auto_save_checkbox.setToolTip("可选：对话与回答仅保存在本机，不保存音频、密钥或模型设置。默认关闭。")
        self.auto_save_checkbox.toggled.connect(self.auto_save_changed.emit)
        row.addWidget(self.auto_save_checkbox)
        self.save_session_button = _button("保存对话", "source")
        self.save_session_button.setToolTip("将本次英文、译文、说话人、问题与回答保存到本机。不会上传对话。")
        self.save_session_button.setShortcut("Ctrl+S")
        self.save_session_button.clicked.connect(self.save_session_requested.emit)
        row.addWidget(self.save_session_button)
        self.session_history_button = _button("历史记录")
        self.session_history_button.setToolTip("查看或导出之前保存在本机的对话；打开历史不会启动监听或调用模型。")
        self.session_history_button.setShortcut("Ctrl+H")
        self.session_history_button.clicked.connect(self.history_requested.emit)
        row.addWidget(self.session_history_button)
        export = _button("导出文字", "export")
        export.setObjectName("quiet")
        export.clicked.connect(self.export_requested.emit)
        row.addWidget(export)
        return row

    def set_auto_save(self, enabled: bool) -> None:
        """Reflect a persisted choice without emitting a new user action."""
        with QSignalBlocker(self.auto_save_checkbox):
            self.auto_save_checkbox.setChecked(bool(enabled))

    def _navigate(self, index: int) -> None:
        if index != 0:
            self._hide_floating_readers()
        self.pages.setCurrentIndex(index)
        if index == 0:
            self._restore_floating_readers()
        for i, button in enumerate(self.nav_buttons):
            button.setChecked(i == index)
        titles = ["面试工作台", "面试资料库", "模型与设置"]
        subtitles = ["听懂问题 · 结合资料 · 自信表达", "把履历、岗位要求和项目经历放进同一份面试背景", "在速度、模型能力和资料发送范围之间自由选择"]
        self.page_title.setText(titles[index])
        self.page_subtitle.setText(subtitles[index])

    def _start(self) -> None:
        if not self.device_select.currentData():
            self.set_status("尚未选中音频输出设备。请先刷新设备。", "warning")
            return
        self._save_settings()
        self.start_requested.emit()

    def _answer(self) -> None:
        question = self.current_question()
        if not question:
            self.set_status("请等待完整问题，或输入英文问题进行演练。", "warning")
            self.question_input.setFocus()
            return
        self._save_settings()
        self.answer_requested.emit(question)

    def _practice(self) -> None:
        self.set_question("Tell me about a challenging project you worked on. What was your contribution, and how did you measure success?")
        self.set_status("已填入演练问题。点击「生成双语回答」运行真实模型；这不是实时会议记录。")
        self.question_input.setFocus()

    def _choose_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "导入面试资料", "",
            "面试资料 (*.pdf *.docx *.pptx *.txt *.md);;所有文件 (*)")
        if paths:
            self.import_requested.emit(paths)

    def _clear_library(self) -> None:
        choice = QMessageBox.question(self, "清空本地索引", "将移除所有已导入的文本索引，原始文件不会删除。继续吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No)
        if choice == QMessageBox.StandardButton.Yes:
            self.clear_requested.emit()

    def _copy_answer(self) -> None:
        if self._answer_en:
            QApplication.clipboard().setText(self._answer_en)
            self.set_status("英文回答已复制。")

    def _toggle_chinese(self, visible: bool) -> None:
        anchor = self._capture_answer_anchor()
        for pair in self._answer_pairs:
            pair.chinese.setVisible(visible and bool(pair.chinese.text()))
        self.chinese_toggle.setArrowType(Qt.ArrowType.DownArrow if visible else Qt.ArrowType.RightArrow)
        self._settle_answer_positions(self.answer_scroll.verticalScrollBar().value(), anchor=anchor)

    def _answer_reader(self, label: QWidget) -> QScrollArea:
        reader = QScrollArea()
        reader.setWidgetResizable(True)
        reader.setMinimumSize(0, 0)
        reader.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored)
        reader.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        reader.viewport().installEventFilter(self)
        reader.verticalScrollBar().sliderPressed.connect(self._cancel_answer_restore)
        reader.verticalScrollBar().actionTriggered.connect(self._cancel_answer_restore)
        content = QWidget()
        layout = _vbox(content, 0, 0)
        layout.addWidget(label)
        layout.addStretch()
        reader.setWidget(content)
        return reader

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.Wheel, QEvent.Type.KeyPress, QEvent.Type.MouseButtonPress):
            if any(watched is reader.viewport() for reader in
                     (getattr(self, "answer_scroll", None), getattr(self, "answer_chinese_scroll", None)) if reader is not None):
                self._cancel_answer_restore()
            timeline = getattr(self, "transcript_scroll", None)
            if timeline is not None and watched is timeline.viewport() and not self.follow_transcript.isChecked():
                self._timeline_scroll_generation += 1
                self._timeline_follow_pending = False
                self._timeline_position_timer.stop()
        return super().eventFilter(watched, event)

    def _resume_timeline_follow(self) -> None:
        self.follow_transcript.setChecked(True)
        self._settle_timeline_position()

    def _follow_latest_changed(self, enabled: bool) -> None:
        self._timeline_scroll_generation += 1
        self._timeline_follow_pending = False
        if enabled:
            self._settle_timeline_position()
        else:
            self._timeline_anchor = self._capture_timeline_anchor()
        if self._overlay:
            self._overlay.follow_action.setChecked(enabled)
        if hasattr(self, "provider_select"):
            self._save_settings()

    def _capture_timeline_anchor(self):
        bar = self.transcript_scroll.verticalScrollBar()
        position = bar.value()
        for index in range(self.transcript_layout.count() - 1):
            item = self.transcript_layout.itemAt(index).widget()
            if item is not None and item.y() + item.height() > position:
                return item, item.y() - position, position
        return None, 0, position

    def _timeline_range_changed(self, _minimum: int, maximum: int) -> None:
        # A range change is an intermediate Qt layout result, not a request to
        # jump to the bottom. Merge wrapping/translation passes into one frame.
        if (not self.follow_transcript.isChecked() and self._timeline_anchor is not None
                and not self._timeline_applying_position):
            self._restore_timeline_anchor(self._timeline_anchor)
        elif not self._timeline_applying_position and not self._timeline_position_timer.isActive():
            self._timeline_position_timer.start(0)

    def _restore_timeline_anchor(self, anchor) -> None:
        item, offset, position = anchor
        self._timeline_applying_position = True
        try:
            if item is not None and item in self._entries.values():
                position = item.y() - offset
            self.transcript_scroll.verticalScrollBar().setValue(position)
        finally:
            self._timeline_applying_position = False

    def _timeline_scroll_changed(self, value: int) -> None:
        bar = self.transcript_scroll.verticalScrollBar()
        self.latest_button.setText("已在最新位置" if value >= bar.maximum() - 2 else "↓ 回到最新")
        if not self.follow_transcript.isChecked() and not self._timeline_applying_position and not self._timeline_follow_pending:
            self._timeline_anchor = self._capture_timeline_anchor()

    def _toggle_history(self, checked: bool) -> None:
        """Legacy handle: the main conversation is always available now."""
        self.transcript_stack.show()
        self.history_controls_widget.show()

    def _toggle_draft(self, checked: bool) -> None:
        self.draft_panel.setVisible(checked)
        self.draft_toggle.setText("收起草稿 ▾" if checked else "查看草稿 ▸")
        self._settle_timeline_position()

    def _show_speaker_assignment(self, turn_id: str) -> None:
        if not turn_id:
            self.set_status("等待一段已识别语音后，可更正该段说话人。", "info")
            return
        known = {}
        original_identity = self._live_speaker_id if turn_id == self._live_turn_id else None
        for item in self._entries.values():
            if item.speaker_id:
                known[item.speaker_label] = str(item.speaker_id)
                if str(turn_id) in item.turn_ids:
                    original_identity = item.speaker_id
        labels = list(known) or ["说话人 1", "说话人 2"]
        dialog = QDialog(self)
        dialog.setWindowTitle("说话人及问答角色 · 人工指定")
        dialog.resize(520, 220)
        layout = _vbox(dialog, 20, 10)
        layout.addWidget(_label("声纹只区分相似声音。你可以补上称呼及问答角色；默认自动判断，不需要逐段操作。", "muted", True))
        form = QFormLayout()
        choice = QComboBox()
        choice.setObjectName("speakerAssignmentChoice")
        choice.setEditable(True)
        choice.addItems(labels)
        choice.setCurrentText(self._live_speaker_label if turn_id == self._live_turn_id and self._live_speaker_id else labels[0])
        form.addRow("说话人", choice)
        role = _combo([("自动判断", "auto"), ("面试官 · 可触发回答", "interviewer"),
                       ("不触发回答", "ignore")])
        role.setObjectName("speakerAssignmentRole")
        _select_value(role, self._speaker_roles.get(known.get(choice.currentText()), "auto"))
        form.addRow("问答角色", role)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted and choice.currentText().strip():
            selected = choice.currentText().strip()
            # Renaming an established acoustic voice must retain its cluster ID
            # so the chosen role also applies to that voice's later utterances.
            identifier = known.get(selected, str(original_identity) if original_identity else "manual:" + selected)
            self.update_speaker(turn_id, identifier, selected, status="manual")
            self.speaker_assigned.emit(str(turn_id), identifier, selected)
            self._speaker_roles[identifier] = str(role.currentData())
            self.speaker_role_changed.emit(identifier, str(role.currentData()))
        dialog.deleteLater()

    def set_preparation_status(self, text: str) -> None:
        self.preparation_status.setText(str(text))

    def update_speaker(self, turn_id: str, speaker_id, speaker_label: str,
                       confidence=None, status="confirmed") -> None:
        """Apply later acoustic evidence or a user assignment without changing text."""
        for item in set(self._entries.values()):
            if str(turn_id) in item.turn_ids:
                item.speaker_id, item.speaker_label = speaker_id, str(speaker_label)
                item.speaker_status = str(status)
                item.speaker.setText(str(speaker_label))
                item.speaker.setToolTip(item.speaker_status_tooltip(status))
        if str(turn_id) == self._live_turn_id:
            self._live_speaker_id, self._live_speaker_label = speaker_id, str(speaker_label)
            self._live_speaker_status = str(status)
            self.live_speaker.setText(str(speaker_label))
            self.live_speaker.setToolTip(TranscriptItem.speaker_status_tooltip(status))
            if self._overlay:
                self._overlay.set_caption(self._live_en, self._live_zh,
                    speaker_label=self._live_speaker_label, state=self._live_state,
                    units=self._live_units)
        latest = self._entries.get(self._latest_caption_id)
        if latest and str(turn_id) in latest.turn_ids:
            self.conversation_title.setText(f"{latest.speaker_label} · 双语对话")
        self._sync_overlay_conversation()

    def restore_reading_layout(self) -> None:
        if not hasattr(self, "reading_docks"):
            return
        self._hidden_floating_docks.clear()
        for dock in self.reading_docks.values():
            dock.setFloating(False)
            self.dock_workspace.removeDockWidget(dock)
        transcript, question, answer = (self.reading_docks[name] for name in ("transcript", "question", "answer"))
        self.dock_workspace.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, transcript)
        self.dock_workspace.splitDockWidget(transcript, answer, Qt.Orientation.Horizontal)
        self.dock_workspace.splitDockWidget(transcript, question, Qt.Orientation.Vertical)
        with QSignalBlocker(self.reading_mode):
            _select_value(self.reading_mode, "split")
        self._question_panel_expanded = True
        self._set_reading_mode()
        self._reading_size_timer.start(0)

    def _size_reading_docks(self) -> None:
        transcript, question, answer = (self.reading_docks[name] for name in ("transcript", "question", "answer"))
        if any(dock.isFloating() for dock in (transcript, question, answer)):
            return
        width, height = self.dock_workspace.width(), self.dock_workspace.height()
        self.dock_workspace.resizeDocks([transcript, answer], [int(width * .40), int(width * .60)], Qt.Orientation.Horizontal)
        self.dock_workspace.resizeDocks([transcript, question], [int(height * .58), int(height * .42)], Qt.Orientation.Vertical)

    def _float_reading_panel(self, name: str) -> None:
        dock = self.reading_docks[name]
        floating = not dock.isFloating()
        dock.setFloating(floating)
        dock.show()
        if floating:
            available = self.screen().availableGeometry()
            dock.resize(min(900 if name == "answer" else 700, available.width()),
                        min(780, available.height()))
            dock.raise_()

    def _dock_floating_changed(self, name: str, floating: bool) -> None:
        button = {"transcript": self.expand_transcript_button, "question": self.expand_question_button,
                  "answer": self.expand_answer_button}[name]
        button.setText("停靠" if floating else "浮出")
        if floating and name in self.reading_docks:
            dock = self.reading_docks[name]
            dock.setWindowFlag(Qt.WindowType.WindowMinMaxButtonsHint, True)
            dock.show()

    def _set_reading_mode(self, *args) -> None:
        if not hasattr(self, "reading_docks"):
            return
        mode = str(self.reading_mode.currentData() or "split")
        self._focus_mode = mode
        self.sidebar.setVisible(mode == "split")
        for name, dock in self.reading_docks.items():
            dock.setVisible(mode == "split" or mode == name)
        self.focus_question_button.setVisible(True)
        self.focus_controls_button.setVisible(True)
        self.timeline_panel.setStyleSheet("QLabel#transcriptEn { font-size:20px; } QLabel#transcriptZh { font-size:18px; }"
            if mode == "transcript" else "")
        self.question_input.setStyleSheet("font-size:20px;" if mode == "question" else "font-size:15px;")
        self.question_packet_view.setStyleSheet("font-size:20px;" if mode == "question" else "font-size:15px;")
        for pair in self._answer_pairs:
            self._style_answer_pair(pair)
        self.page_subtitle.setText("按说话轮次保留双语 · 原话可展开核对" if mode == "transcript" else
                                   "完整双语回答 · 连续滚动阅读" if mode == "answer" else
                                   "完整问题 · 可编辑并自动生成双语建议" if mode == "question" else
                                   "听懂问题 · 结合你的资料 · 英中逐段对照")

    def _toggle_panel_focus(self, mode: str) -> None:
        _select_value(self.reading_mode, "split" if self._focus_mode == mode else mode)

    def _toggle_question_panel(self) -> None:
        if self._focus_mode == "split":
            self._question_panel_expanded = not self._question_panel_expanded
            self._set_reading_mode()
            if self._question_panel_expanded:
                self._grow_question()
        else:
            self._show_question_reader()

    def _toggle_interview_controls(self) -> None:
        if self.interview_controls_panel.isVisible():
            self.interview_controls_panel.hide()
            return
        self._navigate(0)
        panel = self.interview_controls_panel
        panel.adjustSize()
        button = self.focus_controls_button
        below = button.mapToGlobal(QPoint(0, button.height() + 6))
        screen = button.screen().availableGeometry()
        x = max(screen.left(), min(below.x(), screen.right() - panel.width() + 1))
        y = below.y()
        if y + panel.height() > screen.bottom() + 1:
            y = max(screen.top(), button.mapToGlobal(QPoint(0, -panel.height() - 6)).y())
        panel.move(x, y)
        panel.show()
        panel.raise_()
        selected = self.inline_presentation_button if self._interview_phase == "presentation" else self.inline_qa_button
        selected.setFocus(Qt.FocusReason.PopupFocusReason)

    def _answer_from_mode_dropdown(self) -> None:
        self.interview_controls_panel.hide()
        self._answer()

    def _next_from_mode_dropdown(self) -> None:
        self.interview_controls_panel.hide()
        self.new_question_requested.emit()

    def _select_inline_phase(self, phase: str) -> None:
        self._choose_interview_phase(phase)
        self.interview_controls_panel.hide()

    @staticmethod
    def _allow_dialog_maximize(dialog: QDialog) -> None:
        dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowType.WindowMinMaxButtonsHint)

    def _show_question_reader(self) -> None:
        dialog = QDialog(self)
        self._allow_dialog_maximize(dialog)
        dialog.setWindowTitle("完整问题 · 随当前问题更新")
        dialog.resize(min(1200, self.width()), min(800, self.height()))
        layout = _vbox(dialog, 22, 12)
        layout.addWidget(_label("当前问题", "cardTitle"))
        reader = QPlainTextEdit()
        reader.setReadOnly(True)
        reader.setStyleSheet("font-size:20px;")
        reader.setPlainText(self._question_reader_text())
        layout.addWidget(reader, 1)
        raw_toggle = QCheckBox("展开识别原话（用于核对）")
        raw_reader = QPlainTextEdit()
        raw_reader.setReadOnly(True)
        raw_reader.setPlainText(str(self._question_packet.get("raw_question_en", "")))
        raw_reader.setMaximumHeight(150)
        raw_reader.hide()
        raw_toggle.setVisible(bool(raw_reader.toPlainText()))
        raw_toggle.toggled.connect(raw_reader.setVisible)
        layout.addWidget(raw_toggle)
        layout.addWidget(raw_reader)
        def refresh():
            position = reader.verticalScrollBar().value()
            reader.setPlainText(self._question_reader_text())
            raw_reader.setPlainText(str(self._question_packet.get("raw_question_en", "")))
            reader.verticalScrollBar().setValue(position)
        self.question_input.textChanged.connect(refresh)
        self.question_packet_changed.connect(refresh)
        done = _button("返回工作台")
        done.clicked.connect(dialog.accept)
        layout.addWidget(done, 0, Qt.AlignmentFlag.AlignRight)
        try:
            dialog.exec()
        finally:
            self.question_input.textChanged.disconnect(refresh)
            self.question_packet_changed.disconnect(refresh)
            dialog.deleteLater()

    def _cancel_answer_restore(self, *args) -> None:
        self._answer_scroll_generation += 1

    def _settle_timeline_position(self, anchor=None) -> None:
        """Keep stable cards; either follow latest or anchor the reading place."""
        self._sync_overlay_conversation()
        if not self.follow_transcript.isChecked() and anchor is not None:
            self._timeline_anchor = anchor
        self._timeline_scroll_generation += 1
        self._timeline_follow_pending = True
        # Qt already double-buffers widget painting and merges dirty regions.
        # Disabling/re-enabling this viewport invalidates every card, including
        # unchanged English/Chinese, and made each callback flash the reader.
        # Coalesce only the position/layout work; keep retained pixels intact.
        self._timeline_finish_timer.stop()
        self._timeline_position_timer.start(0)

    def _latest_timeline_unit(self):
        card = self._entries.get(self._latest_caption_id)
        if card is None:
            return None
        return card._unit_widgets.get(self._latest_caption_id) or card

    def _apply_timeline_position(self, final: bool = False) -> None:
        generation = self._timeline_scroll_generation
        self.transcript_layout.activate()
        self._timeline_applying_position = True
        try:
            if self.follow_transcript.isChecked():
                # New confirmed speech is appended below the existing cards.
                # Follow its end only after coalescing Qt's wrapping passes;
                # never relocate the latest English line to the reader's top.
                bar = self.transcript_scroll.verticalScrollBar()
                target = bar.maximum()
                latest = self._latest_timeline_unit()
                view = self.transcript_scroll.viewport()
                if latest is not None and latest.height() <= view.height():
                    # Card padding can clip the first English pixels although
                    # the unit fits. Compute one final target before scrolling;
                    # never first jump down and then correct upward in-frame.
                    top = latest.mapTo(self.transcript_scroll.widget(), QPoint(0, 0)).y()
                    target = min(target, top)
                bar.setValue(target)
            elif self._timeline_anchor is not None:
                self._restore_timeline_anchor(self._timeline_anchor)
        finally:
            self._timeline_applying_position = False
            if final:
                self._timeline_follow_pending = False
            else:
                # The content's height-for-width pass may follow its parent.
                # Reconcile the same anchor in the following layout turn.
                self._timeline_finish_generation = generation
                self._timeline_finish_timer.start(0)

    def _finish_timeline_position(self) -> None:
        if self._timeline_finish_generation == self._timeline_scroll_generation:
            self._apply_timeline_position(final=True)

    def _capture_answer_anchor(self):
        reader = self.answer_scroll
        if reader.verticalScrollBar().value() == 0:
            return None
        candidates = [self.question_original, self.question_translation]
        for pair in self._answer_pairs:
            candidates.extend((pair.english, pair.chinese))
        for widget in candidates:
            if widget.isVisible():
                y = widget.mapTo(reader.viewport(), QPoint(0, 0)).y()
                if y + widget.height() > 0 and y < reader.viewport().height():
                    return widget, y
        return None

    def _settle_answer_positions(self, english: int, chinese: int = 0, *, anchor=None) -> None:
        self._answer_scroll_generation += 1
        generation = self._answer_scroll_generation
        def settle(final=False):
            if generation != self._answer_scroll_generation:
                return
            page = self.answer_stack.currentWidget()
            if page.layout():
                page.layout().activate()
            reader = self.answer_scroll
            reader.widget().layout().activate()
            self.answer_article.layout().activate()
            self.answer_paragraphs_layout.activate()
            bar = reader.verticalScrollBar()
            if anchor is not None and anchor[0].isVisible():
                current_y = anchor[0].mapTo(reader.viewport(), QPoint(0, 0)).y()
                bar.setValue(bar.value() + current_y - anchor[1])
            else:
                bar.setValue(english)
            if not final:
                QTimer.singleShot(0, lambda: settle(True))
        QTimer.singleShot(0, settle)

    def _device_changed(self, index: int) -> None:
        if hasattr(self, "provider_select"):
            self._save_settings()

    def _sync_caption_mode(self, *args) -> None:
        with QSignalBlocker(self.capture_mode_select):
            _select_value(self.capture_mode_select, _combo_value(self.caption_speaker_mode_select))

    def _quick_caption_mode_changed(self, *args) -> None:
        if not hasattr(self, "caption_speaker_mode_select"):
            return
        with QSignalBlocker(self.caption_speaker_mode_select):
            _select_value(self.caption_speaker_mode_select, _combo_value(self.capture_mode_select))
        self._save_settings()
        if self._listening:
            self.set_status("对方人数已保存；下次开始监听时生效。", "info")

    def _endpoint_identity(self, provider: str, endpoint: str) -> str:
        # Normalize without reading a vault. Invalid/partial edits still differ
        # from the former endpoint and must not retain its credential draft.
        from .credential_store import credential_target
        try:
            return credential_target(provider, endpoint)
        except ValueError:
            return "invalid:" + str(endpoint).strip()

    def _api_endpoint_edited(self, endpoint: str) -> None:
        provider = _combo_value(self.provider_select)
        if provider not in ("openai_compatible", "deepseek", "qwen"):
            return
        identity = self._endpoint_identity(provider, endpoint)
        if identity == self._api_endpoint_identity:
            return
        self._api_endpoint_identity = identity
        scope = "compatible" if provider == "openai_compatible" else provider
        sensitive = (bool(self.api_key_input.text()) or bool(self._session_api_keys.get(scope))
                     or self.cloud_checkbox.isChecked() or self.free_quota_checkbox.isChecked()
                     or self.qwen_paid_checkbox.isChecked())
        self.api_key_input.clear()
        self._session_api_keys.pop(scope, None)
        self._settings["api_key"] = ""
        self._session_api_bases[provider] = endpoint
        # A new destination needs its own explicit authorization. Revoke first
        # so queued answers cannot send an old endpoint's key or context there.
        for control in (self.cloud_checkbox, self.free_quota_checkbox, self.qwen_paid_checkbox):
            with QSignalBlocker(control):
                control.setChecked(False)
        self._provider_changed()
        if sensitive:
            self._save_settings()
            self.set_status("API 地址已更改：已清除本次密钥并关闭云端与计费授权，请重新核对设置。", "warning")

    def _provider_changed(self, *args) -> None:
        if not hasattr(self, "cloud_checkbox"):
            return
        provider = _combo_value(self.provider_select)
        if hasattr(self, "api_key_input") and provider != self._last_provider:
            self._session_api_bases[self._last_provider] = self.api_base_input.text()
            self._session_answer_models[self._last_provider] = self.answer_model_input.text()
            if self._last_provider in ("openai_compatible", "deepseek", "qwen"):
                scope = "compatible" if self._last_provider == "openai_compatible" else self._last_provider
                self._session_api_keys[scope] = self.api_key_input.text()
            scope = "compatible" if provider == "openai_compatible" else provider
            self.api_key_input.setText(self._session_api_keys.get(scope, ""))
            self.api_base_input.setText(self._session_api_bases.get(provider, ""))
            self.answer_model_input.clear()
            self.answer_model_input.setText(self._session_answer_models.get(provider, ""))
            self._last_provider = provider
        self._api_endpoint_identity = self._endpoint_identity(provider, self.api_base_input.text())
        cloud = provider != "ollama"
        api = provider in ("openai_compatible", "deepseek", "qwen")
        self.ollama_select.setEnabled(provider == "ollama")
        self.answer_model_input.setEnabled(cloud)
        self.answer_speed_select.setEnabled(provider == "codex")
        self.api_base_input.setEnabled(api)
        self.api_key_input.setEnabled(api)
        self.free_quota_checkbox.setEnabled(api)
        self.qwen_paid_checkbox.setEnabled(provider == "qwen")
        if provider == "deepseek":
            self.api_base_input.setPlaceholderText("留空使用 https://api.deepseek.com/v1")
            self.answer_model_input.setPlaceholderText("查询服务商模型目录并选择可用模型")
        elif provider == "qwen":
            self.api_base_input.setPlaceholderText("阿里云百炼：https://dashscope.aliyuncs.com/compatible-mode/v1")
            self.answer_model_input.setPlaceholderText("请在选择回答模型窗口查询并选择千问模型")
        elif provider == "codex":
            self.answer_model_input.setPlaceholderText("留空使用当前可用的默认 GPT 模型")
        self.evidence_checkbox.setEnabled(self.cloud_checkbox.isChecked())
        if not cloud:
            title = "本地模式 · Ollama"
            detail = "问题与资料片段不发送到云模型。请先确认本地模型已安装，并用自己的面试资料演练。任何模型的建议都应核对事实与个人经历。"
        elif not self.cloud_checkbox.isChecked():
            title = "云端未授权"
            detail = "当前选择云模型，但尚未允许发送问题。生成前请明确开启，或切回本地 Ollama。"
        else:
            name = "Codex" if provider == "codex" else "API"
            title = f"混合模式 · {name}"
            detail = ("向所选云模型发送问题、面试背景及检索到的相关片段。" if self.evidence_checkbox.isChecked()
                      else "向所选云模型发送问题与面试背景；资料片段不会发送。")
        self.privacy_chip.setText(title)
        self.privacy_chip.setToolTip(detail)
        self.cloud_note.setText(detail)

    def _check_models(self) -> None:
        self._save_settings()
        self.check_models_requested.emit()

    def _prepare_translation(self) -> None:
        self._save_settings()
        self.install_translation_requested.emit()

    def _choose_interview_phase(self, phase: str) -> None:
        if phase == self._interview_phase:
            self.set_interview_phase(phase)
            return
        self.set_interview_phase(phase)
        # The controller can enable its auto-answer preference when the user
        # deliberately enters Q&A; programmatic set_interview_phase is silent.
        self.interview_phase_changed.emit(phase)
        self._save_settings()

    def _auto_answer_changed(self, *args) -> None:
        self.set_interview_phase(self._interview_phase)
        self._save_settings()

    def set_interview_phase(self, phase: str) -> None:
        phase = phase if phase in ("presentation", "qa") else "qa"
        self._interview_phase = phase
        self._settings["interview_phase"] = phase
        self.presentation_phase_button.setChecked(phase == "presentation")
        self.qa_phase_button.setChecked(phase == "qa")
        self.inline_presentation_button.setChecked(phase == "presentation")
        self.inline_qa_button.setChecked(phase == "qa")
        with QSignalBlocker(self.dropdown_auto_checkbox):
            self.dropdown_auto_checkbox.setChecked(self.auto_checkbox.isChecked())
        self.dropdown_auto_checkbox.setEnabled(phase == "qa")
        self.auto_checkbox.setEnabled(phase == "qa")
        self.auto_checkbox.setToolTip("汇报中仅听译，不自动回答；切入问答后可启用自动整理与回答。")
        self.phase_note.setText("仅听译 · 不自动回答" if phase == "presentation" else
                                "自动建议 · 不发声" if self.auto_checkbox.isChecked() else "手动回答 · 不发声")
        self.focus_controls_button.setText("汇报 · 仅听译" if phase == "presentation" else
                                          "问答 · 自动辅助" if self.auto_checkbox.isChecked() else "问答 · 手动回答")

    def set_answer_model(self, name: str) -> None:
        value = str(name or "").strip()
        self._answer_model_name = value
        if self._answer_active_id in self._answer_history:
            self._answer_history[self._answer_active_id]["model"] = value
            if self._answer_displayed_id != self._answer_active_id:
                return
        self.answer_model_label.setText("模型 · " + (value or "尚未确认"))
        self.answer_model_label.setToolTip(value or "尚未确认实际回答模型")

    def _save_settings_clicked(self) -> None:
        self._save_settings()
        self.set_status("已提交设置。识别/翻译引擎、对话场景与音频设备的更改将在下次开始监听时生效。")

    def _sync_translation_settings(self, *args) -> None:
        if not hasattr(self, "translation_cloud_checkbox"):
            return
        cloud = _combo_value(self.translation_engine_select) == "qwen"
        # Always allow withdrawing upload consent, even after selecting a local
        # engine for the next capture while a cloud capture is still active.
        self.translation_cloud_checkbox.setEnabled(True)
        for control in (self.translation_cloud_note, self.translation_qwen_model_input, self.translation_qwen_base_input,
                        self.translation_timeout_input):
            control.setEnabled(cloud)
        if hasattr(self, "translation_prepare_button"):
            self.translation_prepare_button.setText("检查千问字幕接口" if cloud else "准备离线英译中模型")
            self.translation_prepare_button.setToolTip(
                "检查已授权的千问字幕翻译设置与接口；使用此接口保存的密钥，可能产生调用费用。" if cloud else
                "由控制器下载安装免费的本地翻译模型；仅首次准备需要联网。")

    def _translation_consent_toggled(self, enabled: bool) -> None:
        # Revocation takes effect immediately. A grant still uses the explicit
        # Save/restart path and must never start a new upload from this toggle.
        if not enabled:
            self._save_settings()

    def set_translation_engine_status(self, engine: str, model: str = "", note: str = "") -> None:
        """Controller reports actual active translation; editing settings cannot fake it."""
        names = {"argos": "Argos · 本地 CPU", "nllb": "NLLB · 本地 CPU",
                 "gemma": "TranslateGemma 4B · 本地", "qwen": "千问 · 云翻译"}
        title = names.get(str(engine), str(engine) or "尚未开始监听")
        if model:
            title += " · " + str(model)
        if note:
            title += " · " + str(note)
        self.translation_active_label.setText("当前听译 · " + title)
        self.translation_active_label.setToolTip("本次监听实际使用：" + title + "。更改设置需保存后重新开始监听。")

    def _save_settings(self, *args) -> None:
        if not hasattr(self, "profile_input"):
            return
        self._settings.update(self.get_settings())
        self._apply_topmost(bool(self._settings.get("topmost")))
        self.settings_saved.emit(self.get_settings())

    def _apply_topmost(self, enabled: bool) -> None:
        current = bool(self.windowFlags() & Qt.WindowType.WindowStaysOnTopHint)
        if enabled != current:
            was_visible = self.isVisible()
            self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, enabled)
            if was_visible:
                self.show()

    def get_settings(self) -> dict:
        if hasattr(self, "provider_select"):
            current_provider = _combo_value(self.provider_select)
            if current_provider in ("openai_compatible", "deepseek", "qwen"):
                scope = "compatible" if current_provider == "openai_compatible" else current_provider
                key = self.api_key_input.text()
                if key:
                    self._session_api_keys[scope] = key
                else:
                    self._session_api_keys.pop(scope, None)
        result = dict(self._settings)
        result.update({
            "provider": _combo_value(self.provider_select),
            "ollama_model": _combo_value(self.ollama_select),
            "whisper_size": _combo_value(self.whisper_select),
            "asr_device": _combo_value(self.asr_device_select),
            "compute_type": _combo_value(self.compute_select),
            "translation_engine": _combo_value(self.translation_engine_select),
            "translation_cloud_consent": self.translation_cloud_checkbox.isChecked(),
            "translation_qwen_model": self.translation_qwen_model_input.text().strip(),
            "translation_qwen_base_url": self.translation_qwen_base_input.text().strip().rstrip("/"),
            "translation_timeout": self.translation_timeout_input.value(),
            "auto_answer": self.auto_checkbox.isChecked(),
            "auto_save_sessions": self.auto_save_checkbox.isChecked(),
            "follow_latest": self.follow_transcript.isChecked(),
            "interview_phase": self._interview_phase,
            "topmost": self.topmost_checkbox.isChecked(),
            "speaker_id": str(self.device_select.currentData() or ""),
            "speech_threshold": self.threshold_input.value(),
            "caption_window_seconds": float(_combo_value(self.caption_window_select)),
            "caption_update_mode": _combo_value(self.caption_update_mode_select),
            "speaker_diarization": self.speaker_diarization_checkbox.isChecked(),
            "caption_speaker_mode": _combo_value(self.caption_speaker_mode_select),
            "streaming_captions": self.streaming_captions_checkbox.isChecked(),
            "live_translation": self.live_translation_checkbox.isChecked(),
            "clean_speech": self.clean_speech_checkbox.isChecked(),
            "grammar_refinement": self.grammar_refinement_checkbox.isChecked(),
            "answer_model": self.answer_model_input.text().strip(),
            "answer_speed": _combo_value(self.answer_speed_select),
            "api_base_url": self.api_base_input.text().strip(),
            "api_key": self.api_key_input.text(),
            "session_api_keys": dict(self._session_api_keys),
            "free_quota_confirmed": self.free_quota_checkbox.isChecked(),
            "qwen_paid_usage_authorized": self.qwen_paid_checkbox.isChecked(),
            "brief_answer": self.brief_answer_checkbox.isChecked(),
            "max_answer_words": self.answer_words_input.value(),
            "cloud_consent": self.cloud_checkbox.isChecked(),
            "share_evidence": self.evidence_checkbox.isChecked(),
            "model_cache_dir": self.cache_input.text().strip(),
            "research_profile": self.profile_input.toPlainText().strip(),
            "candidate_profile": self.candidate_profile_input.toPlainText().strip(),
            "prefer_prepared_answers": self.prefer_prepared_answers_checkbox.isChecked(),
        })
        return result

    def set_settings(self, settings: dict) -> None:
        settings = dict(settings)
        # Provider services/model picker use the canonical compatible name;
        # this historic widget value remains an accepted UI alias.
        if str(settings.get("provider", "")).strip().lower() == "compatible":
            settings["provider"] = "openai_compatible"
        self._settings.update(settings)
        if "session_api_keys" in settings:
            self._session_api_keys = dict(settings.get("session_api_keys") or {})
        defaults = {
            "provider": "ollama", "ollama_model": "qwen3:8b", "whisper_size": "small.en",
            "asr_device": "auto", "compute_type": "int8_float16", "translation_engine": "argos", "auto_answer": True,
            "translation_cloud_consent": False,
            "auto_save_sessions": False,
            "translation_qwen_model": "qwen-flash",
            "translation_qwen_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "translation_timeout": 20.0,
            "interview_phase": "qa",
            "topmost": False, "speech_threshold": 0.007, "cloud_consent": False,
            "share_evidence": False, "answer_model": "", "api_base_url": "", "api_key": "",
            "answer_speed": "standard",
            "caption_window_seconds": 3.6,
            "caption_update_mode": "original",
            "follow_latest": True,
            "speaker_diarization": True,
            "caption_speaker_mode": "single_remote",
            "streaming_captions": True, "live_translation": False, "clean_speech": True,
            "grammar_refinement": False,
            "free_quota_confirmed": False, "brief_answer": False,
            "max_answer_words": 220,
            "qwen_paid_usage_authorized": False,
            "model_cache_dir": "", "research_profile": "",
            "candidate_profile": "",
            "prefer_prepared_answers": True,
        }
        value = {**defaults, **self._settings}
        self.set_auto_save(bool(value["auto_save_sessions"]))
        self._last_provider = str(value["provider"])
        if "api_key" not in settings and value["provider"] in ("openai_compatible", "deepseek", "qwen"):
            scope = "compatible" if value["provider"] == "openai_compatible" else str(value["provider"])
            value["api_key"] = self._session_api_keys.get(scope, "")
        controls = [self.provider_select, self.ollama_select, self.whisper_select,
                    self.asr_device_select, self.compute_select, self.auto_checkbox,
                    self.topmost_checkbox, self.cloud_checkbox, self.evidence_checkbox,
                    self.device_select, self.translation_engine_select, self.answer_speed_select,
                    self.free_quota_checkbox, self.qwen_paid_checkbox, self.brief_answer_checkbox, self.caption_window_select,
                    self.caption_update_mode_select,
                    self.speaker_diarization_checkbox, self.caption_speaker_mode_select, self.streaming_captions_checkbox,
                    self.live_translation_checkbox, self.clean_speech_checkbox,
                    self.grammar_refinement_checkbox, self.follow_transcript, self.capture_mode_select,
                    self.prefer_prepared_answers_checkbox, self.answer_words_input,
                    self.translation_cloud_checkbox, self.translation_qwen_model_input,
                    self.translation_qwen_base_input, self.translation_timeout_input]
        blockers = [QSignalBlocker(control) for control in controls]
        for control, key in [(self.provider_select, "provider"), (self.ollama_select, "ollama_model"),
                             (self.whisper_select, "whisper_size"), (self.asr_device_select, "asr_device"),
                             (self.compute_select, "compute_type"),
                             (self.translation_engine_select, "translation_engine"),
                             (self.caption_speaker_mode_select, "caption_speaker_mode"),
                             (self.caption_update_mode_select, "caption_update_mode"),
                             (self.answer_speed_select, "answer_speed")]:
            selected = value[key]
            if key == "caption_update_mode" and selected not in ("responsive", "original"):
                selected = "original"
            _select_value(control, selected)
        for control, key in [(self.auto_checkbox, "auto_answer"), (self.topmost_checkbox, "topmost"),
                             (self.follow_transcript, "follow_latest"),
                             (self.prefer_prepared_answers_checkbox, "prefer_prepared_answers"),
                             (self.cloud_checkbox, "cloud_consent"), (self.evidence_checkbox, "share_evidence"),
                             (self.translation_cloud_checkbox, "translation_cloud_consent"),
                             (self.free_quota_checkbox, "free_quota_confirmed"), (self.brief_answer_checkbox, "brief_answer"),
                             (self.qwen_paid_checkbox, "qwen_paid_usage_authorized"),
                             (self.speaker_diarization_checkbox, "speaker_diarization"),
                             (self.streaming_captions_checkbox, "streaming_captions"),
                             (self.live_translation_checkbox, "live_translation"),
                             (self.clean_speech_checkbox, "clean_speech"),
                             (self.grammar_refinement_checkbox, "grammar_refinement")]:
            control.setChecked(bool(value[key]))
        self.threshold_input.setValue(float(value["speech_threshold"]))
        self.translation_qwen_model_input.setText(str(value["translation_qwen_model"] or ""))
        self.translation_qwen_base_input.setText(str(value["translation_qwen_base_url"] or ""))
        self.translation_timeout_input.setValue(float(value["translation_timeout"]))
        self.answer_words_input.setValue(int(value["max_answer_words"]))
        _select_value(self.caption_window_select, format(float(value["caption_window_seconds"]), "g"))
        self.answer_model_input.setText(str(value["answer_model"] or ""))
        self.api_base_input.setText(str(value["api_base_url"] or ""))
        self.api_key_input.setText(str(value["api_key"] or ""))
        self.cache_input.setText(str(value["model_cache_dir"] or ""))
        self.profile_input.setPlainText(str(value["research_profile"] or ""))
        self.candidate_profile_input.setPlainText(str(value["candidate_profile"] or ""))
        self._sync_caption_mode()
        self._sync_translation_settings()
        self.context_label.setToolTip(str(value["research_profile"] or ""))
        if "speaker_id" in value:
            _select_value(self.device_select, str(value["speaker_id"]))
        del blockers
        self._provider_changed()
        self.set_interview_phase(value["interview_phase"])
        self._apply_topmost(bool(value["topmost"]))

    def set_devices(self, devices: list[tuple[str, str]], selected_id: str = "") -> None:
        previous = selected_id or str(self.device_select.currentData() or self._settings.get("speaker_id", ""))
        blocker = QSignalBlocker(self.device_select)
        self.device_select.clear()
        for identifier, title in devices:
            self.device_select.addItem(str(title), str(identifier))
        if not devices:
            self.device_select.addItem("未发现音频输出设备", "")
        _select_value(self.device_select, previous)
        del blocker
        self.start_button.setEnabled(bool(devices) and not self._listening)

    def append_transcript(self, entry_id: str, en: str, zh: str = "", partial: bool = False,
                          speaker_id=None, speaker_label=None, turn_id=None,
                          raw_en=None, speaker_status=None, translation_pending: bool | None = None) -> None:
        self.update_message(entry_id, en, zh, partial=partial, speaker_id=speaker_id,
                            speaker_label=speaker_label, turn_id=turn_id,
                            raw_en=raw_en, speaker_status=speaker_status,
                            translation_pending=translation_pending)

    def update_message(self, entry_id: str, en: str, zh: str = "", source=None, *,
                       partial: bool = False, speaker_id=None, speaker_label=None, turn_id=None,
                       raw_en=None, speaker_status=None, translation_pending: bool | None = None) -> None:
        """Update a caption without rebuilding history; acoustic turns share a card.

        Caption IDs remain individually addressable for asynchronous translation.
        ``speaker_id`` is an anonymous acoustic cluster, never a person's identity.
        """
        key = str(entry_id)
        if partial:
            # Draft hypotheses can revise word by word. Never insert them into
            # committed cards or change the main reader's height/order.
            self.set_live_translation(en, zh, speaker_label or "当前发言", "正在听译",
                raw=str(raw_en or en), turn_id=str(turn_id or ""), speaker_id=speaker_id,
                speaker_status=speaker_status)
            return
        anchor = None if self.follow_transcript.isChecked() else self._capture_timeline_anchor()
        new_entry = key not in self._entries
        is_latest = new_entry or key == self._latest_caption_id
        if is_latest:
            self._latest_caption_id = key
        if key in self._entries:
            item = self._entries[key]
        else:
            item = self._turn_cards.get(str(turn_id)) if turn_id else None
            # A very long uninterrupted turn gets a continuation card so a
            # single message never becomes an unbounded meeting transcript.
            if item and (len(item.english.text()) > 900 or
                         (speaker_id and item.speaker_id and speaker_id != item.speaker_id)):
                item = None
            if item is None:
                item = TranscriptItem(en, zh, partial, speaker_id, speaker_label)
                item.assign_requested.connect(lambda card=item: self._show_speaker_assignment(
                    next(iter(card.turn_ids), "")))
                self.transcript_layout.insertWidget(self.transcript_layout.count() - 1, item)
                if turn_id:
                    self._turn_cards[str(turn_id)] = item
            self._entries[key] = item
            if len(set(self._entries.values())) > 120:
                old = next(iter(self._entries.values()))
                for old_key in [identifier for identifier, card in self._entries.items() if card is old]:
                    self._entries.pop(old_key, None)
                for old_turn in [identifier for identifier, card in self._turn_cards.items() if card is old]:
                    self._turn_cards.pop(old_turn, None)
                self.transcript_layout.removeWidget(old)
                old.deleteLater()
        old_en, old_zh, _ = item._segments.get(key, ("", "", False))
        # The controller may keep an earlier translation visible while the
        # English source grows. Its presence alone does not mean that the new
        # source revision has been translated; preserve that explicit state.
        if translation_pending is True:
            item._pending_translations.add(key)
        elif translation_pending is False:
            item._pending_translations.discard(key)
        elif en != old_en and not zh:
            item._pending_translations.add(key)
        elif zh:
            item._pending_translations.discard(key)
        if turn_id:
            item.turn_ids.add(str(turn_id))
        changed = item.update_segment(key, en, zh or old_zh, partial, speaker_id, speaker_label, raw_en, speaker_status)
        self.transcript_stack.setCurrentIndex(1)
        if new_entry or changed:
            self._settle_timeline_position(anchor)
        if is_latest and (new_entry or changed):
            self.conversation_title.setText(f"{item.speaker_label} · 双语对话")
            self.set_live_translation(item.english.text(),
                "\n".join(p[1] for p in item._segments.values() if p[1]),
                item.speaker_label, "正在听译" if partial else "已成句",
                raw=item.raw_text.text(), turn_id=str(turn_id or ""), speaker_id=speaker_id,
                speaker_status=item.speaker_status, bilingual_units=[(p[0], p[1]) for p in item._segments.values()])

    def update_translation(self, entry_id: str, zh: str, *, translation_pending: bool = False) -> None:
        item = self._entries.get(str(entry_id))
        if item:
            anchor = None if self.follow_transcript.isChecked() else self._capture_timeline_anchor()
            en, old_zh, partial = item._segments.get(str(entry_id), (item.english.text(), "", False))
            if translation_pending:
                item._pending_translations.add(str(entry_id))
            else:
                item._pending_translations.discard(str(entry_id))
            changed = item.update_segment(str(entry_id), en, zh or "未生成译文", partial)
            if changed:
                self._settle_timeline_position(anchor)
            if changed and str(entry_id) == self._latest_caption_id and self._live_state == "已成句":
                self.set_live_translation(item.english.text(),
                    "\n".join(p[1] for p in item._segments.values() if p[1]),
                    item.speaker_label, "已确认片段 · 正在更新中文" if translation_pending else "已成句", raw=item.raw_text.text(),
                    turn_id=self._live_turn_id, speaker_id=item.speaker_id,
                    speaker_status=item.speaker_status, bilingual_units=[(p[0], p[1]) for p in item._segments.values()])

    def remove_transcript(self, entry_id: str) -> None:
        """Discard a rejected provisional card; committed export rows are separate."""
        anchor = None if self.follow_transcript.isChecked() else self._capture_timeline_anchor()
        item = self._entries.pop(str(entry_id), None)
        if item is not None:
            item._segments.pop(str(entry_id), None)
            item._raw_segments.pop(str(entry_id), None)
            item._pending_translations.discard(str(entry_id))
            if item._segments:
                identifier, part = next(iter(item._segments.items()))
                item.update_segment(identifier, *part)
            else:
                for turn in [identifier for identifier, card in self._turn_cards.items() if card is item]:
                    self._turn_cards.pop(turn, None)
                self.transcript_layout.removeWidget(item)
                item.deleteLater()
        if not self._entries:
            self.transcript_stack.setCurrentIndex(0)
        self._settle_timeline_position(anchor)

    def set_question(self, question: str) -> None:
        # Explicit controller/user changes (including Next question's empty
        # text) replace the current draft and release any manual edit latch.
        self._manual_question_active = False
        self._background_question_packet = {}
        self._question_edit_base = question.strip()
        if not question.strip() or self._question_packet.get("question_en", "").strip() != question.strip():
            self._question_packet = {}
            self.question_reader_stack.setCurrentIndex(0)
            with QSignalBlocker(self.edit_question_button):
                self.edit_question_button.setChecked(False)
            self.edit_question_button.setText("编辑问题")
        if self.current_question() != question.strip():
            self.question_input.setPlainText(question)
        if self._overlay:
            self._sync_overlay_question()

    def set_question_packet(self, packet: dict) -> None:
        """Show the same speaker/turn boundaries that accompany the model request."""
        incoming = dict(packet or {})
        if (str(incoming.get("question_en", "")).strip()
                == str(self._question_packet.get("question_en", "")).strip()
                and not incoming.get("question_zh") and self._question_packet.get("question_zh")):
            incoming["question_zh"] = self._question_packet["question_zh"]
        changed_in_editor = (self.edit_question_button.isChecked()
            and self.current_question() != self._question_edit_base)
        if self._manual_question_active or changed_in_editor:
            # Audio finals, acoustic relabelling, and preparation notifications
            # may arrive while the human edits. Keep their exact selected input;
            # later packets remain available in the background until Next.
            self._manual_question_active = True
            self._background_question_packet = incoming
            if self._overlay:
                self._sync_overlay_question()
        else:
            self._question_packet = incoming
            self.set_question(str(self._question_packet.get("question_en", "")))
        position = self.question_packet_view.verticalScrollBar().value()
        self.question_packet_view.setPlainText(self._question_reader_text())
        self.question_packet_view.verticalScrollBar().setValue(position)
        if self._question_packet and not self.edit_question_button.isChecked():
            self.question_reader_stack.setCurrentIndex(1)
        self._grow_question()
        self.question_packet_changed.emit()

    def set_current_question_translation(self, question_en: str, question_zh: str) -> bool:
        """Retain a translation only for its exact current ask.

        The Current Question reader stays English-only. The selected answer
        keeps its own bilingual question; a delayed callback cannot translate
        a newer question or a human-edited input.
        """
        source, translation = str(question_en).strip(), str(question_zh).strip()
        if (not source or not translation or source != self.current_question().strip()
                or self._manual_question_active or self.edit_question_button.isChecked()):
            return False
        if str(self._question_packet.get("question_en", source)).strip() != source:
            return False
        self._question_packet = dict(self._question_packet or {"question_en": source})
        self._question_packet["question_zh"] = translation
        reader = self.question_packet_view
        position = reader.verticalScrollBar().value()
        updated = self._question_reader_text()
        if reader.toPlainText() != updated:
            reader.setPlainText(updated)
            reader.verticalScrollBar().setValue(position)
            self.question_reader_stack.setCurrentIndex(1)
            self._grow_question()
        return True

    def _question_reader_text(self) -> str:
        packet = self._question_packet
        if not packet or self.edit_question_button.isChecked():
            return self.current_question()
        from .transcript_text import split_sentences
        roles = {"interviewer": "提问", "candidate": "候选人", "ignored": "背景", "unknown": "当前发言"}
        background = str(packet.get("background_en", "")).strip()
        core_question = str(packet.get("answer_question_en", "")).strip()
        if str(packet.get("interaction_kind", "")) == "setup":
            # Confirmation tags such as 'right?' can belong to setup rather
            # than be a request for an interview answer.
            source = background or str(packet.get("question_en", "")).strip()
            return ("背景 · 等待实际提问\n\n正在等待对方提出完整问题…\n\n背景\n"
                + "\n".join(split_sentences(source)))
        if background or core_question:
            # These are semantic sections prepared with the frozen AI input,
            # not a list made from ASR windows or punctuation guesses.
            labels = list(dict.fromkeys(str(turn.get("label") or "说话人待补认")
                for turn in packet.get("turns", [])))
            speakers = {(str(turn.get("speaker_id", "")), str(turn.get("label", "")))
                for turn in packet.get("turns", [])}
            multiple_speakers = len(speakers) > 1
            heading = "多位说话人" if multiple_speakers else ("、".join(labels) or "当前发言")
            kind = str(packet.get("interaction_kind", ""))
            heading += " · 对话协调" if kind == "coordination" else " · 提问"
            blocks = []
            blocks.append(heading)
            if core_question:
                title = "需要回答（合并整理，逐人来源见下方）" if multiple_speakers else "需要回答"
                blocks.append(title + "\n" + "\n".join(split_sentences(core_question)))
            if background:
                title = "背景（合并整理）" if multiple_speakers else "背景"
                blocks.append(title + "\n" + "\n".join(split_sentences(background)))
            if multiple_speakers:
                from .transcript_text import repair_fragment_boundaries
                groups = []
                for turn in packet.get("turns", []):
                    identity = (str(turn.get("speaker_id", "")),
                        str(turn.get("label") or "说话人待补认"), str(turn.get("role", "unknown")))
                    text = str(turn.get("clean_en", "")).strip() or " ".join(
                        str(sentence).strip() for sentence in turn.get("sentences", []) if str(sentence).strip())
                    # Rejoin adjacent ASR cuts by the same voice, while keeping
                    # speaker changes as source boundaries. Never assign a
                    # combined multi-person question to every speaker.
                    if groups and groups[-1][0] == identity:
                        groups[-1][1].append(text)
                    else:
                        groups.append((identity, [text]))
                sources = []
                for (_speaker_id, label, role), fragments in groups:
                    source = repair_fragment_boundaries(" ".join(fragments))
                    if source:
                        sources.append(f"{label} · {roles.get(role, '当前发言')}\n"
                            + "\n".join(split_sentences(source)))
                if sources:
                    blocks.append("逐人来源\n" + "\n\n".join(sources))
            return "\n\n".join(blocks)
        blocks = []
        groups = []
        for turn in packet.get("turns", []):
            identity = (str(turn.get("turn_id", "")), str(turn.get("speaker_id", "")),
                        str(turn.get("label", "")), str(turn.get("role", "unknown")))
            text = str(turn.get("clean_en", "")).strip() or " ".join(
                str(sentence).strip() for sentence in turn.get("sentences", []) if str(sentence).strip())
            if groups and groups[-1][0] == identity:
                groups[-1][2].append(text)
            else:
                groups.append((identity, turn, [text]))
        for identity, turn, fragments in groups:
            label = str(turn.get("label") or "说话人待补认")
            role = roles.get(str(turn.get("role", "unknown")), "当前发言")
            # A single continuous question uses the completely restored form,
            # so ASR chunk boundaries never pretend to be sentence boundaries.
            source = str(packet.get("question_en", "")) if len(groups) == 1 else " ".join(fragments)
            lines = split_sentences(source)
            if lines:
                blocks.append(f"{label} · {role}\n" + "\n".join(lines))
        if not blocks:
            blocks.append("当前问题\n" + "\n".join(split_sentences(str(packet.get("question_en", "")))))
        subquestions = [str(value).strip() for value in packet.get("subquestions", []) if str(value).strip()]
        # Avoid rendering the very same questions twice in the small composer.
        shown = "\n".join(blocks)
        if len(subquestions) > 1 and not all(question in shown for question in subquestions):
            blocks.append("本题需要回答\n" + "\n".join(f"• {line}" for line in subquestions))
        return "\n\n".join(blocks)

    def _toggle_question_editor(self, checked: bool) -> None:
        if checked:
            self._question_edit_base = self.current_question()
        if not checked and (self._manual_question_active or
                self.current_question() != str(self._question_packet.get("question_en", "")).strip()):
            from .transcript_text import split_sentences
            self._manual_question_active = True
            self._question_packet["question_en"] = self.current_question()
            self._question_packet["turns"] = [{"label": "当前问题（手动编辑）", "role": "interviewer",
                "sentences": list(split_sentences(self.current_question()))}]
            self._question_packet["subquestions"] = []
            for field in ("background_en", "answer_question_en", "interaction_kind"):
                self._question_packet.pop(field, None)
            self._question_packet.pop("question_zh", None)
            self.question_packet_view.setPlainText(self._question_reader_text())
        self.question_reader_stack.setCurrentIndex(0 if checked or not self._question_packet else 1)
        self.edit_question_button.setText("返回整理视图" if checked else "编辑问题")
        if checked:
            self.question_input.setFocus()

    def _grow_question(self):
        # The user owns dock sizes; incoming words must not move separators.
        if hasattr(self, "question_panel"):
            self.question_panel.updateGeometry()

    def set_live_caption(self, english: str):
        self.set_live_translation(english, "", self._live_speaker_label, "正在听译",
            raw=english, turn_id=self._live_turn_id, speaker_id=self._live_speaker_id)

    @staticmethod
    def _replace_live_text(reader, text: str) -> None:
        """Revise the same reader without rebuilding cards or stealing selection."""
        if reader.toPlainText() == str(text):
            return
        reader.setPlainText(str(text))
        reader.verticalScrollBar().setValue(reader.verticalScrollBar().maximum())

    def set_live_translation(self, en: str, zh: str = "", speaker_label: str = "",
                             state: str = "正在听译", *, raw: str = "",
                             turn_id: str = "", speaker_id=None, speaker_status=None,
                             bilingual_units=None) -> None:
        """Update optional compact draft/overlay; committed cards keep their layout."""
        self._live_en, self._live_zh = str(en), str(zh)
        self._live_units = list(bilingual_units or [])
        self._live_state = str(state) if str(en).strip() else "等待语音"
        if speaker_status is not None or self._live_turn_id != str(turn_id):
            self._live_speaker_status = speaker_status
        self._live_turn_id = str(turn_id)
        self._live_speaker_id = speaker_id
        if speaker_label:
            self._live_speaker_label = str(speaker_label)
        elif not str(en).strip():
            self._live_speaker_label = "说话人待补认"
        self.live_speaker.setText(self._live_speaker_label)
        self.live_speaker.setToolTip(TranscriptItem.speaker_status_tooltip(self._live_speaker_status))
        self.live_sentence_state.setText(self._live_state)
        self._replace_live_text(self.live_caption, en)
        self._replace_live_text(self.live_translation, zh)
        self._replace_live_text(self.live_raw, raw or en)
        if self._overlay:
            self._overlay.set_caption(en, zh, speaker_label=self._live_speaker_label, state=self._live_state,
                units=self._live_units)

    def show_subtitle_overlay(self):
        if self._overlay is None:
            from .overlay import SubtitleOverlay
            self._overlay = SubtitleOverlay()
            self._overlay.restore_requested.connect(self.restore_workspace)
            self._overlay.next_requested.connect(self.new_question_requested.emit)
            self._overlay.phase_requested.connect(self._choose_interview_phase)
            self._overlay.stop_requested.connect(self.stop_requested.emit)
            self._overlay.move_to_top(self.screen())
        self._overlay.set_caption(self._live_en, self._live_zh,
            speaker_label=self._live_speaker_label, state=self._live_state, units=self._live_units)
        self._overlay.follow_action.setChecked(self.follow_transcript.isChecked())
        self._overlay.set_answer(self._answer_en, self._answer_zh,
                                 answer_pairs=self._answer_visible_pairs)
        self._sync_overlay_question()
        self._sync_overlay_conversation()
        self._overlay.state.setText("监听中" if self._listening else "未监听")
        self.hide()
        self._overlay.show()
        self._overlay.raise_()

    def _sync_overlay_question(self) -> None:
        if self._overlay is None:
            return
        # A pinned or historical answer must retain its own question, even as
        # the next question is being heard or edited in the main workspace.
        paired_answer = self._answer_displayed_id is not None or self._answer_en or self._answer_zh
        self._overlay.set_question(self._answer_question_en if paired_answer else self.current_question())

    def _sync_overlay_conversation(self) -> None:
        if self._overlay is None or not hasattr(self._overlay, "set_conversation"):
            return
        rows, seen = [], set()
        for entry_id, item in self._entries.items():
            if id(item) in seen:
                continue
            seen.add(id(item))
            pending = bool(item._pending_translations) or any(not part[1].strip() for part in item._segments.values())
            rows.append({"entry_id": entry_id, "en": item.english.text(), "zh": item.chinese.text(),
                         "speaker_label": item.speaker_label,
                         "state": "等待中文" if pending else "已成句",
                         "units": [(part[0], part[1]) for part in item._segments.values()]})
        self._overlay.set_conversation(rows[-8:])

    def restore_workspace(self):
        self.showNormal()
        self.raise_()
        if self._overlay:
            self._overlay.hide()

    def showEvent(self, event):
        super().showEvent(event)
        self._restore_floating_readers()
        if not self._reading_layout_checked:
            self._reading_layout_checked = True
            self.dock_workspace.layout().activate()
            self._size_reading_docks()
            self._reading_size_timer.start(0)

    def _hide_floating_readers(self) -> None:
        for name, dock in getattr(self, "reading_docks", {}).items():
            if dock.isFloating() and dock.isVisible():
                self._hidden_floating_docks.add(name)
                dock.hide()

    def _restore_floating_readers(self) -> None:
        if not self.isVisible() or self.pages.currentIndex() != 0:
            return
        for name in list(self._hidden_floating_docks):
            dock = self.reading_docks[name]
            if dock.isFloating() and self._focus_mode in ("split", name):
                dock.show()
            self._hidden_floating_docks.discard(name)

    def hideEvent(self, event):
        self._hide_floating_readers()
        super().hideEvent(event)

    def closeEvent(self, event):
        if self._overlay:
            self._overlay.hide()
            self._overlay.deleteLater()
            self._overlay = None
        super().closeEvent(event)

    def current_question(self) -> str:
        return self.question_input.toPlainText().strip()

    def _update_answer_question(self, question_en: str | None, question_zh: str,
                                question_preview_en: str | None = None,
                                question_raw_en: str | None = None,
                                *, clear: bool = False) -> None:
        if question_en is not None:
            self._answer_question_en = str(question_en).strip()
            self._answer_question_preview_en = ((str(question_preview_en).strip()
                                                if question_preview_en is not None else "")
                                               or self._answer_question_en)
            if question_raw_en is not None:
                self._answer_question_raw_en = str(question_raw_en)
            elif clear or self._answer_question_en != self._answer_question_raw_en:
                self._answer_question_raw_en = ""
        elif clear:
            self._answer_question_en = ""
            self._answer_question_preview_en = ""
            self._answer_question_raw_en = ""
        self._answer_question_zh = question_zh
        present = bool(self._answer_question_en or question_zh)
        self.answer_question_card.setVisible(present)
        # Display source-backed explicit asks ahead of the answer, not a second
        # copy of every introductory/setup sentence. Never shorten by length or
        # trust an arbitrary paraphrase as a replacement for the actual ask.
        from .turns import explicit_question_focus
        focus, background = explicit_question_focus(self._answer_question_en)
        if not focus:
            # Match the packet builder's conservative fallback for technical
            # tokens such as S0. at a sentence boundary. Unknown phrasing is
            # kept whole by analyze_question, never deleted for lack of a '?'.
            from .interview_context import analyze_question
            structure = analyze_question(self._answer_question_en)
            focus, background = structure.answer_question_en, structure.background_en
        preview = self._answer_question_preview_en
        separated = bool(focus and background and preview
                         and " ".join(focus.split()) == " ".join(preview.split()))
        foreground = focus if separated else self._answer_question_en
        self.question_original.set_preview("本题 EN  ·  " + foreground
                                          if foreground else "本题 EN  ·  未提供英文问题")
        # Provider question_zh translates the complete source; label that scope
        # honestly instead of pretending it is only the shorter English focus.
        chinese_prefix = "中（含提问背景）  ·  " if separated else "中  ·  "
        self.question_translation.set_preview(chinese_prefix + question_zh
                                             if question_zh else "中  ·  等待本题中文释义…")
        self.answer_question_background.set_preview(background if separated else "")
        self.answer_question_background_card.setVisible(separated)
        self.answer_question_card.updateGeometry()
        dialog = self._answer_question_dialog
        if (dialog is not None and dialog.property("questionSnapshot") == self._answer_question_en
                and dialog.property("requestSnapshot") == self._answer_displayed_id):
            reader = dialog.findChild(QPlainTextEdit, "answerQuestionChineseFull")
            if reader is not None and reader.toPlainText() != question_zh:
                position = reader.verticalScrollBar().value()
                reader.setPlainText(question_zh)
                reader.verticalScrollBar().setValue(position)

    def _show_answer_question_reader(self) -> None:
        """Expand the answer's snapshot, never the next question being heard."""
        if not (self._answer_question_en or self._answer_question_zh):
            return
        if self._answer_question_dialog is not None:
            self._answer_question_dialog.close()
        dialog = QDialog(self)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self._allow_dialog_maximize(dialog)
        dialog.setWindowTitle("本条回答的问题 · 英中对照")
        dialog.setProperty("questionSnapshot", self._answer_question_en)
        dialog.setProperty("requestSnapshot", self._answer_displayed_id)
        dialog.resize(min(1000, self.width()), min(740, self.height()))
        layout = _vbox(dialog, 22, 12)
        layout.addWidget(_label("需要回答的核心问题", "cardTitle"))
        layout.addWidget(_label("英文问句与中文释义 · 完整背景和识别原话在下方展开", "muted"))
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        for title, name, value, placeholder in (
            ("ENGLISH · 核心问句", "answerQuestionEnglishFocus", self._answer_question_preview_en, "未提供英文问题"),
            ("中文 · 问题释义", "answerQuestionChineseFull", self._answer_question_zh, "等待本题中文释义…"),
        ):
            panel = QWidget()
            column = _vbox(panel, 0, 6)
            column.addWidget(_label(title, "sectionEyebrow"))
            reader = QPlainTextEdit()
            reader.setObjectName(name)
            reader.setReadOnly(True)
            reader.setStyleSheet("font-size:18px;")
            reader.setPlaceholderText(placeholder)
            reader.setPlainText(value)
            column.addWidget(reader, 1)
            splitter.addWidget(panel)
        layout.addWidget(splitter, 1)
        background_toggle = QCheckBox("展开完整英文背景")
        background_reader = QPlainTextEdit()
        background_reader.setObjectName("answerQuestionEnglishFull")
        background_reader.setReadOnly(True)
        background_reader.setPlainText(self._answer_question_en)
        background_reader.setMaximumHeight(145)
        background_reader.hide()
        background_toggle.toggled.connect(background_reader.setVisible)
        layout.addWidget(background_toggle)
        layout.addWidget(background_reader)
        if self._answer_question_raw_en:
            raw_toggle = QCheckBox("展开识别原话（可能含结巴或错误断句）")
            raw_reader = QPlainTextEdit()
            raw_reader.setObjectName("answerQuestionRawFull")
            raw_reader.setReadOnly(True)
            raw_reader.setPlainText(self._answer_question_raw_en)
            raw_reader.setMaximumHeight(145)
            raw_reader.hide()
            raw_toggle.toggled.connect(raw_reader.setVisible)
            layout.addWidget(raw_toggle)
            layout.addWidget(raw_reader)
        done = _button("返回回答")
        done.clicked.connect(dialog.close)
        layout.addWidget(done, 0, Qt.AlignmentFlag.AlignRight)
        self._answer_question_dialog = dialog
        dialog.finished.connect(lambda: setattr(self, "_answer_question_dialog", None))
        dialog.show()

    def set_answer(self, en: str = "", zh: str = "", question_zh: str = "",
                   sources: list | None = None, streaming: bool = False,
                   question_en: str | None = None,
                   question_preview_en: str | None = None,
                   question_raw_en: str | None = None,
                   request_id=None, cancelled: bool = False,
                   answer_pairs: list[dict] | None = None) -> None:
        """Keep completed responses readable while a later request streams.

        Request IDs bind answer text, sources, model and question snapshots.
        Legacy callers without an ID retain the direct presentation API.
        """
        if request_id is None:
            self._display_answer(en, zh, question_zh, sources, streaming,
                                 question_en, question_preview_en, question_raw_en,
                                 answer_pairs=answer_pairs)
            return
        key = str(request_id)
        if key not in self._answer_history:
            self._answer_history[key] = {"number": len(self._answer_history) + 1,
                "en": "", "zh": "", "question_zh": "", "question_en": "",
                "question_preview_en": "", "question_raw_en": "", "sources": [],
                "streaming": True, "cancelled": False, "model": self._answer_model_name,
                "position_en": 0, "position_zh": 0, "answer_pairs": None}
        state = self._answer_history[key]
        latest = self._answer_history.get(self._answer_active_id)
        if latest is None or state["number"] >= latest["number"]:
            self._answer_active_id = key
        newest = key == self._answer_active_id
        if cancelled:
            state.update(streaming=False, cancelled=True)
        else:
            state.update(en=en, zh=zh, question_zh=question_zh, streaming=streaming, cancelled=False)
            if answer_pairs is not None or not streaming:
                # Own a snapshot: later provider mutations cannot rewrite history.
                # A final flat safety/fallback result also replaces streamed
                # pairs; otherwise an earlier provisional claim could survive.
                state["answer_pairs"] = ([dict(pair) for pair in answer_pairs if isinstance(pair, dict)]
                                         if answer_pairs is not None else None)
            for name, value in (("question_en", question_en), ("question_preview_en", question_preview_en),
                                ("question_raw_en", question_raw_en)):
                if value is not None:
                    state[name] = value
            if sources is not None:
                state["sources"] = list(sources)
        # A new request alone never clears readable text. Follow is on by
        # default, and switches only once the latest request has visible words.
        # Older delayed callbacks remain bound to their history entry.
        displayed = self._answer_history.get(self._answer_displayed_id)
        displayed_readable = bool(displayed and not displayed["cancelled"]
                                  and self._answer_has_visible_text(displayed))
        should_show = (self._answer_displayed_id is None or self._answer_displayed_id == key
                       or (newest and not displayed_readable))
        if (self.follow_answers_checkbox.isChecked() and newest
                and self._answer_has_visible_text(state) and not cancelled):
            should_show = True
        if cancelled and self._answer_displayed_id == key:
            previous = [other for other, value in self._answer_history.items()
                        if other != key and (value["en"] or value["zh"])
                        and not value["cancelled"] and not value["streaming"]]
            if previous:
                self._show_answer_record(previous[-1])
            else:
                self._answer_displayed_id = None
                self._display_answer(question_en="", sources=[])
            should_show = False
        if should_show:
            self._show_answer_record(key)
        self._refresh_answer_history()

    def _show_answer_record(self, key: str) -> None:
        if key not in self._answer_history:
            return
        switched = key != self._answer_displayed_id
        if switched and self._answer_displayed_id in self._answer_history:
            previous = self._answer_history[self._answer_displayed_id]
            previous["position_en"] = self.answer_scroll.verticalScrollBar().value()
            previous["position_zh"] = self.answer_chinese_scroll.verticalScrollBar().value()
        state = self._answer_history[key]
        self._answer_displayed_id = key
        self._display_answer(state["en"], state["zh"], state["question_zh"], state["sources"],
                             state["streaming"], state["question_en"], state["question_preview_en"],
                             state["question_raw_en"], answer_pairs=state["answer_pairs"])
        value = state["model"]
        self.answer_model_label.setText("模型 · " + (value or "尚未确认"))
        self.answer_model_label.setToolTip(value or "尚未确认实际回答模型")
        if state["cancelled"]:
            self.answer_state.setText("已停止 · 未完成")
        if switched:
            self._settle_answer_positions(state["position_en"], state["position_zh"])
        self._refresh_answer_history()

    def _refresh_answer_history(self) -> None:
        self.answer_history_bar.setVisible(bool(self._answer_history))
        with QSignalBlocker(self.answer_history_select):
            for key, state in self._answer_history.items():
                index = self.answer_history_select.findData(key)
                status = "已停止" if state["cancelled"] else "生成中" if state["streaming"] else "已完成"
                question = " ".join((state["question_preview_en"] or state["question_en"]).split())
                title = f"{state['number']} · {status} · {question[:75]}"
                if index < 0:
                    self.answer_history_select.addItem(title, key)
                    index = self.answer_history_select.count() - 1
                else:
                    self.answer_history_select.setItemText(index, title)
                self.answer_history_select.setItemData(index, question, Qt.ItemDataRole.ToolTipRole)
            selected = self.answer_history_select.findData(self._answer_displayed_id)
            if selected >= 0:
                self.answer_history_select.setCurrentIndex(selected)
            else:
                self.answer_history_select.setCurrentIndex(-1)
        latest = self._answer_history.get(self._answer_active_id)
        pending = bool(latest and self._answer_active_id != self._answer_displayed_id and not latest["cancelled"])
        self.answer_new_button.setVisible(pending)
        self.answer_new_button.setText("● 查看新回答" if pending and self._answer_has_visible_text(latest) else "下一题生成中")
        self.answer_new_button.setToolTip("等待首段文字期间保留当前答案；也可点击查看下一题，旧答案仍在回答记录中。")

    def _select_answer_history(self, index: int) -> None:
        key = self.answer_history_select.itemData(index)
        if key is not None:
            # Explicit history reading remains pinned even when follow was on.
            with QSignalBlocker(self.follow_answers_checkbox):
                self.follow_answers_checkbox.setChecked(False)
            self._show_answer_record(str(key))

    def _show_latest_answer(self) -> None:
        if self._answer_active_id is not None:
            self._show_answer_record(self._answer_active_id)

    def _follow_new_answers_changed(self, checked: bool) -> None:
        latest = self._answer_history.get(self._answer_active_id)
        if checked and latest and self._answer_has_visible_text(latest) and not latest["cancelled"]:
            self._show_latest_answer()

    @staticmethod
    def _answer_has_visible_text(state: dict) -> bool:
        from .display_text import clean_answer_text
        from .stable_text import completed_text
        streaming = state.get("streaming", False)
        pairs = state.get("answer_pairs")
        if pairs is not None:
            return any(completed_text(clean_answer_text(pair.get(field, ""), streaming=streaming),
                                      final=not streaming or bool(pair.get(field + "_complete")))
                       for pair in pairs for field in ("en", "zh"))
        return any(completed_text(clean_answer_text(state.get(field, ""), streaming=streaming),
                                  final=not streaming) for field in ("en", "zh"))

    def clear_answer_history(self) -> None:
        """Explicit session reset; cancelling one request must never call this."""
        self._answer_history.clear()
        self._answer_active_id = self._answer_displayed_id = None
        for pair in self._answer_pair_widgets.values():
            self.answer_paragraphs_layout.removeWidget(pair)
            pair.deleteLater()
        self._answer_pair_widgets.clear()
        self._answer_pairs = []
        with QSignalBlocker(self.answer_history_select):
            self.answer_history_select.clear()
        self.answer_history_bar.hide()
        self._display_answer(question_en="", sources=[])

    def _display_answer(self, en: str = "", zh: str = "", question_zh: str = "",
                        sources: list | None = None, streaming: bool = False,
                        question_en: str | None = None, question_preview_en: str | None = None,
                        question_raw_en: str | None = None,
                        answer_pairs: list[dict] | None = None) -> None:
        from .display_text import clean_answer_text
        from .stable_text import completed_text
        en = clean_answer_text(en, streaming=streaming)
        zh = clean_answer_text(zh, streaming=streaming)
        question_zh = clean_answer_text(question_zh, streaming=streaming)
        visible_pairs = None
        if answer_pairs is not None:
            visible_pairs = []
            for index, pair in enumerate(answer_pairs):
                if not isinstance(pair, dict):
                    continue
                row = dict(pair)
                row["id"] = str(pair.get("id") or index)
                for field in ("en", "zh"):
                    row[field] = completed_text(clean_answer_text(pair.get(field, ""), streaming=streaming),
                                                final=not streaming or bool(pair.get(field + "_complete")))
                if row["en"] or row["zh"]:
                    visible_pairs.append(row)
            en = "\n\n".join(pair["en"] for pair in visible_pairs if pair["en"])
            zh = "\n\n".join(pair["zh"] for pair in visible_pairs if pair["zh"])
        else:
            en = completed_text(en, final=not streaming)
            zh = completed_text(zh, final=not streaming)
        first_text = not self._answer_en and not self._answer_zh
        english_position = self.answer_scroll.verticalScrollBar().value()
        anchor = self._capture_answer_anchor()
        if first_text or (en and not self._answer_en):
            english_position = 0
            anchor = None
        self._answer_en, self._answer_zh = en, zh
        self._answer_visible_pairs = visible_pairs
        if self._overlay:
            self._overlay.set_answer(en, zh, answer_pairs=visible_pairs)
        self._update_answer_question(question_en, question_zh, question_preview_en, question_raw_en,
                                     clear=not (en or zh or question_zh or streaming))
        self._sync_overlay_question()
        self.answer_stack.setCurrentIndex(1 if en or zh or self._answer_question_en or question_zh or streaming else 0)
        self._render_answer(self.answer_english, en, False)
        self._render_answer(self.answer_chinese, zh, True)
        self._render_answer_paragraphs(en, zh, streaming, answer_pairs=visible_pairs)
        self.answer_state.setText("正在生成…" if streaming else "已生成" if en or zh else "等待生成")
        self.copy_button.setEnabled(bool(en))
        if sources is not None:
            self._sources = list(sources)
        self.source_button.setText(f"来源  {len(self._sources)}")
        self.source_button.setEnabled(bool(self._sources))
        self.answer_context_note.setText("建议基于当前问题与检索资料生成，请核对事实与个人经历。" if self._sources
                                         else "未匹配到资料时，回答属于一般性建议；请核对个人经历与事实。")
        self._settle_answer_positions(english_position, anchor=anchor)

    def _style_answer_pair(self, pair: QWidget) -> None:
        focused = self._focus_mode == "answer"
        pair.english.setStyleSheet(f"font-size:{21 if focused else 18}px; color:#E8F0FC; padding:2px 0;")
        pair.chinese.setStyleSheet(f"font-size:{19 if focused else 17}px; color:#ABC3DB; padding:2px 0;")

    def _render_answer_paragraphs(self, en: str, zh: str, streaming: bool,
                                  answer_pairs: list[dict] | None = None) -> None:
        if answer_pairs is not None:
            # Translation identity comes from the provider's validated bilingual
            # unit, never from coincidentally equal paragraph counts.
            rows = [(str(pair["id"]), pair["en"], pair["zh"]) for pair in answer_pairs]
        else:
            english = [part.strip() for part in re.split(r"\n\s*\n", en) if part.strip()]
            rows = [(f"legacy-en-{index}", part, "") for index, part in enumerate(english)]
            if zh:
                # A single legacy block is the whole answer, so it can share a
                # reader. Multiple English blocks have no proven per-block map.
                if len(english) == 1 and not re.search(r"\n\s*\n", zh):
                    rows[0] = (rows[0][0], rows[0][1], zh)
                else:
                    rows.append(("legacy-zh", "", zh))
        if not rows:
            rows = [("empty", "", "")]
        key = self._answer_displayed_id or ("direct", self._answer_question_en)
        active_pairs = []
        seen = {}
        for index, (identity, english_text, chinese_text) in enumerate(rows):
            # Defensive duplicate suffixing preserves all text even if a legacy
            # adapter supplies repeated IDs. Production validates unique IDs.
            occurrence = seen.get(identity, 0)
            seen[identity] = occurrence + 1
            widget_key = (key, identity, occurrence)
            pair = self._answer_pair_widgets.get(widget_key)
            if pair is None:
                pair = QWidget()
                column = _vbox(pair, 0, 8)
                pair.english = _label("", wrap=True)
                pair.chinese = _label("", wrap=True)
                for label in (pair.english, pair.chinese):
                    label.setTextFormat(Qt.TextFormat.PlainText)
                    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
                    label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
                    column.addWidget(label)
                self._style_answer_pair(pair)
                self._answer_pair_widgets[widget_key] = pair
            if self.answer_paragraphs_layout.indexOf(pair) != index:
                self.answer_paragraphs_layout.removeWidget(pair)
                self.answer_paragraphs_layout.insertWidget(index, pair)
            active_pairs.append(pair)
            pair.show()
            if pair.english.text() != english_text:
                pair.english.setText(english_text)
            if pair.chinese.text() != chinese_text:
                pair.chinese.setText(chinese_text)
            pair.english.setVisible(bool(english_text))
            pair.chinese.setVisible(bool(chinese_text) and self.chinese_toggle.isChecked())
        for pair in self._answer_pairs:
            if pair not in active_pairs:
                pair.hide()
                self.answer_paragraphs_layout.removeWidget(pair)
        self._answer_pairs = active_pairs

    def _render_answer(self, view: QLabel, text: str, chinese: bool) -> None:
        # PlainText labels cannot load image resources, open model-written links,
        # or paint beyond their allocated viewport at fractional Windows DPI.
        visible = text or ("等待生成中文对照…" if chinese else "")
        if view.text() != visible:
            view.setText(visible)

    def set_status(self, message: str, level: str = "info") -> None:
        color = {"info": "#6D9993", "success": "#65DCC5", "warning": "#BF934C", "error": "#BF6B63"}.get(level, "#91A7AC")
        self.status_dot.setStyleSheet(f"color:{color};font-size:8px;")
        self.status_label.setText(str(message).replace("\n", "  ·  "))
        self.status_label.setToolTip(str(message))

    def set_listening(self, listening: bool) -> None:
        if self._overlay:
            self._overlay.state.setText("监听中" if listening else "未监听")
        if listening and not self._listening:
            self._started_at = time.monotonic()
        self._listening = bool(listening)
        self.start_button.setEnabled(not listening and bool(self.device_select.currentData()))
        self.stop_button.setEnabled(listening)
        self.device_select.setEnabled(not listening)
        self.live_chip.setText("正在监听" if listening else "未监听")
        self.live_chip.setObjectName("badge" if listening else "neutralBadge")
        self.live_chip.style().unpolish(self.live_chip)
        self.live_chip.style().polish(self.live_chip)
        if not listening:
            self.level_meter.setValue(0)
        self._update_clock()

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        self.answer_button.setEnabled(not busy)
        self.answer_button.setText("正在生成…" if busy else "生成双语回答")
        self.cancel_button.setVisible(busy)
        self.practice_button.setVisible(not busy)

    def _update_clock(self) -> None:
        if self._listening:
            seconds = max(0, int(time.monotonic() - self._started_at))
            self.session_clock.setText(f"监听中   {seconds // 60:02d}:{seconds % 60:02d}")
        else:
            self.session_clock.setText("监听已停止" if self._started_at else "尚未开始监听")

    def set_documents(self, documents: list[dict]) -> None:
        self.library_count.setText(f"{len(documents)} 份资料")
        self.library_stack.setCurrentIndex(1 if documents else 0)
        self.document_table.setRowCount(len(documents))
        for row, raw in enumerate(documents):
            doc = raw if isinstance(raw, dict) else {"name": str(raw)}
            path = str(doc.get("path", ""))
            title = str(doc.get("name") or doc.get("title") or Path(path).name or "未命名文档")
            suffix = str(doc.get("type") or Path(title).suffix.lstrip(".") or "—").upper()
            chunks = doc.get("chunks", doc.get("passages", doc.get("count", "—")))
            status = str(doc.get("status") or "已建立索引")
            for col, value in enumerate([title, suffix, str(chunks), status]):
                item = QTableWidgetItem(value)
                if col == 0:
                    item.setToolTip(path or title)
                    item.setIcon(_icon("source", "#74A79F"))
                if col == 3:
                    item.setForeground(QColor(TEAL if "失败" not in status else "#B66F67"))
                self.document_table.setItem(row, col, item)

    def set_metrics(self, metrics: dict) -> None:
        """Optional keys: audio_level, asr_device, asr_ms, answer_ms, model_status.

        Unknown keys are ignored; unavailable measurements are never fabricated.
        ``audio_level`` is a normalised 0..1 value, not raw PCM amplitude.
        """
        if "audio_level" in metrics:
            self.level_meter.setValue(max(0, min(100, int(float(metrics["audio_level"]) * 100))))
        parts = []
        if metrics.get("asr_device"):
            parts.append(str(metrics["asr_device"]).upper())
        if metrics.get("asr_ms") is not None:
            parts.append(f"识别 {float(metrics['asr_ms']) / 1000:.1f}s")
        if parts:
            self.asr_metric.setText(" · ".join(parts))
        if metrics.get("answer_ms") is not None:
            self.engine_metric.setText(f"首段回答 {float(metrics['answer_ms']) / 1000:.1f}s")
        stages = [("answer_endpoint_ms", "停顿确认→发起"), ("answer_retrieval_ms", "本地检索"),
                  ("answer_queue_ms", "排队"), ("answer_provider_ttft_ms", "模型首段"),
                  ("answer_total_ms", "总耗时")]
        detail = [f"{title}：{float(metrics[key]) / 1000:.2f} 秒" for key, title in stages
                  if metrics.get(key) is not None]
        if detail:
            self.engine_metric.setToolTip("\n".join(detail))
        if metrics.get("model_status"):
            self.model_status.setText(str(metrics["model_status"]))

    def set_models(self, models: list[str]) -> None:
        current = self.ollama_select.currentText()
        blocker = QSignalBlocker(self.ollama_select)
        self.ollama_select.clear()
        for model in models:
            self.ollama_select.addItem(str(model), str(model))
        self.ollama_select.setEditText(current)
        del blocker

    def set_answer_models(self, models: list) -> None:
        """Populate only a controller-provided real catalog; keep empty default."""
        current = self.answer_model_input.text()
        blocker = QSignalBlocker(self.answer_model_input)
        self.answer_model_input.clear()
        seen = set()
        for entry in models:
            name = str((entry.get("model") or entry.get("id") or "") if isinstance(entry, dict) else entry).strip()
            if name and name not in seen:
                self.answer_model_input.addItem(name, name)
                seen.add(name)
        self.answer_model_input.setText(current)
        del blocker

    def _show_sources(self) -> None:
        if self._sources_dialog:
            self._sources_dialog.close()
        dialog = QDialog(self)
        self._allow_dialog_maximize(dialog)
        dialog.setWindowTitle("Prism · 本次检索证据")
        dialog.resize(min(1000, self.width()), min(820, max(500, self.height() - 50)))
        layout = _vbox(dialog, 24, 14)
        layout.addWidget(_label("EVIDENCE LIBRARY", "sectionEyebrow"))
        layout.addWidget(_label("本次回答的检索来源", "pageTitle"))
        layout.addWidget(_label("以下为检索到的片段。请结合模型回答核对引用、页码与结论。", "muted", True))
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        column = _vbox(content, 0, 13)
        for index, raw in enumerate(self._sources, start=1):
            if isinstance(raw, dict):
                source = raw
            elif hasattr(raw, "__dict__"):
                source = vars(raw)
            else:
                source = {"text": str(raw)}
            name = str(source.get("name") or source.get("title") or source.get("source") or source.get("document") or source.get("file") or source.get("path") or f"资料片段 {index}")
            excerpt = str(source.get("excerpt") or source.get("text") or source.get("content") or source.get("snippet") or "未提供原文片段")
            page = source.get("page", source.get("page_number", source.get("location", "")))
            card = _frame()
            box = _vbox(card, 16, 9)
            box.addWidget(_label(f"[{index}]  {Path(name).name}", "cardTitle", True))
            if page:
                box.addWidget(_label(f"位置  ·  {page}", "muted", True))
            text = _label(excerpt, "transcriptZh", True)
            text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            box.addWidget(text)
            column.addWidget(card)
        column.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)
        close = _button("关闭")
        close.clicked.connect(dialog.close)
        layout.addWidget(close, 0, Qt.AlignmentFlag.AlignRight)
        self._sources_dialog = dialog
        dialog.show()


def smoke_main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", metavar="PNG", help="Render a clearly labelled UI demonstration and exit")
    parser.add_argument("--page", choices=["session", "library", "settings", "empty"], default="session")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=850)
    args = parser.parse_args()
    app = QApplication.instance() or QApplication(sys.argv[:1])
    if args.smoke and not QFontDatabase.families() and sys.platform == "win32":
        # Qt's offscreen plugin may not enumerate Windows fonts. Load system
        # fonts for reproducible screenshots; no font files are redistributed.
        for font_file in ("segoeui.ttf", "seguisb.ttf", "msyh.ttc"):
            candidate = Path("C:/Windows/Fonts") / font_file
            if candidate.is_file():
                QFontDatabase.addApplicationFont(str(candidate))
    app.setApplicationName("Prism Interview")
    app.setFont(QFont(FONT, 10))
    window = CopilotWindow()
    if args.smoke:
        window.resize(args.width, args.height)
        window.set_devices([("demo-only", "界面演示 · 未连接真实音频")])
        window.set_settings({"provider": "codex", "cloud_consent": True, "share_evidence": True, "auto_save_sessions": False})
        window.set_answer_model("界面示例 · 自选模型")
        if args.page != "empty":
            window.append_transcript("demo-1", "Tell me about a challenging project and your role.",
                "请介绍一个有挑战的项目，以及你的具体职责。", speaker_id="speaker_1", speaker_label="面试官", turn_id="demo-t1")
            window.append_transcript("demo-2", "How did you measure success?",
                "你如何衡量项目是否成功？", speaker_id="speaker_1", speaker_label="面试官", turn_id="demo-t1",
                raw_en="and um how did you measure success and what would you do differently next time")
            window.set_preparation_status("界面示例 · 2 份虚构资料 · 未连接音频或模型")
            window.set_question("Tell me about a challenging project. What was your contribution, how did you measure success, and what would you improve?")
            window.set_answer("In this example project, I was responsible for improving the reliability of a customer-facing service. I first clarified the main failure cases with the team, then introduced automated checks and a staged release process.\n\nI would support the outcome with the verified metrics from my project notes. Next time, I would involve users earlier so that reliability improvements also address their most important needs.",
                "在这个示例项目中，我负责提升一项面向客户的服务的可靠性。我先与团队明确主要故障场景，再引入自动检查和分阶段发布流程。\n\n我会引用项目记录中经过核实的指标来说明成果。下一次，我会更早邀请用户参与，让可靠性改进同时解决他们最关心的问题。",
                "请介绍一个有挑战的项目，说明你的贡献、衡量成果的方法与改进思路。",
                [{"name": "示例 · Project notes.pdf", "page": 1, "text": "仅用于界面演示的虚构资料。候选人负责可靠性改进、自动检查和分阶段发布。此示例不代表真实检索结果或模型表现。"}],
                question_en="Tell me about a challenging project. What was your contribution, how did you measure success, and what would you improve?")
            window.set_answer(en=window._answer_en, zh=window._answer_zh,
                question_en=window._answer_question_en, question_zh=window._answer_question_zh,
                answer_pairs=[{"id": str(i), "en": en, "zh": zh}
                    for i, (en, zh) in enumerate(zip(window._answer_en.split("\n\n"), window._answer_zh.split("\n\n")))])
            window.set_documents([{"name": "示例 · Candidate CV.pdf", "chunks": 8, "status": "虚构示例"},
                                  {"name": "示例 · Project notes.pdf", "chunks": 12, "status": "虚构示例"}])
        window.set_status("界面演示 · 示例内容不代表真实识别、模型结果或真实面试表现", "warning")
        if args.page == "library":
            window._navigate(1)
        elif args.page == "settings":
            window._navigate(2)
    window.show()
    if args.smoke:
        def capture():
            destination = Path(args.smoke).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not window.grab().save(str(destination)):
                app.exit(2)
                return
            app.quit()
        QTimer.singleShot(500, capture)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(smoke_main())

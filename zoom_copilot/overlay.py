"""Movable desktop captions with stable, source-paired conversation cards.

The window makes no changes to capture, screen sharing or window exclusion.
"""
from PySide6.QtCore import Qt, Signal, QEvent, QTimer, QSignalBlocker, QSize
from PySide6.QtGui import QColor, QAction
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QWidget, QComboBox,
    QPushButton, QPlainTextEdit, QCheckBox, QApplication, QToolButton,
    QMenu, QGraphicsDropShadowEffect, QGraphicsOpacityEffect, QSizeGrip,
    QSplitter, QFrame, QSizePolicy, QScrollArea, QSlider, QWidgetAction,
)
from .caption_display import caption_english, caption_units, confirmed_caption_state


class CaptionPair(QWidget):
    """One source unit and its translation; never infer sentence alignment."""
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        self.english = QLabel()
        self.english.setObjectName('overlayEnglish')
        self.chinese = QLabel()
        self.chinese.setObjectName('overlayChinese')
        for label in (self.english, self.chinese):
            label.setWordWrap(True)
            label.setTextFormat(Qt.TextFormat.PlainText)
            label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
            label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            shadow = QGraphicsDropShadowEffect(label)
            shadow.setBlurRadius(5)
            shadow.setOffset(0, 1)
            shadow.setColor(QColor(0, 0, 0, 255))
            label.setGraphicsEffect(shadow)
            layout.addWidget(label)
        self._full_english = self._full_chinese = ''

    def set_pair(self, english, chinese):
        english, chinese = caption_english(english), str(chinese or '').strip()
        if (english, chinese) == (self._full_english, self._full_chinese):
            return False
        self._full_english, self._full_chinese = english, chinese
        for label, text in ((self.english, self._full_english),
                            (self.chinese, self._full_chinese)):
            if label.text() != text:
                label.setText(text)
        # A pending translation is a small status in the card header. Do not
        # fill the Chinese reading lane with a repeated placeholder or erase
        # a translated prefix when new confirmed English arrives.
        self.chinese.setVisible(bool(self._full_chinese))
        return True

    def minimumSizeHint(self):
        # Match the workspace paragraph: QLabel's wrapped minimum is measured
        # at its narrowest word and can create thousands of pixels of empty
        # margin before the actual English in a wide floating reader.
        hint = super().minimumSizeHint()
        return QSize(hint.width(), 0)


class CaptionCard(QFrame):
    def __init__(self, identity, parent=None):
        super().__init__(parent)
        self.identity = str(identity)
        self.setObjectName('overlayCaptionCard')
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 14)
        layout.setSpacing(12)
        self.header = QLabel()
        self.header.setObjectName('overlaySpeaker')
        self.header.setWordWrap(True)
        layout.addWidget(self.header)
        self.pairs = []
        self._plain = ''
        self._record = None

    def minimumSizeHint(self):
        hint = super().minimumSizeHint()
        return QSize(hint.width(), 0)

    def set_record(self, record):
        units = caption_units(record)
        snapshot = (str(record.get('speaker_label') or ''), str(record.get('state') or ''), tuple(units))
        if snapshot == self._record:
            return False
        self._record = snapshot
        label = ' · '.join(str(record.get(key) or '') for key in ('speaker_label', 'state')
                           if record.get(key))
        if self.header.text() != label:
            self.header.setText(label)
        self.header.setVisible(bool(label))
        while len(self.pairs) < len(units):
            pair = CaptionPair(self)
            self.pairs.append(pair)
            self.layout().addWidget(pair)
        for index, pair in enumerate(self.pairs):
            pair.setVisible(index < len(units))
            if index < len(units):
                pair.set_pair(*units[index])
        body = '\n\n'.join(en + ('\n' + zh if zh else '') for en, zh in units)
        self._plain = (label + '\n' if label else '') + body
        return True


class CaptionConversation(QScrollArea):
    """Update cards in place and preserve the reader's viewport anchor."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName('overlayConversation')
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.setMinimumHeight(65)
        self.host = QWidget()
        self.host.setObjectName('overlayConversationHost')
        self.column = QVBoxLayout(self.host)
        self.column.setContentsMargins(0, 0, 5, 0)
        self.column.setSpacing(10)
        # Empty room belongs only below a short conversation. A fixed trailing
        # reading margin created a scroll range beyond the actual Chinese tail.
        self.column.addStretch(1)
        self.setWidget(self.host)
        self.viewport().setAutoFillBackground(False)
        self.host.setAutoFillBackground(False)
        self._cards = {}
        self._order = []
        self._revision = 0
        self._follow = True
        self._pending_anchor = None
        self._position_timer = QTimer(self)
        self._position_timer.setSingleShot(True)
        self._position_timer.timeout.connect(self._apply_position)
        self._position_finish_timer = QTimer(self)
        self._position_finish_timer.setSingleShot(True)
        self._position_finish_timer.timeout.connect(self._finish_position)
        self._position_finish_revision = 0
        self._anchor_finish_timer = QTimer(self)
        self._anchor_finish_timer.setSingleShot(True)
        self._anchor_finish_timer.timeout.connect(self._finish_restore)
        self._anchor_finish_revision = 0
        self.verticalScrollBar().rangeChanged.connect(self._range_changed)
        self.verticalScrollBar().sliderPressed.connect(self._cancel_restore)
        self.verticalScrollBar().actionTriggered.connect(self._cancel_restore)
        self.viewport().installEventFilter(self)

    def toPlainText(self):
        return '\n\n'.join(self._cards[key]._plain for key in self._order)

    def _anchor(self):
        position = self.verticalScrollBar().value()
        for key in self._order:
            card = self._cards[key]
            if card.y() + card.height() > position:
                return key, card.y() - position, position
        return None, 0, position

    def set_records(self, records, *, follow=True):
        anchor = self._pending_anchor or self._anchor()
        changed = self._follow != bool(follow)
        self._follow = bool(follow)
        wanted = []
        for index, record in enumerate(records):
            key = str(record.get('entry_id') or f'caption-{index}')
            if key in wanted:
                continue
            wanted.append(key)
            card = self._cards.get(key)
            if card is None:
                card = CaptionCard(key, self.host)
                self._cards[key] = card
                changed = True
            changed = card.set_record(record) or changed
            if self.column.indexOf(card) != len(wanted) - 1:
                self.column.insertWidget(len(wanted) - 1, card)
                changed = True
        for key in list(self._cards):
            if key not in wanted:
                card = self._cards.pop(key)
                self.column.removeWidget(card)
                card.deleteLater()
                changed = True
        self._order = wanted
        if not changed:
            return False
        self.column.activate()
        self._revision += 1
        revision = self._revision
        self._pending_anchor = anchor
        # Qt preserves buffered pixels and merges each label's dirty region.
        # Re-enabling an entire disabled viewport invalidated every old card,
        # which flashed the transparent desktop reader on every callback.
        if self._follow:
            self._position_timer.start(0)
        else:
            self._settle(anchor, revision)
            self._position_finish_revision = revision
            self._position_finish_timer.start(0)
        self._anchor_finish_revision = revision
        self._anchor_finish_timer.start(60)
        return True

    def _range_changed(self, *_args):
        if self._pending_anchor is not None and not self._follow:
            self._settle(self._pending_anchor, self._revision)
        elif self._follow and not self._position_timer.isActive():
            self._position_timer.start(0)

    def _apply_position(self):
        self._settle(self._pending_anchor or self._anchor(), self._revision)
        self._position_finish_revision = self._revision
        self._position_finish_timer.start(0)

    def _finish_position(self):
        revision = self._position_finish_revision
        if revision == self._revision:
            self._settle(self._pending_anchor or self._anchor(), revision)

    def _finish_restore(self):
        revision = self._anchor_finish_revision
        if revision == self._revision:
            self._pending_anchor = None

    def _cancel_restore(self, *_args):
        self._revision += 1
        self._pending_anchor = None
        self._position_timer.stop()
        self._position_finish_timer.stop()

    def eventFilter(self, watched, event):
        if watched is self.viewport() and event.type() == QEvent.Type.Resize:
            self._position_timer.start(0)
        if watched is self.viewport() and event.type() == QEvent.Type.Wheel:
            self._cancel_restore()
        return super().eventFilter(watched, event)

    def _settle(self, anchor, revision):
        if revision != self._revision:
            return
        self.column.activate()
        key, offset, position = anchor
        bar = self.verticalScrollBar()
        if self._follow:
            # Match the workspace: older speech above, newest speech below.
            # A late Chinese translation must advance the tail too, including
            # when one complete pair exceeds the available screen height.
            # Full English remains in its original card and can be scrolled.
            bar.setValue(bar.maximum())
        else:
            card = self._cards.get(key)
            bar.setValue(card.y() - offset if card is not None else position)


class SubtitleOverlay(QDialog):
    restore_requested = Signal()
    next_requested = Signal()
    phase_requested = Signal(str)
    stop_requested = Signal()

    def __init__(self):
        super().__init__(None, Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint |
                         Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowTitle("Prism Interview · 桌面双语字幕（拖动标题移动，右下角缩放）")
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.setMinimumSize(560, 280)
        self.resize(1120, 520)
        self._transparent = True
        self._background_transparency = 80
        self._font_size = 23
        self._caption_text = self._caption_zh = self._question_text = ""
        self._caption_units = []
        self._caption_speaker = self._caption_state = ""
        self._conversation = []
        self._answer_en = self._answer_zh = ""
        self._anchor = "top"
        self._hide_timer = QTimer(self)
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._hide_controls)
        self._fit_timer = QTimer(self)
        self._fit_timer.setSingleShot(True)
        self._fit_timer.timeout.connect(self._fit_current_pair)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 6, 10, 6)
        layout.setSpacing(4)
        self.controls = QFrame()
        self.controls.setObjectName("overlayControls")
        row = QHBoxLayout(self.controls)
        row.setContentsMargins(8, 5, 8, 5)
        row.setSpacing(6)
        self.drag_handle = QLabel("拖动 · Prism Interview")
        self.drag_handle.setToolTip("按住拖动，可移至其他显示器")
        self.drag_handle.setCursor(Qt.CursorShape.SizeAllCursor)
        self.drag_handle.installEventFilter(self)
        row.addWidget(self.drag_handle)
        self.state = QLabel("未监听")
        row.addWidget(self.state)
        self.mode = QComboBox()
        self.mode.addItem("双语对话", "captions")
        self.mode.addItem("听译 + 回答", "qa")
        self.mode.addItem("完整回答", "answer")
        self.mode.addItem("完整问题 + 回答", "question")
        self.mode.setMinimumWidth(125)
        self.mode.currentIndexChanged.connect(self._apply_mode)
        row.addWidget(self.mode)
        row.addStretch()
        for text, slot in (("置顶", lambda: self.move_to_top(self.screen())),
                           ("置底", lambda: self.move_to_bottom(self.screen())),
                           ("A−", lambda: self.set_font_size(self._font_size - 2)),
                           ("A+", lambda: self.set_font_size(self._font_size + 2))):
            button = QPushButton(text)
            button.clicked.connect(slot)
            row.addWidget(button)
        more = QToolButton()
        more.setText("更多")
        more.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.menu = QMenu(more)
        self.transparent_action = QAction("透明背景（桌面歌词）", self.menu)
        self.transparent_action.setCheckable(True)
        self.transparent_action.setChecked(True)
        self.transparent_action.toggled.connect(self.set_transparent)
        self.menu.addAction(self.transparent_action)
        background = QWidget()
        background_layout = QVBoxLayout(background)
        background_layout.setContentsMargins(10, 6, 10, 6)
        self.transparency_label = QLabel()
        background_layout.addWidget(self.transparency_label)
        self.transparency_slider = QSlider(Qt.Orientation.Horizontal)
        self.transparency_slider.setRange(0, 100)
        self.transparency_slider.setValue(self._background_transparency)
        self.transparency_slider.setToolTip('只改变背景透明度；英文、中文保持清晰。0%为不透明，100%为完全透明。')
        self.transparency_slider.valueChanged.connect(self.set_background_transparency)
        background_layout.addWidget(self.transparency_slider)
        background_action = QWidgetAction(self.menu)
        background_action.setDefaultWidget(background)
        self.menu.addAction(background_action)
        self.transparency_label.setText(f'背景透明度 {self._background_transparency}%')
        self.follow_action = QAction("跟随最新听译", self.menu)
        self.follow_action.setCheckable(True)
        self.follow_action.setChecked(True)
        self.follow_action.toggled.connect(self._follow_changed)
        self.menu.addAction(self.follow_action)
        self.auto_height_action = QAction('自动增高以显示当前语段', self.menu)
        self.auto_height_action.setCheckable(True)
        self.auto_height_action.setChecked(True)
        self.auto_height_action.toggled.connect(lambda _checked: self._schedule_fit())
        self.menu.addAction(self.auto_height_action)
        self.draft_action = QAction('显示正在识别的英文（会修订）', self.menu)
        self.draft_action.setCheckable(True)
        self.draft_action.setChecked(False)
        self.draft_action.setToolTip('默认显示已确认语段。开启后可查看会逐词修订的临时英文；临时英文不代表最终识别。')
        self.draft_action.toggled.connect(lambda _checked: self._refresh_caption())
        self.menu.addAction(self.draft_action)
        screens = self.menu.addMenu("移动到显示器")
        for index, screen in enumerate(QApplication.screens()):
            action = screens.addAction(f"{index + 1} · {screen.name()}")
            action.triggered.connect(lambda checked=False, target=screen: self._move_to_screen(target))
        self.menu.addSeparator()
        for text, action in (("问答 · 自动辅助", lambda: self.phase_requested.emit("qa")),
                             ("汇报 · 仅听译", lambda: self.phase_requested.emit("presentation")),
                             ("开始下一题", self.next_requested.emit),
                             ("停止监听", self.stop_requested.emit),
                             ("返回主界面", self.restore_requested.emit)):
            self.menu.addAction(text).triggered.connect(action)
        more.setMenu(self.menu)
        row.addWidget(more)
        restore = QPushButton("返回")
        restore.clicked.connect(self.restore_requested.emit)
        row.addWidget(restore)
        self.maximize_button = QPushButton("放大")
        self.maximize_button.clicked.connect(self._toggle_maximize)
        row.addWidget(self.maximize_button)
        self._controls_opacity = QGraphicsOpacityEffect(self.controls)
        self.controls.setGraphicsEffect(self._controls_opacity)
        layout.addWidget(self.controls)
        self.content = QFrame()
        self.content.setObjectName("overlayContent")
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(12, 2, 12, 2)
        content_layout.setSpacing(4)
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(9)
        self.caption_panel = QFrame()
        self.caption_panel.setObjectName('overlayCaptionPanel')
        caption_layout = QVBoxLayout(self.caption_panel)
        caption_layout.setContentsMargins(0, 0, 0, 0)
        caption_layout.setSpacing(7)
        self.caption_title = QLabel('双语对话 · 每段英文下方对应中文')
        self.caption_title.setObjectName('overlaySection')
        caption_layout.addWidget(self.caption_title)
        self.caption = CaptionConversation()
        self.caption.viewport().installEventFilter(self)
        caption_layout.addWidget(self.caption, 1)
        self.answer_panel = QFrame()
        answer_layout = QVBoxLayout(self.answer_panel)
        answer_layout.setContentsMargins(0, 0, 0, 0)
        answer_layout.setSpacing(6)
        answer_title = QLabel('回答建议 · 英文 / 中文')
        answer_title.setObjectName('overlaySection')
        answer_layout.addWidget(answer_title)
        self.answer = self._reader("回答建议将在这里出现；可用滚轮查看完整内容")
        answer_layout.addWidget(self.answer, 1)
        self.splitter.addWidget(self.caption_panel)
        self.splitter.addWidget(self.answer_panel)
        self.splitter.setSizes([260, 180])
        content_layout.addWidget(self.splitter, 1)
        layout.addWidget(self.content, 1)
        self.footer = QWidget()
        self.footer.setObjectName('overlayFooter')
        footer_layout = QHBoxLayout(self.footer)
        footer_layout.setContentsMargins(6, 0, 0, 0)
        self.hint = QLabel("完整英中直接显示 · 自动增高 · 拖动标题到扩展屏 · 更多中可关闭跟随")
        footer_layout.addWidget(self.hint, 1)
        self.previous_page = QPushButton("上一页")
        self.next_page = QPushButton("下一页")
        self.previous_page.clicked.connect(lambda: self._page(-1))
        self.next_page.clicked.connect(lambda: self._page(1))
        footer_layout.addWidget(self.previous_page)
        footer_layout.addWidget(self.next_page)
        self.grip = QSizeGrip(self)
        footer_layout.addWidget(self.grip)
        self._footer_opacity = QGraphicsOpacityEffect(self.footer)
        self.footer.setGraphicsEffect(self._footer_opacity)
        layout.addWidget(self.footer)
        self.show_answer = QCheckBox("显示回答", self)
        self.show_answer.hide()
        self.show_answer.setChecked(False)
        self.show_answer.toggled.connect(lambda checked: self.mode.setCurrentIndex(1 if checked else 0))
        self._apply_appearance()
        self.mode.setCurrentIndex(0)
        self._apply_mode()

    def _reader(self, placeholder):
        reader = QPlainTextEdit()
        reader.setReadOnly(True)
        reader.setFrameShape(QFrame.Shape.NoFrame)
        reader.setPlaceholderText(placeholder)
        reader.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        reader.setMinimumHeight(65)
        reader.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        reader.viewport().setMouseTracking(True)
        reader.viewport().installEventFilter(self)
        shadow = QGraphicsDropShadowEffect(reader)
        shadow.setBlurRadius(7)
        shadow.setOffset(0, 2)
        shadow.setColor(QColor(0, 0, 0, 240))
        reader.setGraphicsEffect(shadow)
        return reader

    def _apply_appearance(self):
        alpha = round(255 * (100 - self._background_transparency) / 100) if self._transparent else 255
        panel = "transparent" if self._transparent else "#101C2E"
        card = f'rgba(15,32,42,{alpha})'
        border_alpha = min(alpha, 65)
        self.setStyleSheet(f"""
            QDialog {{ background: transparent; }}
            QFrame#overlayContent {{ background:{panel}; border-radius:12px; }}
            QFrame#overlayControls {{ background:rgba(15,32,42,235); border-radius:9px; }}
            QFrame#overlayCaptionPanel, QScrollArea#overlayConversation,
            QWidget#overlayConversationHost {{ background:transparent; border:0; }}
            QFrame#overlayCaptionCard {{ background:{card}; border:1px solid rgba(132,192,187,{border_alpha});
                border-left:3px solid rgba(105,201,182,{border_alpha}); border-radius:10px; }}
            QLabel#overlayEnglish {{ font-size:{self._font_size}px; color:#E8F0FC; font-weight:500; }}
            QLabel#overlayChinese {{ font-size:{max(16,self._font_size-2)}px; color:#B0E5DA; font-weight:400; }}
            QLabel#overlaySpeaker {{ color:#65DCC5; font-size:13px; font-weight:600; }}
            QLabel#overlaySection {{ color:#CDDEDA; font-size:12px; padding:5px 9px;
                background:{card}; border-radius:6px; }}
            QWidget#overlayFooter {{ background:rgba(15,32,42,235); border-radius:6px; }}
            QToolButton#overlayExpand {{ font-size:12px; padding:3px 8px; }}
            QLabel {{ color:#F4FCFA; font-family:'Microsoft YaHei UI'; font-size:12px; background:transparent; }}
            QPlainTextEdit {{ color:#FFFFFF; background:transparent; border:0; font-family:'Microsoft YaHei UI';
                font-size:{self._font_size}px; font-weight:600; selection-background-color:#246D76; padding:2px; }}
            QPlainTextEdit:focus {{ border:0; }}
            QPushButton, QToolButton, QComboBox {{ background:rgba(22,39,60,235); color:#ECF9F5;
                border:1px solid #405775; border-radius:5px; padding:5px 7px; font-size:12px; }}
            QPushButton:hover, QToolButton:hover {{ background:#28544F; }}
            QMenu, QComboBox QAbstractItemView {{ background:#142337; color:white; padding:7px; }}
            QMenu::item:selected {{ background:#28544F; }}
            QScrollBar:vertical {{ background:transparent; width:8px; }}
            QScrollBar::handle:vertical {{ background:rgba(215,240,233,150); border-radius:4px; min-height:20px; }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background:transparent; }}
            QSplitter::handle {{ background:rgba(235,250,244,35); }}
        """)

    def set_transparent(self, enabled):
        self._transparent = bool(enabled)
        self.transparency_slider.setEnabled(self._transparent)
        self._apply_appearance()

    def set_background_transparency(self, value):
        self._background_transparency = max(0, min(100, int(value)))
        with QSignalBlocker(self.transparency_slider):
            self.transparency_slider.setValue(self._background_transparency)
        self.transparency_label.setText(f'背景透明度 {self._background_transparency}%')
        self._apply_appearance()

    def set_font_size(self, value):
        self._font_size = max(16, min(44, int(value)))
        self._apply_appearance()
        self._schedule_fit()

    def _schedule_fit(self):
        if hasattr(self, 'auto_height_action') and self.auto_height_action.isChecked():
            # Allow nested scroll/card layouts to settle before measuring; their
            # initial 100px widths can otherwise force a full-screen empty box.
            self._fit_timer.start(35)

    def _fit_current_pair(self):
        if (not self.auto_height_action.isChecked() or not self.follow_action.isChecked()
                or self.isMaximized() or not self.caption_panel.isVisible()):
            return
        card = self.caption._cards.get(self.caption._order[-1]) if self.caption._order else None
        pair = next((p for p in reversed(card.pairs) if not p.isHidden()), None) if card else None
        screen = self.screen() or QApplication.primaryScreen()
        if pair is None or screen is None:
            return
        self.layout().activate()
        margins = card.layout().contentsMargins()
        width = max(120, self.caption.viewport().width() - margins.left() - margins.right() - 7)
        needed = sum(max(label.heightForWidth(width), label.minimumHeight())
                     for label in (pair.english, pair.chinese) if not label.isHidden()) + pair.layout().spacing()
        chrome = self.height() - self.caption.viewport().height()
        target = min(screen.availableGeometry().height() - 24,
                     chrome + needed + card.header.sizeHint().height() + 40)
        # Grow only: late translations cannot repeatedly shrink/reflow a reader.
        if target > self.height() + 18:
            target = max(target, min(screen.availableGeometry().height() - 24, self.height() + 96))
            self.resize(self.width(), target)
            box = screen.availableGeometry()
            self.move(max(box.x(), min(self.x(), box.right() - self.width() + 1)),
                      max(box.y() + 12, min(self.y(), box.bottom() - self.height() - 12)))
            self.caption._position_timer.start(0)

    def _apply_mode(self, *args):
        if not hasattr(self, "caption"):
            return
        mode = self.mode.currentData()
        self.caption_panel.setVisible(mode != "answer")
        self.answer_panel.setVisible(mode != "captions")
        self._refresh_caption()

    @staticmethod
    def _replace(view, text, follow=False):
        if view.toPlainText() == text:
            return
        bar = view.verticalScrollBar()
        position = bar.value()
        view.setPlainText(text)
        bar.setValue(bar.maximum() if follow else position)

    def _refresh_caption(self):
        if self.mode.currentData() == "question" and self._question_text:
            english = "问题 · " + self._question_text
            chinese = self._caption_zh if self._caption_text.strip() == self._question_text.strip() else ""
            records = [{'entry_id':'current-question','en':english,'zh':chinese,
                        'speaker_label':'完整问题'}]
            self.caption_title.setText('完整问题 · 可滚动阅读全文')
        elif self._conversation:
            records = list(self._conversation)
            self.caption_title.setText(f'双语对话 · 最近 {len(records)} 段 · 英文下方对应中文')
            # Provisional speech is a separate, stable-identity card. It must
            # remain visible while Whisper confirms the next source unit;
            # never replace or mutate any confirmed historical card.
            live = self._live_record()
            if live is not None:
                records.append(live)
        else:
            english, chinese = self._caption_text, self._caption_zh
            visible = self.draft_action.isChecked() or confirmed_caption_state(self._caption_state)
            records = [{'entry_id':'current-caption','en':english,'zh':chinese,
                        'speaker_label':self._caption_speaker,'state':self._caption_state,
                        'units':self._caption_units}] if visible and (english or chinese) else []
            self.caption_title.setText('双语对话 · 每段英文下方对应中文' if records or not english else
                                       '正在识别 · 确认后显示完整语段')
        if self.caption.set_records(records, follow=self.follow_action.isChecked()):
            self._schedule_fit()

    def _live_record(self):
        stable = confirmed_caption_state(self._caption_state)
        if not (self.draft_action.isChecked() or stable) or not self._caption_text.strip():
            return None
        normalized = lambda text: ' '.join(str(text).casefold().split()).strip(' .!?')
        live = normalized(self._caption_text)
        # Confirmation can precede/follow the ledger notification. Retire the
        # temporary card only once the same source is represented by history.
        if any(live == normalized(row.get('en', '')) or
               any(live == normalized(en) for en, _zh in row.get('units', []))
               for row in self._conversation):
            return None
        if '等待语音' in self._caption_state:
            return None
        return {'entry_id': '__prism_live_caption__', 'en': self._caption_text,
                'zh': self._caption_zh, 'speaker_label': self._caption_speaker or '当前发言',
                'state': self._caption_state or '正在识别 · 会修订',
                'units': self._caption_units}

    def set_conversation(self, entries):
        """Accept stable committed cards from the main conversation's ledger."""
        unique = {}
        for record in entries or []:
            key = str(record.get('entry_id') or '')
            if key and (str(record.get('en') or '').strip() or str(record.get('zh') or '').strip()):
                unique[key] = dict(record, entry_id=key)
        self._conversation = list(unique.values())[-8:]
        self._refresh_caption()

    def set_caption(self, english, chinese="", *, speaker_label="", state="已成句", units=None):
        self._caption_text, self._caption_zh = str(english), str(chinese)
        self._caption_units = list(units or [])
        self._caption_speaker, self._caption_state = str(speaker_label), str(state)
        self._refresh_caption()

    def _toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
            self.maximize_button.setText("放大")
        else:
            self.showMaximized()
            self.maximize_button.setText("恢复")

    def _follow_changed(self, enabled):
        if hasattr(self, 'caption'):
            self.caption._cancel_restore()
            self.caption._follow = bool(enabled)
        if enabled and hasattr(self, "caption"):
            self.caption._position_timer.start(0)
            self._schedule_fit()

    def set_question(self, question):
        self._question_text = str(question)
        if self.mode.currentData() == "question":
            self._refresh_caption()

    def set_answer(self, english, chinese, *, answer_pairs=None):
        from .display_text import clean_answer_text
        self._answer_en, self._answer_zh = clean_answer_text(english), clean_answer_text(chinese)
        if answer_pairs is not None:
            paragraphs = []
            for pair in answer_pairs:
                en, zh = clean_answer_text(pair.get('en', '')), clean_answer_text(pair.get('zh', ''))
                if en or zh:
                    paragraphs.append(en + ('\n' + zh if zh else ''))
            display = '\n\n'.join(paragraphs)
        else:
            # Legacy output has no source-pair identity. Keep it complete;
            # equal paragraph counts alone do not establish correspondence.
            display = self._answer_en + ("\n\n" + self._answer_zh if self._answer_zh else "")
        self._replace(self.answer, display)

    def _page(self, direction):
        target = self.answer if self.answer.isVisible() else self.caption
        bar = target.verticalScrollBar()
        bar.setValue(bar.value() + direction * max(1, bar.pageStep() - 1))

    def _move_to_screen(self, screen):
        (self.move_to_bottom if self._anchor == "bottom" else self.move_to_top)(screen)

    def _place(self, screen, bottom=False):
        screen = screen or self.screen() or QApplication.primaryScreen()
        if screen:
            box = screen.availableGeometry()
            self.resize(min(self.width(), box.width()), min(self.height(), box.height()))
            self.move(box.x() + (box.width() - self.width()) // 2,
                      box.bottom() - self.height() - 12 if bottom else box.y() + 12)

    def move_to_top(self, screen=None):
        self._anchor = "top"
        self._place(screen)

    def move_to_bottom(self, screen=None):
        self._anchor = "bottom"
        self._place(screen, True)

    def _show_controls(self):
        self._controls_opacity.setOpacity(1)
        self._footer_opacity.setOpacity(1)
        self.controls.setEnabled(True)
        self.footer.setEnabled(True)
        self._hide_timer.start(2500)

    def _hide_controls(self):
        if self.menu.isVisible() or self.controls.underMouse() or self.footer.underMouse():
            self._hide_timer.start(1000)
            return
        self._controls_opacity.setOpacity(0)
        self._footer_opacity.setOpacity(0)
        self.controls.setEnabled(False)
        self.footer.setEnabled(False)

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.Type.Enter, QEvent.Type.MouseMove):
            self._show_controls()
        if watched is self.drag_handle and event.type() == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.LeftButton and self.windowHandle():
                self.windowHandle().startSystemMove()
                return True
        return super().eventFilter(watched, event)

    def enterEvent(self, event):
        self._show_controls()
        super().enterEvent(event)

    def mouseMoveEvent(self, event):
        self._show_controls()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._hide_timer.start(900)
        super().leaveEvent(event)

    def showEvent(self, event):
        self._show_controls()
        super().showEvent(event)
        self._schedule_fit()

    def closeEvent(self, event):
        event.ignore()
        self.restore_requested.emit()

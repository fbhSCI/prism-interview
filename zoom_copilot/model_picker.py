"""A dedicated, resizable answer-model picker with read-only discovery."""
from __future__ import annotations

from dataclasses import replace
import queue
import threading
import time

from PySide6.QtCore import Qt, QSignalBlocker, QTimer
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
    QFrame, QLabel, QLineEdit, QListWidget, QListWidgetItem, QPlainTextEdit, QPushButton, QScrollArea,
    QSplitter, QVBoxLayout, QWidget,
)

from .config import Preferences
from . import credential_store
from .model_catalog import (TESTED_FAST_GPT, TESTED_FAST_NOTE, choose_default_gpt,
                            is_text_answer_model, model_id, supports_fast)
from .providers import QWEN_REGION_ENDPOINTS


class ModelPickerDialog(QDialog):
    """`exec()` then `selected_preferences()` returns a copy; no inference here.

    `catalog_loader`, when supplied, accepts a Preferences copy on a background
    thread and returns official model dictionaries. This also supports isolated UI
    tests without network/authentication or subscription use.
    """
    def __init__(self, preferences: Preferences, parent=None, catalog_loader=None):
        super().__init__(parent)
        self.setWindowTitle("选择回答模型 · Prism Interview")
        from .ui import STYLE, _brand_icon
        self.setStyleSheet(STYLE)
        self.setWindowIcon(_brand_icon())
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowMinMaxButtonsHint)
        self.setSizeGripEnabled(True)
        self.resize(960, 760)
        self.setMinimumSize(680, 580)
        screen = self.screen()
        if screen:
            available = screen.availableGeometry()
            self.resize(min(960, max(680, available.width() - 60)),
                        min(760, max(580, available.height() - 60)))
        self._preferences = replace(preferences, session_api_keys=dict(preferences.session_api_keys))
        self._loader = catalog_loader
        self._states: dict[str, dict] = {}
        self._provider = ""
        self._key_drafts: dict[str, str] = {}
        self._key_input_scope: str | None = None
        self._key_from_default = False
        self._key_default_available = False
        self._catalog: list[dict] = []
        self._generation = 0
        self._closed = threading.Event()
        self.destroyed.connect(lambda *_args, closed=self._closed: closed.set())
        self._test_generation = 0
        self._test_cancel = threading.Event()
        self._test_parts = {"answer_en": "", "answer_zh": ""}
        # Workers own only ordinary Python data, never a QWidget/QObject or a
        # closure containing the dialog. Releasing a completed Thread target on
        # its worker thread must not destroy a Qt widget there. All Qt updates
        # and widget lifetimes stay on the GUI thread through this queue/timer.
        self._events = queue.SimpleQueue()
        self._event_timer = QTimer(self)
        self._event_timer.setInterval(20)
        self._event_timer.timeout.connect(self._drain_events)
        self._event_timer.start()
        self.finished.connect(self._close_tasks)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(22, 20, 22, 18)
        title = QLabel("回答模型")
        title.setStyleSheet("font-size:22px;font-weight:650;")
        outer.addWidget(title)
        intro = QLabel("从当前账号的实际目录选择。切换模型不会发送提问；只有生成回答时才会使用相应额度。")
        intro.setWordWrap(True)
        outer.addWidget(intro)
        # API settings plus a test result can exceed a 720 px laptop viewport.
        # Scroll only this body; model test/accept/cancel actions remain visible.
        self.body_scroll = QScrollArea()
        self.body_scroll.setWidgetResizable(True)
        self.body_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.body_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.body_scroll.setMinimumSize(0, 0)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 4, 0)
        self.body_scroll.setWidget(body)
        outer.addWidget(self.body_scroll, 1)
        form = QFormLayout()
        self.provider_select = QComboBox()
        for name, provider in (("GPT · 当前 Codex 订阅", "codex"), ("DeepSeek · 独立 API", "deepseek"),
                               ("千问 Qwen · 阿里云 Model Studio", "qwen"), ("本地 Ollama", "ollama"),
                               ("自定义兼容 API", "compatible")):
            self.provider_select.addItem(name, provider)
        form.addRow("提供方", self.provider_select)

        self.api_fields = QWidget()
        api_form = QFormLayout(self.api_fields)
        api_form.setContentsMargins(0, 0, 0, 0)
        self.region_select = QComboBox()
        self.region_select.addItem("中国内地 · 北京", "cn-beijing")
        self.region_select.addItem("国际 · 新加坡", "ap-southeast-1")
        self.region_select.addItem("其他地域 / 业务空间专属地址（手动填写）", "workspace")
        self.region_label = QLabel("千问地域")
        api_form.addRow(self.region_label, self.region_select)
        self.endpoint_input = QLineEdit()
        self.endpoint_input.setPlaceholderText("官方 API Base URL，不含 /chat/completions")
        api_form.addRow("API 地址", self.endpoint_input)
        self.key_input = QLineEdit()
        self.key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_input.setPlaceholderText("填写此服务和地址的密钥；默认密钥会自动以密码形式填入")
        api_form.addRow("API Key", self.key_input)
        self.key_status_label = QLabel()
        self.key_status_label.setObjectName("apiKeyStatus")
        self.key_status_label.setWordWrap(True)
        api_form.addRow(self.key_status_label)
        key_actions = QHBoxLayout()
        self.remember_key_button = QPushButton("记住本机默认密钥")
        self.forget_key_button = QPushButton("删除本机默认密钥")
        self.remember_key_button.setToolTip("保存到当前 Windows 用户的凭据管理器，按提供方和 API 地址隔离；不进入迁移包。")
        self.forget_key_button.setToolTip("仅删除当前提供方、当前 API 地址的本机默认密钥。")
        key_actions.addWidget(self.remember_key_button)
        key_actions.addWidget(self.forget_key_button)
        api_form.addRow(key_actions)
        self.remember_key_button.clicked.connect(self._store_default_key)
        self.forget_key_button.clicked.connect(self._delete_default_key)
        self.free_checkbox = QCheckBox("已确认此提供方有免费额度，且已在控制台禁止超额计费")
        api_form.addRow(self.free_checkbox)
        api_note = QLabel("程序无法验证免费余额或冻结计费。回答需确认免费额度，或已有千问付费授权；保存密钥不会改变授权。密钥需与地域匹配。")
        api_note.setWordWrap(True)
        api_form.addRow(api_note)
        layout.addLayout(form)
        layout.addWidget(self.api_fields)

        controls = QHBoxLayout()
        self.catalog_title = QLabel("当前订阅可用 GPT 文本模型")
        self.catalog_title.setStyleSheet("font-weight:650;")
        controls.addWidget(self.catalog_title, 1)
        self.refresh_button = QPushButton("刷新官方目录")
        controls.addWidget(self.refresh_button)
        layout.addLayout(controls)
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("筛选模型名称，例如 luna、sol、flash、qwen")
        layout.addWidget(self.search_input)
        self.model_list = QListWidget()
        self.model_list.setObjectName("answerModelList")
        # Style both row states explicitly so native Windows palette changes
        # cannot make selectable model names disappear on an unfocused list.
        self.model_list.setStyleSheet("""
            QListWidget#answerModelList {
                background: #111D2E; color: #DDE7F4;
                border: 1px solid #30445E; border-radius: 8px;
                padding: 4px; outline: 0;
                selection-background-color: #214B4A;
                selection-color: #D9FFF4;
            }
            QListWidget#answerModelList::item {
                padding: 10px 12px; border: 1px solid transparent;
                border-radius: 6px;
            }
            QListWidget#answerModelList::item:hover:!selected {
                background: #1D3349; color: #DDE7F4;
            }
            QListWidget#answerModelList::item:selected,
            QListWidget#answerModelList::item:selected:!active {
                background: #214B4A; color: #D9FFF4;
                border: 1px solid #4F9B8C;
            }
        """)
        self.model_list.setToolTip("单击模型行即可选择，也可用上下方向键切换。")
        self.model_list.setMinimumHeight(100)
        self.model_list.setWordWrap(True)
        self.model_list.setSpacing(3)
        self.result_splitter = QSplitter(Qt.Orientation.Vertical)
        self.result_splitter.addWidget(self.model_list)
        self.test_panel = QWidget()
        test_layout = QVBoxLayout(self.test_panel)
        test_layout.setContentsMargins(0, 5, 0, 0)
        self.test_metrics = QLabel("测试结果")
        self.test_metrics.setWordWrap(True)
        test_layout.addWidget(self.test_metrics)
        self.test_output = QPlainTextEdit()
        self.test_output.setReadOnly(True)
        self.test_output.setMinimumHeight(80)
        test_layout.addWidget(self.test_output, 1)
        self.result_splitter.addWidget(self.test_panel)
        self.result_splitter.setChildrenCollapsible(False)
        self.test_panel.hide()
        layout.addWidget(self.result_splitter, 1)
        selected = QFormLayout()
        self.model_input = QLineEdit()
        self.model_input.setPlaceholderText("Codex 留空：优先选择本机已测较快的可用 GPT")
        selected.addRow("选中模型", self.model_input)
        self.speed_select = QComboBox()
        self.speed_select.addItem("Fast · 较快消耗现有订阅额度", "fast")
        self.speed_select.addItem("标准", "standard")
        selected.addRow("Codex 速度", self.speed_select)
        self.fast_note = QLabel("自动优先 GPT-5.6 Luna（目录存在时），否则选择可用 Luna / 账号默认。" + TESTED_FAST_NOTE + "。")
        self.fast_note.setWordWrap(True)
        selected.addRow(self.fast_note)
        self.cloud_checkbox = QCheckBox("允许将问题、近期问答和所选候选人背景发送到选中的云服务")
        self.cloud_checkbox.setChecked(preferences.cloud_consent)
        selected.addRow(self.cloud_checkbox)
        self.evidence_checkbox = QCheckBox("发送检索命中的少量资料片段")
        self.evidence_checkbox.setChecked(preferences.share_evidence)
        selected.addRow(self.evidence_checkbox)
        layout.addLayout(selected)
        self.status_label = QLabel("尚未读取目录。目录可见不等于有免费额度或必然可调用。")
        self.status_label.setWordWrap(True)
        self.status_label.setMinimumHeight(36)
        outer.addWidget(self.status_label)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("使用此模型")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.test_button = self.buttons.addButton("测试短回答", QDialogButtonBox.ButtonRole.ActionRole)
        self.test_button.setToolTip("发送固定的通用面试常识题，使用所选服务额度；不读取或发送个人文档、资料库和历史问答。")
        self.cancel_test_button = self.buttons.addButton("停止测试", QDialogButtonBox.ButtonRole.ActionRole)
        self.cancel_test_button.setEnabled(False)
        outer.addWidget(self.buttons)

        self.provider_select.currentIndexChanged.connect(self._provider_changed)
        self.region_select.currentIndexChanged.connect(self._region_changed)
        self.endpoint_input.textChanged.connect(self._endpoint_key_changed)
        self.key_input.textChanged.connect(self._key_edited)
        self.refresh_button.clicked.connect(self.refresh_catalog)
        self.search_input.textChanged.connect(self._filter_models)
        self.model_list.currentItemChanged.connect(self._selection_changed)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.test_button.clicked.connect(self.start_answer_test)
        self.cancel_test_button.clicked.connect(self.cancel_answer_test)
        initial = preferences.provider.lower()
        if initial == "openai_compatible":
            initial = "compatible"
        keys = self._preferences.session_api_keys
        if preferences.api_key and initial not in keys:
            keys[initial] = preferences.api_key
        self._states[initial] = {"model": preferences.ollama_model if initial == "ollama" else preferences.answer_model,
                                 "endpoint": preferences.api_base_url, "free": preferences.free_quota_confirmed,
                                 "speed": preferences.answer_speed, "region": preferences.qwen_region}
        initial_endpoint = (preferences.api_base_url.strip() or
            ("https://api.deepseek.com/v1" if initial == "deepseek" else
             QWEN_REGION_ENDPOINTS.get(preferences.qwen_region, "") if initial == "qwen" else ""))
        initial_scope = self._credential_scope(initial, initial_endpoint)
        if initial_scope and keys.get(initial):
            self._key_drafts[initial_scope] = keys[initial]
        index = self.provider_select.findData(initial)
        self.provider_select.setCurrentIndex(max(0, index))
        if not self._provider:
            self._provider_changed()
        # A read-only subscription catalog may be loaded automatically. API
        # catalogs wait for an explicit click after provider/key/region selection.
        if initial == "codex" and preferences.cloud_consent:
            QTimer.singleShot(0, self.refresh_catalog)

    def _remember_current(self):
        if not self._provider:
            return
        self._states[self._provider] = {"model": self.model_input.text().strip(),
                                       "endpoint": self.endpoint_input.text().strip(),
                                       "free": self.free_checkbox.isChecked(),
                                       "speed": self.speed_select.currentData(),
                                       "region": self.region_select.currentData()}
        if self._provider in ("deepseek", "qwen", "compatible"):
            self._remember_key_draft()
            key = self.key_input.text().strip()
            if key and not self._key_from_default:
                self._preferences.session_api_keys[self._provider] = key
            else:
                self._preferences.session_api_keys.pop(self._provider, None)

    def _provider_changed(self, *_):
        self.cancel_answer_test()
        self._remember_current()
        self._provider = str(self.provider_select.currentData())
        self._generation += 1
        self._catalog = []
        self.model_list.clear()
        self.search_input.clear()
        state = self._states.get(self._provider, {})
        api = self._provider in ("deepseek", "qwen", "compatible")
        self.api_fields.setVisible(api)
        self.region_select.setVisible(self._provider == "qwen")
        self.region_label.setVisible(self._provider == "qwen")
        self.speed_select.setEnabled(self._provider == "codex")
        self.fast_note.setVisible(self._provider == "codex")
        self.cloud_checkbox.setEnabled(self._provider != "ollama")
        self.evidence_checkbox.setEnabled(self._provider != "ollama")
        self.model_input.setText(state.get("model", ""))
        self.model_input.setPlaceholderText("留空：优先选择本机已测较快的可用 GPT" if self._provider == "codex" else
                                            "请从目录选择，或填写该地域控制台中的确切文本模型标识")
        self.free_checkbox.setChecked(bool(state.get("free", False)))
        self.speed_select.setCurrentIndex(max(0, self.speed_select.findData(state.get("speed", "fast"))))
        self.region_select.blockSignals(True)
        self.region_select.setCurrentIndex(max(0, self.region_select.findData(state.get("region", "cn-beijing"))))
        self.region_select.blockSignals(False)
        default_endpoint = ("https://api.deepseek.com/v1" if self._provider == "deepseek" else
                            QWEN_REGION_ENDPOINTS.get(self.region_select.currentData(), "") if self._provider == "qwen" else "")
        with QSignalBlocker(self.endpoint_input):
            self.endpoint_input.setText(state.get("endpoint", "") or default_endpoint)
        self._refresh_key_input()
        self.catalog_title.setText({"codex": "当前订阅可用 GPT 文本模型", "deepseek": "DeepSeek 官方模型目录",
                                    "qwen": "所选地域的千问文本模型", "ollama": "本机已安装模型", "compatible": "兼容服务的模型目录"}[self._provider])
        self.status_label.setText("点击刷新读取实际目录。千问若未提供 /models，可从对应地域控制台复制确切文本模型标识。")
        self.refresh_button.setEnabled(True)

    @staticmethod
    def _credential_scope(provider: str, endpoint: str) -> str | None:
        if provider not in ("deepseek", "qwen", "compatible"):
            return None
        try:
            return credential_store.credential_target(provider, endpoint)
        except ValueError:
            return None

    def _remember_key_draft(self):
        if self._key_input_scope is None:
            return
        key = self.key_input.text().strip()
        if key and not self._key_from_default:
            self._key_drafts[self._key_input_scope] = key
        else:
            self._key_drafts.pop(self._key_input_scope, None)

    def _show_key(self, key: str, *, saved: bool, available: bool):
        self._key_from_default = saved
        self._key_default_available = available
        with QSignalBlocker(self.key_input):
            self.key_input.setText(key)
        self._update_key_status()

    def _update_key_status(self):
        present = bool(self.key_input.text().strip())
        if self._key_input_scope is None:
            message = "请先填写有效的 HTTPS API 地址，再读取或保存本机默认密钥。"
        elif self._key_from_default and present:
            message = "已自动填入此服务和地址的本机默认密钥（密码隐藏）；更换电脑后需重新设置。"
        elif present:
            message = "当前为本次会话输入；点击“记住本机默认密钥”后，下次会自动填入。"
        elif self._key_default_available:
            message = "此服务和地址已保存默认密钥；留空调用时仍会使用它。"
        else:
            message = "此服务和地址尚未保存本机默认密钥。填写后可选择记住。"
        self.key_status_label.setText(message)
        self.key_status_label.setStyleSheet("color:#65DCC5;" if self._key_from_default and present else "color:#9BB2CC;")
        self.remember_key_button.setEnabled(self._key_input_scope is not None and present)
        self.forget_key_button.setEnabled(self._key_input_scope is not None and self._key_default_available)

    def _key_edited(self, *_):
        self._key_from_default = False
        self._remember_key_draft()
        self._update_key_status()

    def _endpoint_key_changed(self, *_):
        # Clear the old address's secret immediately. Only this app's exact
        # provider/HTTPS target may populate the replacement password field.
        scope = self._credential_scope(self._provider, self.endpoint_input.text())
        if scope != self._key_input_scope:
            self._remember_key_draft()
            self._refresh_key_input()

    def _refresh_key_input(self):
        endpoint = self.endpoint_input.text().strip()
        scope = self._credential_scope(self._provider, endpoint)
        self._key_input_scope = scope
        self._preferences.session_api_keys.pop(self._provider, None)
        if scope is None:
            self._show_key("", saved=False, available=False)
            return
        try:
            default = credential_store.load_key(self._provider, endpoint)
        except (OSError, ValueError):
            self._show_key(self._key_drafts.get(scope, ""), saved=False, available=False)
            self.key_status_label.setText("未能读取此服务和地址的本机默认密钥；可重新填写并保存。")
            return
        draft = self._key_drafts.get(scope, "")
        self._show_key(draft or default, saved=bool(default and not draft), available=bool(default))

    def _region_changed(self, *_):
        if self._provider != "qwen":
            return
        self.cancel_answer_test()
        self.endpoint_input.setText(QWEN_REGION_ENDPOINTS.get(self.region_select.currentData(), ""))
        self._generation += 1
        self._catalog = []
        self.model_list.clear()
        self.model_input.clear()
        self.free_checkbox.setChecked(False)
        self.status_label.setText("地域已更改。请核对该地域的密钥、免费额度和模型目录；其他地域需要业务空间专属地址。")
        self.refresh_button.setEnabled(True)

    def _store_default_key(self):
        key = self.key_input.text().strip()
        try:
            credential_store.store_key(self._provider, self.endpoint_input.text(), key)
        except (OSError, ValueError):
            self.status_label.setText("未能保存默认密钥。请核对 API 地址与密钥，并确认正在 Windows 上运行。")
            return
        self._key_drafts.pop(self._key_input_scope, None)
        self._preferences.session_api_keys.pop(self._provider, None)
        self._show_key(key, saved=True, available=True)
        self.status_label.setText("已保存为本机默认密钥，并以密码形式显示。下次打开会自动填入；迁移电脑后需重新设置。")

    def _delete_default_key(self):
        try:
            credential_store.delete_key(self._provider, self.endpoint_input.text())
        except (OSError, ValueError):
            self.status_label.setText("未能删除默认密钥，请核对当前 API 地址。")
            return
        self._key_drafts.pop(self._key_input_scope, None)
        self._preferences.session_api_keys.pop(self._provider, None)
        self._show_key("", saved=False, available=False)
        self.status_label.setText("已删除此提供方和地址的本机默认密钥。若设置了环境变量密钥，它仍按原配置生效。")

    def selected_preferences(self) -> Preferences:
        self._remember_current()
        prefs = replace(self._preferences, session_api_keys=dict(self._preferences.session_api_keys))
        prefs.provider = self._provider
        prefs.cloud_consent = self.cloud_checkbox.isChecked()
        prefs.share_evidence = self.evidence_checkbox.isChecked()
        prefs.answer_speed = str(self.speed_select.currentData())
        prefs.api_base_url = self.endpoint_input.text().strip() if self._provider not in ("codex", "ollama") else ""
        prefs.qwen_region = str(self.region_select.currentData())
        prefs.api_key = prefs.session_api_keys.get(self._provider, "")
        prefs.free_quota_confirmed = self.free_checkbox.isChecked() if self._provider in ("deepseek", "qwen", "compatible") else False
        if self._provider == "ollama":
            prefs.ollama_model = self.model_input.text().strip() or self._preferences.ollama_model
        else:
            prefs.answer_model = self.model_input.text().strip()
        return prefs

    def refresh_catalog(self):
        prefs = self.selected_preferences()
        if prefs.provider != "ollama" and not prefs.cloud_consent:
            self.status_label.setText("请先允许连接所选云服务，或选择本地 Ollama；本次未读取目录。")
            return
        self._generation += 1
        generation = self._generation
        self.refresh_button.setEnabled(False)
        self.status_label.setText("正在读取实际模型目录……不会发送论文或生成测试回答。")
        loader, closed, result_queue = self._loader, self._closed, self._events
        def work():
            service = None
            try:
                if loader:
                    catalog = loader(prefs)
                else:
                    from .answers import AnswerService
                    service = AnswerService(prefs)
                    catalog = service.available_models()
                error = ""
            except Exception as exc:
                catalog, error = [], str(exc)
            finally:
                if service:
                    service.close()
            if not closed.is_set():
                result_queue.put(("catalog", generation, catalog, error))
        threading.Thread(target=work, daemon=True, name="answer-model-catalog").start()

    def _drain_events(self):
        if self._closed.is_set():
            return
        for _ in range(256):
            try:
                kind, generation, value, extra = self._events.get_nowait()
            except queue.Empty:
                break
            if kind == "catalog":
                self._catalog_ready(generation, value, extra)
            elif kind == "test_delta":
                self._test_delta(generation, value, extra)
            elif kind == "test_done":
                self._test_done(generation, value, extra)

    def _catalog_ready(self, generation: int, catalog, error: str):
        if generation != self._generation:
            return
        self.refresh_button.setEnabled(True)
        if error:
            self.status_label.setText(error)
            return
        self._catalog = [item for item in catalog if isinstance(item, dict) and is_text_answer_model(item, self._provider)]
        selected = self.model_input.text().strip()
        if not selected and self._provider == "codex":
            selected = choose_default_gpt(self._catalog)
        self.model_list.blockSignals(True)
        self.model_list.clear()
        selected_item = None
        for metadata in self._catalog:
            identity = model_id(metadata)
            name = str(metadata.get("displayName") or identity)
            tags = []
            if supports_fast(metadata):
                tags.append("支持 Fast")
            if self._provider == "codex" and identity == TESTED_FAST_GPT:
                tags.append("本机单题已测较快")
            if metadata.get("isDefault"):
                tags.append("账号默认")
            subtitle = str(metadata.get("description") or "")
            text = name + (" · " + " · ".join(tags) if tags else "")
            if name != identity:
                text += "\n" + identity
            if subtitle:
                text += "\n" + subtitle
            row = QListWidgetItem(text)
            row.setData(Qt.ItemDataRole.UserRole, metadata)
            self.model_list.addItem(row)
            if identity == selected:
                selected_item = row
        if selected_item:
            self.model_list.setCurrentItem(selected_item)
        self.model_list.blockSignals(False)
        if selected_item:
            self._selection_changed(selected_item, None)
        self._filter_models()
        count = len(self._catalog)
        self.status_label.setText(f"已读取 {count} 个文本回答模型。模型可见不保证有免费额度或可调用；实际回答仍执行额度检查。" if count else
                                  "目录没有返回文本回答模型。请核对账号、地域，或从服务商控制台填写确切模型标识。")

    def _selection_changed(self, current, _previous):
        if current is None:
            return
        metadata = current.data(Qt.ItemDataRole.UserRole) or {}
        self.model_input.setText(model_id(metadata))
        if self._provider == "codex" and not supports_fast(metadata):
            self.speed_select.setCurrentIndex(self.speed_select.findData("standard"))
            self.status_label.setText("该模型目录未标明支持 Fast，已选标准速度。")

    def _filter_models(self, *_):
        query = self.search_input.text().strip().casefold()
        for index in range(self.model_list.count()):
            row = self.model_list.item(index)
            row.setHidden(bool(query and query not in row.text().casefold()))

    def _close_tasks(self, *_):
        self._closed.set()
        self._test_cancel.set()
        self._event_timer.stop()

    def closeEvent(self, event):
        self._close_tasks()
        super().closeEvent(event)

    def cancel_answer_test(self):
        self._test_cancel.set()
        self._test_generation += 1
        if hasattr(self, "test_button"):
            self.test_button.setEnabled(True)
            self.cancel_test_button.setEnabled(False)
        if hasattr(self, "test_metrics") and self.test_panel.isVisible():
            self.test_metrics.setText("测试已停止；已发送的服务请求可能仍计入提供方额度。")

    def start_answer_test(self):
        """Only this explicit click authorizes the fixed, corpus-free model call."""
        prefs = self.selected_preferences()
        if prefs.provider != "ollama" and not prefs.cloud_consent:
            self.status_label.setText("测试前请允许连接所选云服务。本次未发送问题。")
            return
        if prefs.provider in ("deepseek", "qwen", "compatible"):
            try:
                from .providers import provider_connection
                provider_connection(prefs)  # Validate key, endpoint and no-overage confirmation before a worker.
                if not prefs.answer_model:
                    raise ValueError("请先选择或填写确切模型标识。本次未发送问题。")
            except ValueError as exc:
                self.status_label.setText(str(exc))
                return
        self.cancel_answer_test()
        self._test_cancel = threading.Event()
        cancel = self._test_cancel
        generation = self._test_generation
        self._test_parts = {"answer_en": "", "answer_zh": ""}
        self.test_output.clear()
        self.test_panel.show()
        self.result_splitter.setSizes([180, 180])
        QTimer.singleShot(0, self._reveal_test_result)
        self.test_metrics.setText("正在测试：What makes a clear interview answer? Give one sentence and one practical tip.")
        self.status_label.setText("已发送固定常识题；不读取或发送个人文档、资料库、候选人背景和会话历史。")
        self.test_button.setEnabled(False)
        self.cancel_test_button.setEnabled(True)
        # Replace the profile as well as the evidence/history: no user-specific
        # context may accidentally leak through this connectivity/speed test.
        prefs = replace(prefs, research_profile="Generic interview connectivity test. No candidate-specific facts are provided.",
                        candidate_profile="", prefer_prepared_answers=False,
                        share_evidence=False, top_k=0, brief_answer=True, max_answer_words=45,
                        session_api_keys=dict(prefs.session_api_keys))
        question = "What makes a clear interview answer? Give one sentence and one practical tip."
        closed, result_queue = self._closed, self._events
        def work():
            from .answers import AnswerService
            service = AnswerService(prefs)
            started = time.perf_counter()
            first = {}
            def delta(field, value):
                if value and field in ("answer_en", "answer_zh"):
                    first.setdefault(field, time.perf_counter() - started)
                    if not closed.is_set() and not cancel.is_set():
                        result_queue.put(("test_delta", generation, field, value))
            try:
                result = service.answer(question, [], on_delta=delta, cancel_event=cancel, history=[])
                payload = {"answer": result, "first": first, "elapsed": time.perf_counter() - started}
                error = ""
            except Exception as exc:
                payload, error = None, str(exc)
            finally:
                service.close()
            if not closed.is_set() and not cancel.is_set():
                result_queue.put(("test_done", generation, payload, error))
        threading.Thread(target=work, daemon=True, name="user-requested-answer-test").start()

    def _reveal_test_result(self):
        if not self._closed.is_set():
            self.body_scroll.ensureWidgetVisible(self.test_panel)

    def _test_delta(self, generation: int, field: str, value: str):
        if generation != self._test_generation or self._test_cancel.is_set() or field not in self._test_parts:
            return
        from .display_text import clean_answer_text
        self._test_parts[field] += value
        en = clean_answer_text(self._test_parts["answer_en"], streaming=True)
        zh = clean_answer_text(self._test_parts["answer_zh"], streaming=True)
        self.test_output.setPlainText(en + ("\n\n" + zh if zh else ""))

    def _test_done(self, generation: int, payload, error: str):
        if generation != self._test_generation or self._test_cancel.is_set():
            return
        self.test_button.setEnabled(True)
        self.cancel_test_button.setEnabled(False)
        if error:
            self.test_metrics.setText("测试失败；请核对所选服务、模型及额度。")
            self.status_label.setText(error)
            return
        from .display_text import clean_answer_text
        result = payload["answer"]
        self.test_output.setPlainText(clean_answer_text(result.answer_en) + "\n\n" + clean_answer_text(result.answer_zh))
        first = payload["first"].get("answer_en")
        first_label = f"首段英文 {first:.2f} 秒 · " if first is not None else ""
        confirmed = result.provider == "codex" or result.diagnostics.get("reported_model_confirmed", False)
        model_label = "返回模型" if confirmed else "所请求模型（服务端未报告标识）"
        self.test_metrics.setText(f"{model_label} {result.provider} / {result.model} · {first_label}完成 {payload['elapsed']:.2f} 秒")
        self.status_label.setText("固定问题测试完成；此次速度仅代表当前测试，不是正式面试的延迟保证。")

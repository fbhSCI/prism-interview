"""Qt controller: capture, subtitles, question turns and cancellable answers."""
from __future__ import annotations
from collections import deque
from dataclasses import asdict, fields, replace
from datetime import datetime
import json
import os
import argparse
import shutil
from pathlib import Path
import queue
import re
import sys
import threading
import time
import uuid

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QApplication, QFileDialog, QMessageBox, QDialog,
    QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPlainTextEdit,
    QPushButton, QSplitter)

from .answers import AnswerService
from .codex_client import AnswerCancelled
from .audio import AudioCapture, DualAudioCapture, input_devices, speaker_loopbacks
from .config import Preferences, app_dir
from .knowledge import KnowledgeBase
from .runtime import default_model_cache
from .speech import SpeechWorker
from .translation import ArgosTranslator, TranslationWorker
from .turns import (QuestionAssembler, CaptionTurn, acknowledgment_only,
    announces_question, announced_question_offset, next_addressed_question_offset,
    has_complete_direct_question, confirmed_question_boundary, join_fragments,
    is_question_invitation, invitation_prefix_offset, invitation_segments,
    question_after_moderation_offset, question_introduction_prefix,
    speculative_question_snapshot, is_non_request_context)
from .ui import CopilotWindow, FONT
from .interview_context import (InterviewIntent, InterviewPreparation,
    is_interview_request as looks_like_question, starts_new_question)
from .transcript_text import TranscriptRestorer, caption_tail_incomplete, sentence_continues
from .transcript_refine import LocalGrammarWorker
from .question_packet import QuestionContext
from .research_facts import ResearchFactStore
from .session_store import SessionStore, build_snapshot, new_session_id, render_text, utc_now


def load_runtime():
    base = Path(sys.executable).parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parents[1]
    file = base / "runtime.json"
    if file.is_file():
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            for key, env in (("data_dir", "PRISM_INTERVIEW_HOME"),
                             ("model_dir", "PRISM_INTERVIEW_MODELS"),
                             ("cuda_dir", "PRISM_INTERVIEW_CUDA_DIR")):
                if data.get(key) and env not in os.environ:
                    value = Path(data[key])
                    os.environ[env] = str(value if value.is_absolute() else base / value)
        except (OSError, ValueError, TypeError):
            pass


class CopilotController:
    def __init__(self, window: CopilotWindow, preferences: Preferences | None = None):
        self.window = window
        self.pref = preferences or Preferences.load()
        if not self.pref.model_cache_dir:
            self.pref.model_cache_dir = str(default_model_cache())
        self.events = queue.Queue()
        self.grammar = LocalGrammarWorker(lambda result:self.post('grammar',result),enabled=self.pref.grammar_refinement)
        self._grammar_versions = {}
        self._grammar_serial = 0
        self.kb = KnowledgeBase(app_dir() / "knowledge.json")
        self.service = AnswerService(replace(self.pref))
        self.service.research_facts = ResearchFactStore(app_dir() / 'research_facts.json', self.kb)
        self.restorer = TranscriptRestorer(Path(self.pref.model_cache_dir))
        self.preparation = InterviewPreparation()
        self.intent = InterviewIntent()
        self.question_context = QuestionContext()
        self._question_packets = {}
        self._answer_request_timing = {}
        self.last_answer_timing = {}
        self.last_retrieval_coverage = {}
        self._context_serial = 0
        self._last_question_display = ''
        self.audio = self.speech = self.translation = None
        self.listening = self.preparing = self.busy = False
        self.stopping = False
        self._session = 0
        self._phase_epoch = 0
        self._capture_started = False
        self._capture_error_handled = False
        self._device_refresh_generation = 0
        self._caption_active_source = "loopback"
        self._caption_source_states = {}
        self._translation_ready = False
        self._restoration_ready = False
        self._listen_pref = replace(self.pref)
        self._closed = False
        self._model_check_running = False
        self._active = ""
        self._active_prefs = None
        self._cancel = threading.Event()
        self._answer_started = 0.0
        self._first_answer = False
        self._stream = {}
        self._stream_pairs = []
        self._requests = deque()
        self._history = deque(maxlen=3)
        self._transcripts = []
        self._answers = []
        self.session_store = SessionStore(app_dir() / "sessions")
        self._saved_session_id = new_session_id()
        self._session_created_at = utc_now()
        self._session_dirty = False
        self._partial_id = None
        self._partial_text = ""
        self._partial_clock = None
        self._partial_text_end_sample = 0
        self.assembler = QuestionAssembler(silence_seconds=self.pref.auto_endpoint_seconds)
        self.caption_turn = CaptionTurn()
        self._caption_sources = {}
        self._caption_translation_sources = {}
        self._caption_metadata = {}
        self._caption_acoustic_turn = None
        self._caption_source_clock = None
        self._caption_source_end = None
        self._caption_last_final_at = None
        self._caption_raw = {}
        self._manual_turns = {}
        self._speaker_aliases = {}
        self._live_token = 0
        self._live_data = {}
        self._last_final_end = {}
        self._last_prefetch = 0.0
        self._question_original = ''
        self._question_raw_by_request = {}
        self._answering_voice_seen = False
        self._semantic_response = False
        self._question_intro_tail = ''
        self._awaiting_actual_question = False
        self._last_question_speaker = None
        self._automatic_ids = set()
        self._sealed_automatic_ids = set()
        self._last_automatic_snapshot = None
        self._question_uncertain = False
        self._recognition_gap_active = False
        self._answer_prefetch = None
        self._answer_prefetch_candidate = None
        self._answer_prefetch_generation = None
        self._answer_prefetch_attempts = 0
        self._answer_prefetch_last_key = None
        self._answer_prefetch_library_epoch = 0
        self.window.set_settings(asdict(self.pref))
        self.window.set_answer_model(self.pref.answer_model or (self.pref.ollama_model if self.pref.provider == "ollama" else "连接后确认"))
        window.start_requested.connect(self.start_listening)
        window.stop_requested.connect(self.stop_listening)
        window.answer_requested.connect(self.request_answer)
        window.cancel_requested.connect(self.cancel_answer)
        window.import_requested.connect(self.import_documents)
        window.clear_requested.connect(self.clear_documents)
        window.refresh_devices_requested.connect(self.refresh_devices)
        window.input_source_changed.connect(self.change_input_source)
        window.check_models_requested.connect(self.check_models)
        window.install_translation_requested.connect(self.prepare_translation)
        if hasattr(window, "prepare_speech_requested"):
            window.prepare_speech_requested.connect(self.prepare_speech_models)
        if hasattr(window, "install_speaker_requested"):
            window.install_speaker_requested.connect(self.prepare_speaker)
        window.export_requested.connect(self.export_text)
        if hasattr(window, "save_session_requested"):
            window.save_session_requested.connect(self.save_conversation)
        if hasattr(window, "history_requested"):
            window.history_requested.connect(self.show_session_history)
        if hasattr(window, "auto_save_changed"):
            window.auto_save_changed.connect(self.set_auto_save_sessions)
        window.settings_saved.connect(self.save_settings)
        window.interview_phase_changed.connect(self.change_interview_phase)
        window.new_question_requested.connect(self.new_question)
        if hasattr(window, "model_picker_requested"):
            window.model_picker_requested.connect(self.choose_answer_model)
        if hasattr(window, "speaker_assigned"):
            window.speaker_assigned.connect(self.assign_speaker)
        if hasattr(window, "speaker_role_changed"):
            window.speaker_role_changed.connect(self.change_speaker_role)
        self.timer = QTimer(window)
        self.timer.timeout.connect(self.pump)
        self.timer.start(60)
        self.auto_timer = QTimer(window)
        self.auto_timer.timeout.connect(self.auto_answer)
        self.auto_timer.start(250)
        self.session_save_timer = QTimer(window)
        self.session_save_timer.timeout.connect(self._auto_save_conversation)
        self.session_save_timer.start(3000)
        self.refresh_docs()
        QTimer.singleShot(0, self.refresh_devices)
        self.window.set_status("准备就绪 · 先检查模型与收音设备，再开始会话。")

    def post(self, kind, payload=None):
        self.events.put((kind, payload))

    def background(self, operation, name="PrismWorker"):
        def run():
            try:
                operation()
            except Exception as exc:
                self.post("error", str(exc))
        thread = threading.Thread(target=run, name=name, daemon=True)
        thread.start()
        return thread

    def save_settings(self, data):
        old_prefetch_policy = self._answer_prefetch_policy()
        old_cloud, old_share = self.pref.cloud_consent, self.pref.share_evidence
        old_provider = self.pref.provider
        old_grammar = self.pref.grammar_refinement
        old_caption_consent = getattr(self.pref, "translation_cloud_consent", False)
        provider = data.get("provider", old_provider)
        data = dict(data)
        # Keys belong to one provider and one application session. Selecting a
        # different provider must never send a previous provider's secret there.
        keys = dict(data.get("session_api_keys", getattr(self.pref, "session_api_keys", {})))
        key_provider = "compatible" if provider == "openai_compatible" else provider
        if "api_key" in data:
            key = str(data["api_key"] or "").strip()
            if provider != old_provider and key == self.pref.api_key:
                key = keys.get(key_provider, "")
            if provider not in ("codex", "ollama"):
                if key:
                    keys[key_provider] = key
                else:
                    keys.pop(key_provider, None)
            data["api_key"] = key if provider not in ("codex", "ollama") else ""
        data["session_api_keys"] = keys
        if "auto_save_sessions" in data:
            data["auto_save_sessions"] = data["auto_save_sessions"] is True
        allowed = {f.name for f in fields(Preferences)}
        for key, value in data.items():
            if key in allowed:
                setattr(self.pref, key, value)
        # Controller updates bypass the dataclass constructor. Apply the same
        # strict consent validation before testing revocation, billing or queues.
        self.pref.__post_init__()
        try:
            self.assembler.silence_seconds = max(.6, min(2.5, float(self.pref.auto_endpoint_seconds)))
        except (ValueError, TypeError):
            self.assembler.silence_seconds = 1.0
        privacy_tightened = ((old_cloud and not self.pref.cloud_consent)
                             or (old_share and not self.pref.share_evidence))
        caption_permission_revoked = (old_caption_consent and not self.pref.translation_cloud_consent)
        caption_billing_disabled = not (self.pref.qwen_paid_usage_authorized or self.pref.free_quota_confirmed)
        if ((caption_permission_revoked or caption_billing_disabled)
                and self.translation and self._listen_pref.translation_engine == "qwen"):
            self.translation.stop()
            self.translation = None
            self.window.set_status("已停止千问听译并取消待发送语段；英文识别继续，重新启用后需停止并重新监听。", "warning")
        if privacy_tightened:
            # Queued questions cannot retain permissions the user has revoked.
            self._requests = deque((identity, question, replace(
                prefs, cloud_consent=bool(prefs.cloud_consent and self.pref.cloud_consent),
                share_evidence=bool(prefs.share_evidence and self.pref.share_evidence)))
                for identity, question, prefs in self._requests)
            if self.busy and self._active_prefs and self._active_prefs.provider != "ollama":
                self._cancel.set()
                self.window.set_status("云端权限已收紧，正在停止当前云回答；排队问题会按新权限处理。")
        self.pref.save()  # Preferences excludes the in-memory API key.
        if self._answer_prefetch_policy() != old_prefetch_policy:
            self._cancel_answer_prefetch()
        if old_grammar!=self.pref.grammar_refinement:
            self.grammar.set_enabled(self.pref.grammar_refinement)

    def choose_answer_model(self):
        from .model_picker import ModelPickerDialog
        self.save_settings(self.window.get_settings())
        dialog = ModelPickerDialog(replace(self.pref), self.window)
        try:
            if dialog.exec() != dialog.DialogCode.Accepted:
                return
            prefs = dialog.selected_preferences()
        finally:
            dialog.deleteLater()
        settings = asdict(prefs)
        self.window.set_settings(settings)
        self.save_settings(settings)
        name = prefs.ollama_model if prefs.provider == "ollama" else prefs.answer_model or "自动选择已测较快模型"
        self.window.set_answer_model(name)
        self.window.set_status("新提问使用：" + prefs.provider + " / " + name
                               + ("；正在生成或已排队的回答保留其原模型。" if self.busy else "。"))
        if prefs.provider=='codex' and prefs.cloud_consent and not self.busy:
            self.check_models()

    def change_interview_phase(self, phase):
        if phase not in ("presentation", "qa"):
            return
        self._cancel_answer_prefetch()
        self._phase_epoch += 1
        if self.audio:
            self.audio.set_context(self._phase_epoch)
        self.assembler.reset()
        self.intent.begin_question()
        self.question_context.begin_question()
        self._last_question_speaker = None
        self._question_original = ''
        self._answering_voice_seen = False
        self._semantic_response = False
        self._awaiting_actual_question = False
        self._live_token += 1
        self._live_data = {}
        self.window.set_live_translation('')
        self._question_uncertain = False
        self._seal_all_caption_sources()
        self._partial_id = None
        self._partial_text = ""
        self.pref.interview_phase = phase
        if phase == "qa":
            self.pref.auto_answer = True
        else:
            # Changing back to presentation also stops pending automatic work.
            self._requests = deque(item for item in self._requests if item[0] not in self._automatic_ids)
            if self._active in self._automatic_ids and self.busy:
                self._cancel.set()
        self.window.set_settings(asdict(self.pref))
        self.pref.save()
        self._show_question_packet('')
        self.window.set_status("问答中：问句结束后自动检索并给出双语建议。" if phase == "qa"
                               else "汇报中：持续听译，不自动生成回答。")

    def new_question(self):
        """Explicit boundary when the interviewer changes question rapidly."""
        self._cancel_answer_prefetch()
        self._phase_epoch += 1
        if self.audio:
            self.audio.set_context(self._phase_epoch)
        self.assembler.reset()
        self._invalidate_changed_automatic_question('')
        self.intent.begin_question()
        self.question_context.begin_question()
        self._last_question_speaker = None
        self._question_original = ''
        self._answering_voice_seen = False
        self._semantic_response = False
        self._awaiting_actual_question = False
        self._live_token += 1
        self._live_data = {}
        self._seal_all_caption_sources()
        self._partial_id = None
        self._question_uncertain = False
        self.window.set_question("")
        self._show_question_packet('')
        self.window.set_live_caption("")
        self.window.set_status("已开始下一题。短暂停顿会保留并补充同一个问题。")

    def change_input_source(self, source):
        if source not in ("loopback", "microphone", "both"):
            return
        active = self.listening or self.preparing or self.stopping
        if active:
            # Finish the old streams first. Never combine old device audio with
            # the new source or automatically resume recording after a switch.
            self.stop_listening(drain=True)
        self.save_settings(self.window.get_settings())
        self.refresh_devices()
        self.window.set_status("输入来源已更改。正在保留原收音尾句；结束后选择新设备并点击开始监听。" if active else
                               "请选择新来源的设备，再点击开始监听；麦克风中的我只听译，不自动作答。")

    def refresh_devices(self):
        self._device_refresh_generation += 1
        generation = self._device_refresh_generation
        selected = self.pref.input_source
        sources = ("loopback", "microphone") if selected == "both" else (selected,)
        def work():
            for source in sources:
                self.post("devices", {"generation": generation, "input_source": source,
                                      "devices": input_devices(source)})
        self.background(work, "PrismDevices")

    def refresh_docs(self):
        self._answer_prefetch_library_epoch += 1
        self._cancel_answer_prefetch()
        records = []
        for name in self.kb.document_names:
            record = self.kb.document_records.get(name, {})
            count = sum(p.source == name for p in self.kb.passages)
            records.append({"name": Path(name).name, "source": name,
                            "passages": count, "chunks": count,
                            "size": record.get("size", 0), "status": "已建立本地索引"})
        self.window.set_documents(records)
        # Index changes invalidate related cards immediately; the answer worker
        # rechecks again before every request, including external file changes.
        for warning in self.service.research_facts.refresh():
            self.post('status_global', warning)
        if not self.kb.passages:
            self.window.set_preparation_status(self.preparation.prepare(self.kb))
            return
        def prepare():
            self.post("preparation", self.preparation.prepare(self.kb))
        self.background(prepare, "PrismInterviewPreparation")

    def import_documents(self, filenames):
        if not filenames:
            return
        self._cancel_answer_prefetch()
        self.window.set_status(f"正在本地解析 {len(filenames)} 份资料…")
        def work():
            count, issues = self.kb.add_files(list(filenames))
            self.post("documents", (count, issues))
        self.background(work, "PrismImport")

    def clear_documents(self):
        self._cancel_answer_prefetch()
        def work():
            self.kb.clear()
            self.post("documents", (0, []))
        self.background(work, "PrismClear")

    def prepare_speech_models(self):
        if self.listening or self.preparing or self.stopping:
            self.window.set_status("请先停止监听，再下载或检查语音模型。", "warning")
            return
        values = self.window.get_settings()
        for key in ("model_cache_dir", "whisper_size", "streaming_captions"):
            if key in values:
                setattr(self.pref, key, values[key])
        self.pref.__post_init__()
        self.pref.save()
        from .model_setup import show_model_setup
        show_model_setup(self.window, self.pref)

    def start_listening(self):
        if getattr(getattr(self.window, "_model_setup_dialog", None), "active", False):
            self.window.set_status("语音模型仍在准备，请完成后再开始监听。", "warning")
            return
        if self.listening or self.preparing or self.stopping:
            return
        self.save_settings(self.window.get_settings())
        selected_id = self.pref.microphone_id if self.pref.input_source == "microphone" else self.pref.speaker_id
        if (not selected_id or not self.window.device_select.currentData()
                or (self.pref.input_source == "both" and (not self.pref.microphone_id or not self.window.microphone_select.currentData()))):
            self.window.set_status("请选择当前来源的有效设备；双方同时识别需要会议输出和麦克风各一个。", "warning")
            return
        self._session += 1
        self._phase_epoch += 1
        session = self._session
        self._listen_pref = replace(self.pref)
        self.preparing = True
        if self.pref.provider=='codex' and self.pref.cloud_consent and not self.busy:
            self.check_models()
        self._capture_started = False
        self._capture_error_handled = False
        self._caption_source_states.clear()
        self._caption_active_source = "loopback"
        self.caption_turn = CaptionTurn()
        self.assembler.reset()
        self.intent.reset()
        self.question_context.reset()
        self._last_question_speaker = None
        self._question_original = ''
        self._answering_voice_seen = False
        self._semantic_response = False
        self._awaiting_actual_question = False
        self._manual_turns.clear()
        self._speaker_aliases.clear()
        self._last_final_end.clear()
        self._live_token += 1
        self._live_data = {}
        self._question_uncertain = False
        self.caption_turn.seal()
        self._partial_id = None
        self._partial_text = ""
        if self.speech:
            self.speech.stop()
        if self.translation:
            self.translation.stop()
        self.window.set_status("准备本地语音模型；就绪之前不会开始采集。")
        self.window.start_button.setEnabled(False)
        cache = Path(self.pref.model_cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        model_path = cache / "asr" / self.pref.whisper_size
        if not (model_path/"model.bin").exists() and shutil.disk_usage(cache).free < 4*1024**3:
            self.preparing = False
            self.window.start_button.setEnabled(True)
            self.window.set_status("模型目录不足4GB空闲空间，请在设置中选择其他磁盘的模型目录。", "error")
            return
        model = str(model_path) if (model_path/"model.bin").exists() else self.pref.whisper_size
        translation_engine = self.pref.translation_engine
        try:
            translator = self._make_translator(cache, translation_engine)
        except Exception as exc:
            self.preparing = False
            self.window.start_button.setEnabled(True)
            self.window.set_status("听译未启动：" + str(exc), "error")
            return
        if hasattr(self.window, "set_translation_engine_status"):
            self.window.set_translation_engine_status(translation_engine,
                getattr(translator, "model", ""), "当前收音会话")
        self._translation_ready = translator.ready and translation_engine != "gemma"
        self.translation = TranslationWorker(
            translator,
            lambda identity, zh: self.post("translation", (session, identity, zh)),
            lambda message: self.post("status", (session, message)),
            lambda metrics: self.post("metrics", metrics),
            on_preview=lambda identity, source, zh: self.post("translation_preview", (session, identity, source, zh)),
            preview_interval=.4, allow_prefix_preview=True)
        self.translation.start()
        self._restoration_ready = False
        self.restorer = TranscriptRestorer(cache)
        def prepare_text():
            self.restorer.prepare()
            self.post("restoration_ready", session)
        self.background(prepare_text, "PrismPrepareText")
        if not self._translation_ready:
            def prepare_mt():
                try:
                    if translation_engine == "nllb":
                        translator.install(on_status=lambda text: self.post("status", (session, text)))
                    else:
                        translator.install_english_chinese(lambda text: self.post("status", (session, text)))
                    translator.translate("Could you explain your method?")
                    self.post("translation_ready", session)
                except Exception as exc:
                    self.post("status", (session, "翻译模型准备失败："+str(exc)))
            self.background(prepare_mt, "PrismPrepareMT")
        self.speech = SpeechWorker(
            model, on_text=lambda kind, en, context: self.post("transcript", (session, kind, en, context)),
            on_status=lambda message: self.post("status", (session, message)),
            device=self.pref.asr_device, compute_type=self.pref.compute_type,
            download_root=cache / "asr-hub",
            on_ready=lambda: self.post("asr_ready", session),
            on_metrics=lambda metrics: self.post("metrics", metrics),
            diarizer=self._make_diarizer(cache),
            streaming_model_root=cache if self.pref.streaming_captions else None)
        self.speech.start()

    def _make_diarizer(self, cache):
        if self._microphone_source() or self._single_remote_source() or not self.pref.speaker_diarization:
            return None
        from .speaker import OnlineSpeakerDiarizer
        return OnlineSpeakerDiarizer(cache)

    def _capture_preferences(self):
        return self._listen_pref if (self.listening or self.preparing or self.stopping) else self.pref

    def _microphone_source(self):
        return self._capture_preferences().input_source == "microphone"

    def _single_remote_source(self):
        """Single source is an explicit selection, never a voice identity claim."""
        prefs = self._capture_preferences()
        return prefs.input_source != "microphone" and prefs.caption_speaker_mode == "single_remote"

    _CAPTION_SOURCE_FIELDS = ("caption_turn", "_caption_acoustic_turn", "_caption_source_clock",
        "_caption_source_end", "_caption_last_final_at", "_partial_id", "_partial_text",
        "_partial_clock", "_partial_text_end_sample")

    def _store_caption_source(self):
        self._caption_source_states[self._caption_active_source] = {
            field: getattr(self, field) for field in self._CAPTION_SOURCE_FIELDS}

    def _activate_caption_source(self, source):
        if source == self._caption_active_source:
            return
        self._store_caption_source()
        state = self._caption_source_states.setdefault(source, {
            "caption_turn": CaptionTurn(), "_caption_acoustic_turn": None,
            "_caption_source_clock": None, "_caption_source_end": None,
            "_caption_last_final_at": None, "_partial_id": None, "_partial_text": "",
            "_partial_clock": None, "_partial_text_end_sample": 0})
        for field, value in state.items():
            setattr(self, field, value)
        self._caption_active_source = source

    def _seal_all_caption_sources(self):
        self.caption_turn.seal()
        self._partial_id, self._partial_text = None, ""
        self._partial_clock, self._partial_text_end_sample = None, 0
        for state in self._caption_source_states.values():
            state["caption_turn"].seal()
            state["_partial_id"], state["_partial_text"] = None, ""
            state["_partial_clock"], state["_partial_text_end_sample"] = None, 0

    def _seal_question_caption(self):
        if self._caption_active_source == "loopback":
            self.caption_turn.seal()
        elif "loopback" in self._caption_source_states:
            self._caption_source_states["loopback"]["caption_turn"].seal()

    def _remote_partial_id(self):
        if self._caption_active_source == "loopback":
            return self._partial_id
        return self._caption_source_states.get("loopback", {}).get("_partial_id")

    def _caption_inactivity(self, context, now):
        # Source clocks are authoritative when available: slow ASR processing
        # must not turn ordinary speech into an apparent 12-second pause.
        if isinstance(context, dict) and context.get("stream_id") is not None:
            clock = (context["stream_id"], context.get("phase_epoch"))
            start = context.get("start_sample")
            end = context.get("end_sample")
            gap = 0.0
            if (clock == self._caption_source_clock and isinstance(start, (int, float))
                    and isinstance(self._caption_source_end, (int, float))):
                gap = max(0.0, (start - self._caption_source_end) / 16000)
            self._caption_source_clock = clock
            if isinstance(end, (int, float)):
                self._caption_source_end = end
            self._caption_last_final_at = now
            return gap
        gap = 0.0 if self._caption_last_final_at is None else now - self._caption_last_final_at
        self._caption_last_final_at = now
        return gap

    def _capture_confirmation_options(self):
        """Use the measured profile only on its tested decoder/source path.

        A requested CUDA device is insufficient: auto mode may have fallen back
        to CPU. Custom long windows and acoustic multi-speaker mode retain the
        original cadence. These values never modify Whisper's decoding gates.
        """
        prefs = self._listen_pref
        try:
            standard_window = abs(float(prefs.caption_window_seconds) - 3.6) < .001
        except (TypeError, ValueError):
            standard_window = False
        if (getattr(prefs, 'caption_update_mode', 'original') == 'responsive'
                and prefs.caption_speaker_mode == 'single_remote'
                and prefs.input_source == 'loopback'
                and prefs.whisper_size == 'large-v3-turbo' and standard_window
                and prefs.compute_type == 'int8_float16'
                and getattr(self.speech, 'actual_device', '') == 'cuda'):
            return {'commit_interval_sec': 1.6, 'initial_maximum_sec': 3.0}
        return {}

    def _start_capture(self, session):
        if session != self._session or not self.preparing or self._capture_started:
            return
        if not self._translation_ready or not self._restoration_ready or not self.speech or not self.speech.ready_event.is_set():
            return
        self._capture_started = True
        worker = self.speech
        source = self._listen_pref.input_source
        capture_type = DualAudioCapture if source == "both" else AudioCapture
        selected_id = self._listen_pref.microphone_id if source == "microphone" else self._listen_pref.speaker_id
        arguments = (selected_id, self._listen_pref.microphone_id) if source == "both" else (selected_id,)
        options = {} if source == "both" else {"input_source": source}
        self.audio = capture_type(
            *arguments, float(self._listen_pref.speech_threshold),
            lambda kind, wav: worker.submit(kind, wav),
            lambda level: self.post("level", (session, level)),
            lambda message: self.post("status", (session, message)),
            lambda active: self.post("voice", session) if active and source != "microphone" else None,
            context=self._phase_epoch, maximum_sec=self._listen_pref.caption_window_seconds,
            streaming=self._listen_pref.streaming_captions,
            on_ready=lambda:self.post('capture_ready',session),
            **options, **self._capture_confirmation_options())
        self.audio.start()

    def stop_listening(self, drain=True):
        self._cancel_answer_prefetch()
        if self.stopping and drain:
            return
        if drain and (self.audio is not None or self.speech is not None):
            self.stopping = True
            self.listening = self.preparing = False
            self.window.set_listening(False)
            self.window.start_button.setEnabled(False)
            self.window.set_status("正在停止收音并保留最后一句及译文…")
            capture, speech, session = self.audio, self.speech, self._session
            if capture:
                capture.stop()
            def finish_audio():
                for worker in (capture,):
                    thread = getattr(worker, "thread", None)
                    if thread:
                        thread.join(timeout=5)
                        if thread.is_alive():
                            self.post("drain_failed", (session, "音频线程未能及时停止，最后一句可能未完成；请核对后再开始。"))
                            return
                if speech:
                    speech.stop(drain=True)
                    thread = getattr(speech, "thread", None)
                    if thread:
                        thread.join(timeout=30)
                        if thread.is_alive():
                            self.post("drain_failed", (session, "识别排空超时，最后一句可能未完成；请核对已显示文字。"))
                            return
                self.post("capture_drained", session)
            self.background(finish_audio, "PrismDrainAudio")
            return
        self._session += 1
        if self.audio:
            self.audio.stop()
            self.audio = None
        if self.speech:
            self.speech.stop()
            self.speech = None
        if self.translation:
            self.translation.stop()
            self.translation = None
        self.listening = self.preparing = False
        self.stopping = False
        self._capture_started = False
        self._partial_id = None
        self._partial_text = ""
        self.assembler.reset()
        self._seal_all_caption_sources()
        self.window.set_listening(False)
        self.window.set_metrics({"audio_level": 0})
        self.window.set_status("收音已停止；本次文字仍在内存中，可手动导出。")

    def _finish_listening_drain(self, session):
        if session != self._session:
            return
        translator = self.translation
        def wait_translation():
            deadline = time.monotonic() + 30
            while getattr(translator, "pending", 0) and time.monotonic() < deadline:
                time.sleep(.05)
            self.post("listening_drained", (session, not bool(getattr(translator, "pending", 0))))
        self.background(wait_translation, "PrismDrainTranslation")

    def assign_speaker(self, turn_id, speaker_id, speaker_label):
        self._cancel_answer_prefetch()
        value={"speaker_id": speaker_id, "speaker_label": speaker_label, "speaker_status": "manual"}
        self._manual_turns[str(turn_id)]=value
        if speaker_id:self._speaker_aliases[str(speaker_id)]=speaker_label
        self._apply_speaker_revision({"turn_id":turn_id,**value})

    def change_speaker_role(self, speaker_id, role):
        self._cancel_answer_prefetch()
        self.intent.set_role(speaker_id,'ignore' if role=='candidate' else role)
        self.question_context.set_role(speaker_id,role)
        if role in ('ignore','candidate'):
            filtered=self.question_context.filter_question(self.assembler.text)
            if filtered!=self.assembler.text:
                self.assembler.text=filtered
                self.assembler.version+=1
                self._question_original=filtered
                dropped=[item[0] for item in self._requests if item[0] in self._automatic_ids]
                self._requests=deque(item for item in self._requests if item[0] not in self._automatic_ids)
                for identity in dropped:
                    self._question_packets.pop(identity,None)
                    self._question_raw_by_request.pop(identity,None)
                if self.busy and self._active in self._automatic_ids:
                    self._cancel.set()
            if self.question_context.last_filter_warnings:
                self._question_uncertain=True
                self.window.set_status('角色已更正；有重复短句无法可靠从当前题中移除，请在问题预览中核对。','warning')
        self._show_question_packet()

    def _make_question_packet(self, question=None, *, raw_question=None, manual=False):
        raw=self._question_original or self.assembler.text
        if question is None:
            question=self._restored(raw)[0] if raw else ''
        packet=self.question_context.build(question,raw_question=raw if raw_question is None else raw_question,
            research_profile=self.pref.research_profile,preparation=self.preparation,manual=manual)
        # Accepted setup belongs to the input, but acceptance alone does not
        # prove its speaker is an interviewer. Preserve it as attributed context
        # until an actual question begins, unless a human supplied the role.
        found_question=False;turns=[]
        for turn in packet.turns:
            found_question=found_question or looks_like_question(turn.raw_en) or has_complete_direct_question(turn.raw_en)
            if not found_question and turn.role_source!='manual':
                turn=replace(turn,role='unknown')
            turns.append(turn)
        return replace(packet,turns=tuple(turns))

    def _show_question_packet(self, question=None):
        packet=self._make_question_packet(question)
        if packet.question_en != self._last_question_display:
            self._session_dirty = True
        self._last_question_display=packet.question_en
        if hasattr(self.window,'set_question_packet'):
            self.window.set_question_packet(packet.to_ui_data())
        else:
            self.window.set_question(packet.question_en)
        return packet

    def _apply_speaker_revision(self, revision):
        self._session_dirty = True
        turn=str(revision.get("turn_id") or "")
        value=self._manual_turns.get(turn, revision)
        sid=value.get("speaker_id")
        label=self._speaker_aliases.get(str(sid),value.get("speaker_label") or "发言者待补认")
        self.question_context.update_speaker(turn,sid,label,value.get('speaker_status','confirmed'),
            manual=value.get('speaker_status')=='manual')
        for row in self._transcripts:
            if str(row.get("turn_id") or "")==turn:
                row.update(speaker_id=sid,speaker_label=label,speaker_status=value.get("speaker_status","confirmed"))
                self._caption_metadata[row["id"]].update(speaker_id=sid,speaker_label=label,
                    speaker_status=value.get("speaker_status","confirmed"))
        self.window.update_speaker(turn,sid,label,value.get("speaker_confidence"),
                                   status=value.get("speaker_status","confirmed"))
        if str(self._live_data.get("turn_id") or "")==turn:
            self._live_data.update(speaker_id=sid,speaker_label=label)
            self._publish_live()
        self._show_question_packet()

    def _publish_live(self):
        if self._live_data:
            d=self._live_data
            self.window.set_live_translation(d.get("en",""),d.get("zh",""),d.get("speaker_label","当前发言"),
                d.get("state","正在听译"),raw=d.get("raw",""),turn_id=d.get("turn_id",""),
                speaker_id=d.get("speaker_id"),speaker_status=d.get("speaker_status"))

    def _restored(self, raw, final=True, words=None):
        if self.pref.clean_speech:
            value=self.restorer.restore(raw,final=final,word_times=words)
            return value.clean,value.display
        return raw,raw

    def _confirmed_source_boundary(self, context):
        """Advance even an empty overlap commit; retire only its old draft.

        A live decoder's audio clock can run far into silence after its last
        lexical word. Compare lexical watermarks rather than that silent tail.
        A genuinely newer draft stays visible and keeps automatic answers gated.
        """
        if not isinstance(context, dict) or not context.get('stream_id'):
            return
        clock = (context['stream_id'], context.get('phase_epoch'))
        end = context.get('end_sample', 0)
        if not isinstance(end, (int, float)):
            return
        self._last_final_end[clock] = max(self._last_final_end.get(clock, 0), end)
        # The latest visible card may belong to the microphone. Retire this
        # input's old draft by its own lexical watermark, independently of it.
        if (self._partial_clock == clock and self._partial_text_end_sample <= self._last_final_end[clock]):
            self._partial_id, self._partial_text = None, ""
        live = self._live_data
        if (live.get('clock') == clock and
                live.get('text_end_sample', 0) <= self._last_final_end[clock]):
            self._partial_id = None
            self._partial_text = ''
            identity = live.get('entry_id')
            row = next((r for r in reversed(self._transcripts) if r['id'] == identity), None)
            if row:
                self._live_token += 1
                self._live_data = {'en': row['en'], 'source': self._caption_sources.get(identity, row['en']),
                    'raw': row.get('raw_en', row['en']), 'zh': row['zh'], 'token': self._live_token,
                    'entry_id': identity, 'state': '已成句', **self._caption_metadata.get(identity, {})}
                self._publish_live()

    def _caption_paragraph_full(self, incoming=""):
        raw = self.caption_turn.committed
        # An ASR crop is not a new paragraph. In particular a brief, hesitant
        # continuation such as "Or convey" must not become a standalone card
        # solely because the preceding crop happened to end in a period.
        # Acoustic speaker/turn changes are handled separately by the caller.
        continuation = str(incoming or '').strip()
        if (len(continuation.split()) <= 4
                and re.match(r'^(?:and|or|but|because|so|I\s+mean)\b', continuation, re.I)
                and not re.match(r'^(?:and|or|but|so)[, ]+I\s+(?:agree|disagree|see|understand)\b',
                                 continuation, re.I)):
            return False
        if (len(raw) < 180 or re.search(r'(?:\.{2,}|…)\s*$', raw) or QuestionAssembler.incomplete_text(raw)
                or caption_tail_incomplete(raw) or sentence_continues(raw, incoming)):
            return False
        # Confirmation windows and character limits are not sentence endings.
        # Look at the next confirmed words before sealing the translation unit;
        # Whisper may itself have inserted a period mid-phrase at a hard cut.
        return raw.rstrip().endswith(('.', '?', '!'))

    def _restore_caption(self, raw, kind, context=None):
        hard_cut = bool(context.get("hard_cut")) if isinstance(context, dict) else False
        final = (kind != "partial" and not caption_tail_incomplete(raw)
                 and not QuestionAssembler.incomplete_text(raw)
                 and not re.search(r'(?:\.{2,}|…)\s*$', raw)
                 and (not hard_cut or raw.rstrip().endswith(('.', '?', '!'))))
        return self._restored(raw, final=final)

    @staticmethod
    def _closing_courtesy_only(text):
        """A standalone closing is a caption, not a changed interview ask.

        This deliberately narrow allowlist also prevents an occasional Whisper
        silence-tail 'Thank you' from replacing a completed prepared answer.
        Mixed speech and any substantive follow-up remain untouched.
        """
        return bool(re.fullmatch(
            r"\s*(?:thanks(?:\s+very\s+much)?(?:\s+for\s+listening)?|"
            r"thank\s+you(?:\s+very\s+much)?(?:\s+for\s+listening)?|"
            r"you\s+for\s+listening)[.,!?:;\s]*", str(text), re.I))

    @staticmethod
    def _acknowledgment_only(text):
        # Declarative acknowledgments do not add a question or a constraint.
        # Retain question-mark confirmations and every mixed/content-bearing turn.
        return bool(re.fullmatch(r"\s*(?:okay|ok|right|all\s+right|I\s+see|"
            r"got\s+it|understood|makes\s+sense|that\s+makes\s+sense)[.,!\s]*",str(text),re.I))

    def _queue_caption_translation(self, translation_key, cleaned, identity, combined):
        if self.translation:
            self.translation.submit(translation_key, cleaned)
        if self.pref.grammar_refinement:
            self._grammar_serial += 1
            self._grammar_versions[identity] = self._grammar_serial
            self.grammar.submit(identity, self._grammar_serial, combined, cleaned)

    @staticmethod
    def _caption_translation_prefix(source, current):
        """An exact confirmed-source prefix ending outside a lexical token.

        ``can``/``cannot``, contractions, decimals and compound words must not
        become prefix matches merely because their initial characters agree.
        No case/punctuation normalization or inference about speech is allowed.
        """
        if not source or not current or not current.startswith(source):
            return False
        boundary = len(source)
        return not any(word.start() < boundary < word.end() for word in
                       re.finditer(r"\w+(?:[’']\w+)*(?:[.\-]\w+)*%?", current))

    def _answer_prefetch_policy(self):
        # In-memory equality only. Never export or log this tuple: it includes
        # endpoint-scoped session credentials so a changed key cannot reuse work.
        names = ('provider', 'answer_model', 'answer_speed', 'cloud_consent',
                 'auto_answer', 'interview_phase', 'input_source', 'speaker_id', 'microphone_id', 'caption_speaker_mode',
                 'share_evidence', 'api_base_url', 'api_key', 'session_api_keys',
                 'qwen_region', 'free_quota_confirmed', 'qwen_paid_usage_authorized',
                 'answer_timeout', 'research_profile', 'candidate_profile',
                 'prefer_prepared_answers', 'max_answer_words', 'brief_answer', 'top_k')
        # Default Qwen endpoints/models are canonicalized by Preferences.
        # The empty "automatic official endpoint" and its explicit equivalent
        # must not discard a correctly prepared answer during save_settings.
        canonical = replace(self.pref)
        return tuple((name, json.dumps(getattr(canonical, name), sort_keys=True,
                                      ensure_ascii=False)) for name in names)

    def _answer_prefetch_scope(self):
        stamps = []
        for path in (app_dir() / 'knowledge.json', app_dir() / 'research_facts.json'):
            try:
                stat = path.stat()
                stamps.append((stat.st_mtime_ns, stat.st_size))
            except OSError:
                stamps.append(None)
        return (self._session, self._phase_epoch, self.assembler.generation,
                self._answer_prefetch_library_epoch, tuple(stamps),
                self._answer_prefetch_policy(),
                json.dumps(list(self._history), sort_keys=True, ensure_ascii=False))

    def _cancel_answer_prefetch(self):
        self._answer_prefetch_candidate = None
        slot = self._answer_prefetch
        if slot and not slot.get('promoted'):
            slot['cancel'].set()
            slot['events'].clear()
            if slot.get('finished'):
                self._answer_prefetch = None

    def _matching_answer_prefetch(self, raw, question):
        slot = self._answer_prefetch
        return (slot if slot and not slot.get('promoted') and not slot['cancel'].is_set()
                and slot['raw'] == raw and slot['question'] == question
                and slot['scope'] == self._answer_prefetch_scope() else None)

    def _maybe_prefetch_answer(self):
        """Overlap Qwen computation with endpoint confirmation, without UI output.

        This does not dispatch, consume, seal or shorten the source question.
        At most one hidden worker and two attempts per question are allowed.
        Every result still needs a real endpoint and an exact source/scope match.
        """
        if (self._closed or self._capture_error_handled or self._microphone_source() or not self.listening or self.busy or self._requests
                or self.pref.provider != 'qwen' or not self.pref.cloud_consent
                or not self.pref.auto_answer or self.pref.interview_phase != 'qa'
                or self._question_uncertain or self._semantic_response):
            self._cancel_answer_prefetch()
            return
        if is_non_request_context(self._question_original or self.assembler.text):
            self._cancel_answer_prefetch()
            return
        raw = speculative_question_snapshot(self._question_original or self.assembler.text)
        if not raw:
            self._cancel_answer_prefetch()
            return
        scope = self._answer_prefetch_scope()
        key = (scope, raw)
        slot = self._answer_prefetch
        if slot:
            if not slot.get('promoted') and (slot['scope'], slot['raw']) != key:
                self._cancel_answer_prefetch()
            # Do not pile up cancelled network workers; wait for its terminal.
            if self._answer_prefetch is not None:
                return
        generation = (self._session, self._phase_epoch, self.assembler.generation)
        if generation != self._answer_prefetch_generation:
            self._answer_prefetch_generation = generation
            self._answer_prefetch_attempts = 0
            self._answer_prefetch_last_key = None
        if self._answer_prefetch_attempts >= 2 or self._answer_prefetch_last_key == key:
            return
        now = time.monotonic()
        candidate = self._answer_prefetch_candidate
        if candidate is None or candidate[0] != key:
            self._answer_prefetch_candidate = (key, now)
            return
        if now - candidate[1] < .3:
            return
        question, _ = self._restored(raw)
        packet = self._make_question_packet(question, raw_question=raw)
        if packet.interaction_kind in ('setup', 'invitation'):
            return
        from .quick_reply import quick_coordination_reply
        if quick_coordination_reply(question, packet=packet) is not None:
            return  # The normal endpoint already answers coordination locally.
        identity = str(uuid.uuid4())
        prefs = replace(self.pref)
        history = list(self._history)
        cancel = threading.Event()
        slot = {'id': identity, 'raw': raw, 'question': question, 'scope': scope,
                'cancel': cancel, 'events': [], 'bytes': 0, 'finished': False,
                'promoted': False, 'started_at': time.perf_counter()}
        self._answer_prefetch = slot
        self._answer_prefetch_attempts += 1
        self._answer_prefetch_last_key = key
        def emit(kind, payload):
            self.post('answer_prefetch_event', (identity, kind, payload))
        def work():
            began = time.perf_counter()
            provider_started = None
            first = False
            def delta(field, value):
                nonlocal first
                if field in ('answer_en', 'answer_pair') and value and not first:
                    first = True
                    emit('answer_stage', (identity, {'answer_provider_ttft_ms':
                        max(0., (time.perf_counter() - (provider_started or began))*1000)}))
                emit('answer_delta', (identity, field, value))
            try:
                if cancel.is_set():
                    raise AnswerCancelled('已取消预先准备。')
                hits, coverage = self._retrieve_question(question, packet, prefs.top_k,
                    prefs=prefs, return_coverage=True)
                warning = ''
                if coverage.get('uncovered_subquestion_indices'):
                    missing = ', '.join(map(str, coverage['uncovered_subquestion_indices']))
                    warning = f'第 {missing} 个子问未分配到检索资料；请区分一般推理与已有证据，不要补造个人结果。'
                prepared_packet = replace(packet, retrieval_coverage=coverage,
                    warnings=packet.warnings + ((warning,) if warning else ()))
                provider_started = time.perf_counter()
                emit('answer_stage', (identity, {'answer_retrieval_ms':
                    max(0., (provider_started-began)*1000)}))
                result = self.service.answer(question, hits, on_delta=delta,
                    cancel_event=cancel, history=history, request_id=identity,
                    preferences=prefs, raw_question=raw,
                    interview_packet=prepared_packet.to_prompt_data(max_chars=4200))
                result.diagnostics['retrieval_coverage'] = coverage
                if warning and warning not in result.warnings:
                    result.warnings.append(warning)
                if cancel.is_set():
                    raise AnswerCancelled('已取消预先准备。')
                emit('answer', (identity, question, result))
            except Exception as exc:
                emit('answer_error', (identity, str(exc), cancel.is_set()))
        self.background(work, 'PrismAnswerPrefetch')

    def _invalidate_changed_automatic_question(self, question):
        """Freeze out obsolete suggestions as soon as confirmed words change.

        A continuation may not reach its next endpoint for several seconds.
        During that interval, old deltas/results must not be displayed against
        the revised question, nor should an obsolete queued request start.
        Manual requests keep their deliberately selected question snapshots.
        """
        def words(value):
            return re.findall(r"\w+(?:['’]\w+)*|[%=<>±×÷]", str(value).casefold())
        current = words(question)
        slot = self._answer_prefetch
        if (slot and not slot.get('promoted') and
                str(question).strip() != slot['raw'] and
                speculative_question_snapshot(question) != slot['raw']):
            self._cancel_answer_prefetch()
        submitted = self._last_automatic_snapshot
        if (submitted and submitted[0] == self.assembler.generation
                and words(submitted[1]) != current):
            self._last_automatic_snapshot = None
        obsolete = {identity for identity, queued, _ in self._requests
                    if identity in self._automatic_ids and identity not in self._sealed_automatic_ids
                    and words(self._question_raw_by_request.get(identity, queued)) != current}
        if obsolete:
            self._requests = deque(item for item in self._requests if item[0] not in obsolete)
            for identity in obsolete:
                self._question_packets.pop(identity, None)
                self._question_raw_by_request.pop(identity, None)
                self._answer_request_timing.pop(identity, None)
                self._automatic_ids.discard(identity)
        if (self.busy and self._active in self._automatic_ids
                and self._active not in self._sealed_automatic_ids
                and words(self._question_raw_by_request.get(self._active, '')) != current):
            self._cancel.set()
            self._discard_unfinished_answer()
            self.window.set_status('问题有补充，正在保留完整问题；说完后会自动重新回答。')

    def on_transcript(self, kind, english, context=None):
        source = context.get("input_source") if isinstance(context, dict) else None
        if source not in ("loopback", "microphone"):
            source = "microphone" if self._microphone_source() else "loopback"
        self._activate_caption_source(source)
        try:
            self._on_source_transcript(kind, english, context)
        finally:
            self._store_caption_source()

    def _on_source_transcript(self, kind, english, context=None):
        if kind == "final" and str(english).strip():
            self._session_dirty = True
        epoch = context.get("phase_epoch") if isinstance(context, dict) else None
        timed = bool(context.get("word_timed")) if isinstance(context, dict) else False
        microphone = self._caption_active_source == "microphone"
        single_remote = self._single_remote_source() and not microphone
        speaker = ({key: context.get(key) for key in ("speaker_id", "speaker_label", "turn_id")}
                   if isinstance(context, dict) else {})
        speaker["speaker_label"] = self._speaker_aliases.get(str(speaker.get("speaker_id")),
            speaker.get("speaker_label") or "发言者待补认")
        speaker["speaker_status"] = context.get("speaker_status") if isinstance(context,dict) else None
        if microphone:
            speaker = {"speaker_id": "local-microphone", "speaker_label": "我 · 麦克风",
                       "turn_id": f"mic-source-{self._session}-{epoch if epoch is not None else self._phase_epoch}",
                       "speaker_status": "source"}
        elif single_remote:
            # The user selected a single remote SOURCE. Do not infer or claim
            # acoustic identity, and do not allow short-window diarizer guesses
            # to relabel it. Other system-output audio shares this source too.
            speaker = {"speaker_id": None, "speaker_label": "对方 · 会议声音",
                       "turn_id": f"zoom-source-{self._session}-{epoch if epoch is not None else self._phase_epoch}",
                       "speaker_status": "source"}
        elif str(speaker.get("turn_id")) in self._manual_turns:
            speaker.update(self._manual_turns[str(speaker["turn_id"])])
        if isinstance(context,dict) and not single_remote and not microphone:
            for revision in context.get("speaker_revisions",[]):
                self._apply_speaker_revision(revision)
        if epoch is not None and epoch != self._phase_epoch:
            # Preserve an old phase's confirmed words as history, while keeping
            # them out of the new QA question and its caption card.
            if kind == "final" and str(english).strip():
                identity = str(uuid.uuid4())
                clean,display=self._restored(english)
                self._caption_sources[identity] = clean
                self._caption_raw[identity] = english
                self._caption_metadata[identity] = speaker
                self.window.append_transcript(identity, display, "", partial=False, raw_en=english, **speaker)
                self._transcripts.append({"id": identity, "en": display, "raw_en":english, "zh": "",
                                          "at": datetime.now().isoformat(timespec="seconds"), **speaker})
                if self.translation:
                    self.translation.submit((identity, clean), clean)
                if self._live_data:
                    self._publish_live()
                else:
                    self.window.set_live_translation('')
            return
        if kind == "rejected":
            if not microphone:
                self._cancel_answer_prefetch()
            identity = self.caption_turn.identity
            committed = self.caption_turn.committed
            if not microphone:
                self._question_uncertain = True
                self._question_intro_tail = ''  # Never bridge missing remote audio.
            self._partial_id = None
            self._partial_text = ""
            if identity:
                if committed:
                    record = next((row for row in reversed(self._transcripts) if row["id"] == identity), {})
                    # Rejecting an unconfirmed tail must not repunctuate the
                    # accepted prefix and invalidate its outstanding MT key.
                    clean = self._caption_sources.get(identity)
                    if clean is None:
                        clean,display=self._restore_caption(committed,"final",{"hard_cut":True})
                        self._caption_sources[identity] = clean
                        if self.translation:
                            self.translation.submit((identity,clean),clean)
                    else:
                        display = record.get("en",clean)
                    self.window.append_transcript(identity, display, record.get("zh", ""), partial=False, raw_en=committed,
                                                  translation_pending=record.get("translation_pending",self.translation is not None),
                                                  **self._caption_metadata.get(identity, {}))
                    self._live_token+=1
                    self._live_data={"en":display,"source":clean,"raw":committed,"zh":record.get('zh',''),
                        "token":self._live_token,"entry_id":identity,"state":"已保留确认部分 · 后段识别未通过",
                        **self._caption_metadata.get(identity,{})}
                    self._publish_live()
                else:
                    self._caption_sources.pop(identity, None)
                    self.window.remove_transcript(identity)
                    self.caption_turn.seal()
            if not microphone and self._question_uncertain:
                self.window.set_status("刚才一段未通过识别确认；请核对问题后手动生成，或等待下一句。", "warning")
            return
        if not str(english).strip():
            if kind == 'final' and isinstance(context, dict) and context.get('empty_commit'):
                self._confirmed_source_boundary(context)
            if kind=='partial' and isinstance(context,dict) and context.get('stream_id'):
                # A committed boundary can leave no lexical tail. Retire the
                # provisional frame even when the decoder emits no new word.
                identity=self.caption_turn.identity
                row=next((r for r in reversed(self._transcripts) if r['id']==identity),None)
                if row and (not self._live_data.get('clock') or self._live_data['clock']==(context.get('stream_id'),epoch)):
                    self._live_token+=1
                    self._live_data={"en":row['en'],"source":self._caption_sources[identity],"raw":row.get('raw_en',row['en']),
                        "zh":row['zh'],"token":self._live_token,"entry_id":identity,"state":"已成句",
                        **self._caption_metadata.get(identity,{})}
                    self._partial_id=None
                    self._publish_live()
            return
        if kind=="partial" and isinstance(context,dict):
            clock=(context.get("stream_id"),epoch)
            lexical_end=context.get('text_end_sample',context.get('end_sample',0))
            if lexical_end and lexical_end<=self._last_final_end.get(clock,-1):
                self._confirmed_source_boundary(dict(context,end_sample=self._last_final_end[clock]))
                return
        if kind != "partial":
            # Acoustic turns, not arbitrary ASR windows, determine who owns a
            # card. Bound long monologues into readable paragraphs independently
            # of QuestionAssembler, which must retain the whole question.
            current = self._caption_metadata.get(self.caption_turn.identity, {})
            changed_turn = (speaker.get("turn_id") is not None
                            and speaker.get("turn_id") != self._caption_acoustic_turn)
            changed_voice = (speaker.get("speaker_id") is not None
                             and current.get("speaker_id") is not None
                             and speaker["speaker_id"] != current["speaker_id"])
            committed = self.caption_turn.committed
            paragraph_full = self._caption_paragraph_full(english)
            inactive = self._caption_inactivity(context, time.monotonic()) > 12.0
            if self.caption_turn.identity and (changed_turn or changed_voice or paragraph_full or (single_remote and inactive)):
                self.caption_turn.seal()
            self._caption_acoustic_turn = speaker.get("turn_id")
        identity, combined = self.caption_turn.update(english, final=kind != "partial", deduplicate=not timed)
        cleaned,display=self._restore_caption(combined,kind,context)
        # Keys retain exact source revisions. A late MT may show a verified
        # confirmed prefix; a conflicting ASR revision must never overwrite it.
        translation_key = (identity, cleaned)
        if kind == "partial":
            self._partial_id = identity
            self._partial_text = combined
            self._partial_clock = (context.get("stream_id"), epoch) if isinstance(context, dict) else None
            self._partial_text_end_sample = context.get("text_end_sample", context.get("end_sample", 0)) if isinstance(context, dict) else 0
            old=self._live_data
            keep_zh=old.get("zh","") if old.get("token")==self._live_token and cleaned.startswith(old.get("source","\0")) else ""
            self._live_data={"en":display,"source":cleaned,"raw":combined,"zh":keep_zh,"token":self._live_token,
                "entry_id":identity,"speaker_label":speaker["speaker_label"] if single_remote or microphone else "当前发言 · 待补认",
                "state":"正在听译 · 词句会修订",
                "clock":(context.get('stream_id'),epoch) if isinstance(context,dict) else None,
                "text_end_sample":context.get('text_end_sample',context.get('end_sample',0)) if isinstance(context,dict) else 0}
            if single_remote or microphone:
                self._live_data.update(speaker)
            self._publish_live()
            if self.pref.live_translation and self.translation and len(cleaned.split())>=2:
                self.translation.submit_preview(("live",self._phase_epoch,self._live_token),cleaned)
            if not microphone and self.kb.passages and len(cleaned.split())>=7 and time.monotonic()-self._last_prefetch>1.2:
                self._last_prefetch=time.monotonic()
                self.background(lambda:self.preparation.search(self.kb,cleaned,self.pref.top_k),"PrismQuestionPrefetch")
            return
        self._caption_sources[identity] = cleaned
        self._caption_raw[identity] = combined
        self._caption_metadata[identity] = speaker
        previous = next((row for row in reversed(self._transcripts) if row["id"] == identity), None)
        previous_zh=previous.get('zh','') if previous else ''
        # Keep the translated prefix visible until the newer confirmed text is
        # translated. Clearing it would shrink and regrow the whole card.
        translation_pending = self.translation is not None
        self.window.append_transcript(identity, display, previous_zh, partial=False, raw_en=combined,
                                      translation_pending=translation_pending, **speaker)
        newer_live=(isinstance(context,dict) and context.get('stream_id') is not None
                    and self._live_data.get('clock')==(context.get('stream_id'),epoch)
                    and self._live_data.get('text_end_sample',0)>context.get('end_sample',0))
        if not newer_live:
            self._live_token+=1
            self._live_data={"en":display,"source":cleaned,"raw":combined,"zh":previous_zh,"token":self._live_token,
                "entry_id":identity,"state":"已确认片段 · 正在更新中文",**speaker}
        self._publish_live()
        if isinstance(context,dict):
            clock=(context.get("stream_id"),epoch)
            self._last_final_end[clock]=max(self._last_final_end.get(clock,0),context.get("end_sample",0))
        if not newer_live:
            self._partial_id = None
            self._partial_text = ""
        if previous is not None:
            previous.update(en=display, raw_en=combined, zh=previous_zh,
                            translation_pending=translation_pending, **speaker)
        else:
            self._transcripts.append({"id": identity, "en": display, "raw_en":combined, "zh": "",
                                      "translation_pending":translation_pending,
                                      "at": datetime.now().isoformat(timespec="seconds"), **speaker})
        if microphone:
            # My speech is readable/saved, but never becomes the interviewer's
            # question, invalidates its integrity or starts an automatic answer.
            self._queue_caption_translation(translation_key, cleaned, identity, combined)
            return
        gap = bool(context.get('recognition_gap')) if isinstance(context, dict) else False
        self._recognition_gap_active = gap
        if gap:
            self._question_uncertain = True
            self._cancel_answer_prefetch()
            self._question_intro_tail = ''
        try:
            self._route_question_fragment(english, speaker, identity, timed=timed)
        finally:
            self._recognition_gap_active = False
        if gap:
            self.window.set_status('本段部分语音未通过确认；已保留识别文字，请核对完整问题后手动生成。', 'warning')
        self._queue_caption_translation(translation_key, cleaned, identity, combined)

    def _record_question_fragment(self, identity, english, speaker, accepted):
        self._context_serial += 1
        source_id = f'{identity}:{self._context_serial}'
        self.question_context.record(source_id, english, self._restored(english)[0],
            speaker_id=speaker.get('speaker_id'), speaker_label=speaker.get('speaker_label',''),
            turn_id=speaker.get('turn_id') or '', speaker_status=speaker.get('speaker_status'),
            accepted=accepted)
        return source_id

    def _begin_semantic_question(self):
        self._cancel_answer_prefetch()
        self.assembler.reset()
        self.question_context.begin_question()
        self.intent.begin_question()
        self._question_original = ''
        self._answering_voice_seen = False
        self._semantic_response = False
        self._question_intro_tail = ''
        self._awaiting_actual_question = False
        # A confirmed new question has its own recognition integrity. Do not
        # inherit a rejected segment from the previous answer/question; normal
        # additions within this question deliberately never clear this flag.
        self._question_uncertain = self._recognition_gap_active

    def _seal_semantic_question(self, *, confirmed_response=False):
        """Freeze a confirmed turn, independently of global voice/ASR activity."""
        if self._capture_error_handled or self._semantic_response or not has_complete_direct_question(
                self.assembler.text, require_terminal=not confirmed_response):
            return False
        if self._question_uncertain:
            return False
        self._semantic_response = True
        if self.pref.auto_answer and self.pref.interview_phase == 'qa':
            submitted = self._last_automatic_snapshot
            same_snapshot = (submitted and submitted[0] == self.assembler.generation
                and ' '.join(submitted[1].split()) == ' '.join(self._question_original.split()))
            if same_snapshot:
                # Adding a reply crop and trimming it back can increment the
                # assembly version without changing a single question word.
                # Freeze the prior request even when its worker already ended.
                self.assembler.consume(retain=True)
                if submitted[2] in self._automatic_ids:
                    self._sealed_automatic_ids.add(submitted[2])
            elif self.assembler.version != self.assembler._answered_version:
                self.request_answer(self.assembler.text, automatic=True, endpoint='semantic_boundary')
            else:
                # The same question may already be answering after a brief
                # pause. A subsequent reply freezes that existing snapshot.
                for identity in self._automatic_ids:
                    if self._question_raw_by_request.get(identity) == self._question_original:
                        self._sealed_automatic_ids.add(identity)
        return True

    def _completed_independent_question(self, english):
        """Recognize only an independent ask after a matching completed answer.

        A completed worker alone is never a turn boundary. Retained qualifiers,
        corrections and referential follow-ups need the earlier source for
        retrieval, so they deliberately remain in the same question. No draft,
        failed request or stale generation may retire a confirmed question.
        """
        snapshot = self._last_automatic_snapshot
        if (self._semantic_response or self._question_uncertain or not snapshot
                or snapshot[0] != self.assembler.generation or not self._answers):
            return False
        completed = self._answers[-1]['result']
        if getattr(completed, 'request_id', None) != snapshot[2]:
            return False
        normalized = lambda value: ' '.join(str(value).split())
        if (normalized(snapshot[1]) != normalized(self._question_original)
                or normalized(snapshot[1]) != normalized(self.assembler.text)):
            return False
        dependent = re.search(
            r'\b(?:it|its|this|that|these|those|such|same|else|other|additional|further|more)\b',
            english, re.I) or re.search(
            r'^(?:(?:so|well|okay|ok|now|but|then)[, :]+)*'
            r'(?:and|also|or|plus|especially|for\s+example)\b', english, re.I)
        return bool(not dependent and has_complete_direct_question(english)
                    and next_addressed_question_offset(english) == 0)

    def _route_question_fragment(self, english, speaker, identity, *, timed=False):
        """Route confirmed words; captions retain the complete unsplit original.

        Explicit introduction/response cues, not each ASR punctuation mark,
        bound question snapshots. No provisional words or model guesses enter.
        """
        english = str(english).strip()
        if not english:
            return
        if self._completed_independent_question(english):
            # The successful earlier answer remains visible and in history;
            # begin_question retains its source ledger as follow-up context.
            self._begin_semantic_question()
        if not self._semantic_response:
            self._question_intro_tail = ''
        # Resolve an invitation split across ASR windows before it can become a
        # spurious technical question. A prefix such as 'Do you have any' also
        # matches substantive asks, so remove it only after 'questions' confirms
        # an invitation; e.g. 'any evidence' continues normally.
        invite_start = invitation_prefix_offset(self.assembler.text)
        if invite_start is not None:
            tail = self.assembler.text[invite_start:]
            combined = join_fragments(tail, english)
            pieces = invitation_segments(combined)
            if pieces and pieces[0][0]:
                self.assembler.text = self.assembler.text[:invite_start].strip()
                self._question_original = self.assembler.text
                english = combined
        pieces = invitation_segments(english)
        if any(invitation for invitation, _ in pieces):
            for invitation, part in pieces:
                if not invitation:
                    self._route_question_fragment(part, speaker, identity, timed=timed)
                    continue
                # A moderator opening the floor separates turns. Freeze any
                # preceding actual ask, but never send the invitation itself.
                self._seal_semantic_question()
                self._begin_semantic_question()
                self._awaiting_actual_question = True
                self._record_question_fragment(identity, part, speaker, False)
                self._show_question_packet('')
                self.window.set_status('已识别为邀请提问，正在等待实际问题。')
            return
        if self._awaiting_actual_question:
            assembled = (join_fragments(self.assembler.text, english) if not timed else
                         ' '.join(part for part in (self.assembler.text, english) if part))
            onset = question_after_moderation_offset(assembled)
            if onset is not None:
                # A rejected moderator-name/thanks crop precedes this actual
                # question. Start integrity tracking at its first confirmed
                # word, even if the question opening spans several crops.
                # Scientific setup cannot match the pure-moderation prefix.
                english = assembled[onset:].strip()
                self._begin_semantic_question()
                timed = True
        intro = announced_question_offset(english)
        if intro is not None and intro > 0:
            self._route_question_fragment(english[:intro], speaker, identity, timed=timed)
            self._route_question_fragment(english[intro:], speaker, identity, timed=timed)
            return
        if self._semantic_response:
            if self._question_intro_tail:
                joined_intro = join_fragments(self._question_intro_tail, english)
                if announces_question(joined_intro) or question_introduction_prefix(joined_intro):
                    english = joined_intro
                self._question_intro_tail = ''
            next_ask = next_addressed_question_offset(english)
            if (self.intent.roles.get(str(speaker.get('speaker_id'))) == 'interviewer'
                    and looks_like_question(english)):
                next_ask = 0  # An explicit human role correction outranks cues.
            if next_ask is None:
                self._question_intro_tail = question_introduction_prefix(english)
                self._record_question_fragment(identity, english, speaker, False)
                return
            if next_ask:
                self._record_question_fragment(identity, english[:next_ask].strip(), speaker, False)
                english = english[next_ask:]
            self._begin_semantic_question()
        if self._closing_courtesy_only(english) or acknowledgment_only(english):
            # "Sure/Yep" can be the only boundary between a question and a
            # continuous video answer; still preserve them in the transcript.
            self._seal_semantic_question(confirmed_response=True)
            self._record_question_fragment(identity, english, speaker, False)
            return
        single_remote = self._single_remote_source()
        accepted = True if single_remote else self.intent.accepts(english, speaker.get('speaker_id'))
        if accepted:
            explicit_boundary = bool((self.assembler.text or self._question_uncertain) and
                (starts_new_question(english) or announces_question(english)))
            new_addressed_voice = (speaker.get('speaker_id') is not None
                and self._last_question_speaker is not None
                and str(speaker['speaker_id']) != str(self._last_question_speaker)
                and self.assembler.text and not looks_like_question(self.assembler.text)
                and looks_like_question(english))
            if explicit_boundary:
                self._seal_semantic_question()
            if explicit_boundary or (self._answering_voice_seen and looks_like_question(english)) or new_addressed_voice:
                self._begin_semantic_question()
            if speaker.get('speaker_id') is not None:
                self._last_question_speaker = speaker['speaker_id']
            previous = self.assembler.text
            question = self.assembler.add(english, deduplicate=not timed)
            boundary = confirmed_question_boundary(question)
            if boundary is not None and not self._question_uncertain:
                prefix, suffix = question[:boundary].strip(), question[boundary:].strip()
                # Only the confirmed prefix belongs to this question. Split a
                # single ASR crop's source ledger without altering its caption.
                self.assembler.text = prefix
                self._question_original = prefix
                accepted_part = prefix[len(previous):].strip() if prefix.startswith(previous) else prefix
                if accepted_part:
                    self._record_question_fragment(identity, accepted_part, speaker, True)
                self._invalidate_changed_automatic_question(prefix)
                self._show_question_packet()
                self._seal_semantic_question(confirmed_response=True)
                self._route_question_fragment(suffix, speaker, identity, timed=True)
                return
            self._question_original = question
            self._invalidate_changed_automatic_question(question)
        elif len(english.split()) >= 6:
            self._answering_voice_seen = True
        if not accepted and self.intent._response_active:
            trimmed = re.sub(r"\s*\bI(?:'m| am)\s+not(?:\s+sure)?\b.*$", '',
                self.assembler.text, flags=re.I).strip()
            if trimmed != self.assembler.text:
                self.assembler.text = trimmed
                self.assembler.version += 1
                self._question_original = trimmed
                self._invalidate_changed_automatic_question(trimmed)
        self._record_question_fragment(identity, english, speaker, accepted)
        self._show_question_packet()
        if (self.pref.auto_answer and self.pref.interview_phase == 'qa' and not self.busy
                and has_complete_direct_question(self.assembler.text)):
            if self._question_uncertain:
                self.window.set_status('已识别提问，但其中一段未通过语音确认，暂不自动提交；已保留确认部分，可核对后手动生成。', 'warning')
            else:
                self.window.set_status('已识别提问，正在等待完整句尾或明确答话衔接，将自动生成双语回答。')
        if not accepted and self.assembler.text and not self.assembler.incomplete and looks_like_question(self.assembler.text):
            if not self._seal_semantic_question() and (self.pref.auto_answer
                    and self.pref.interview_phase == 'qa'
                    and self.assembler.version != self.assembler._answered_version
                    and not self._question_uncertain):
                question, _ = self.assembler.consume(retain=True)
                self.request_answer(question, automatic=True)

    def auto_answer(self):
        if not self.listening or self._capture_error_handled or self._microphone_source():
            return
        self._maybe_prefetch_answer()
        if (is_question_invitation(self.assembler.text)
                or invitation_prefix_offset(self.assembler.text) is not None):
            return
        pending = (self.speech.pending_for_source("loopback") if hasattr(self.speech, "pending_for_source") else self.speech.pending) if self.speech else 0
        partial_id = self._remote_partial_id()
        if self._question_uncertain:
            # A rejected middle/end must not be silently bridged by a later
            # accepted fragment. Retire that incomplete turn at real inactivity;
            # its question stays editable for an explicit manual retry.
            now = time.monotonic()
            if (not pending and partial_id is None
                    and now - max(self.assembler.last_voice, self.assembler.last_text) >= self.assembler.continuation_seconds):
                self.assembler.consume()
                self._seal_question_caption()
                self._question_uncertain = False
                self.window.set_status("本轮识别不完整，未自动回答；可核对当前问题后手动生成。", "warning")
            return
        if is_non_request_context(self._question_original or self.assembler.text):
            # Retire recognition gaps above even for quoted/self-questions;
            # otherwise one old incomplete presentation can block later asks.
            return
        if not self.assembler.ready(pending=pending) or partial_id is not None:
            return
        if not (looks_like_question(self.assembler.text) or
                has_complete_direct_question(self.assembler.text, require_terminal=False)):
            # A professor may describe the setup BEFORE asking. Keep that
            # context across ordinary pauses; only retire an idle statement.
            if time.monotonic() - max(self.assembler.last_voice, self.assembler.last_text) > 15:
                self.assembler.consume()
                self._seal_question_caption()
            return
        if self.pref.auto_answer and self.pref.interview_phase == "qa":
            question, _ = self.assembler.consume(retain=True)
            self.request_answer(question, automatic=True)

    def request_answer(self, question, automatic=False, *, endpoint="silence"):
        if self._closed or (automatic and (self._capture_error_handled or self._microphone_source())):
            return
        if not automatic:
            self._cancel_answer_prefetch()
        question = str(question).strip()
        if automatic:
            if self._question_uncertain:
                self._cancel_answer_prefetch()
                return
            question,_=self._restored(question)
            if is_non_request_context(self._question_original or question):
                self._cancel_answer_prefetch()
                return
        if not question:
            self.window.set_status("请等待完整问题，或输入练习问题。", "warning")
            return
        self.save_settings(self.window.get_settings())
        self.assembler.consume(retain=automatic)
        if not automatic:
            self._seal_question_caption()
        self._question_uncertain = False
        if automatic and (self.pref.interview_phase != "qa" or not self.pref.auto_answer):
            return
        if len(self._requests) >= 12:
            self.window.set_status("待回答队列已满（12题）；请先取消或等待，当前问题仍保留在输入框。", "warning")
            return
        if self.busy and self._requests and self._requests[-1][1] == question:
            return
        same_current=' '.join(question.split())==' '.join(self._last_question_display.split())
        raw=(self._question_original or question) if automatic or same_current else question
        prepared = self._matching_answer_prefetch(raw, question) if automatic else None
        identity = prepared['id'] if prepared else str(uuid.uuid4())
        if not prepared:
            self._cancel_answer_prefetch()
        packet=self._make_question_packet(question,raw_question=raw,
            manual=not(automatic or same_current))
        if automatic and packet.interaction_kind in ('setup', 'invitation'):
            self.window.set_status('正在保留对话背景，等待对方提出完整问题。')
            return
        if automatic:
            # Silence alone is not a caption boundary. Only a real answer
            # dispatch ends the question card; presentation keeps its paragraph.
            self._seal_question_caption()
        self._question_raw_by_request[identity]=raw
        self._question_packets[identity]=packet
        requested_at = time.perf_counter()
        now = time.monotonic()
        timing = {'requested_at': requested_at, 'automatic': bool(automatic),
                  'answer_trigger': endpoint if automatic else 'manual'}
        if automatic and endpoint == 'silence' and self.assembler.last_voice:
            timing['answer_endpoint_ms'] = max(0., (now-self.assembler.last_voice)*1000)
        if automatic and self.assembler.last_text:
            timing['answer_postcommit_ms'] = max(0., (now-self.assembler.last_text)*1000)
        self._answer_request_timing[identity] = timing
        if automatic:
            # A newer automatic question must not wait behind obsolete automatic
            # suggestions. Explicit manual questions retain their queue order.
            obsolete = {item[0] for item in self._requests if item[0] in self._automatic_ids
                        and item[0] not in self._sealed_automatic_ids}
            self._requests = deque(item for item in self._requests if item[0] not in obsolete)
            for old in obsolete:
                self._question_packets.pop(old, None)
                self._question_raw_by_request.pop(old, None)
                self._answer_request_timing.pop(old, None)
                self._automatic_ids.discard(old)
            self._automatic_ids.add(identity)
            if endpoint == "semantic_boundary":
                self._sealed_automatic_ids.add(identity)
            if (self.busy and self._active in self._automatic_ids
                    and self._active not in self._sealed_automatic_ids):
                self._cancel.set()
        self._requests.append((identity, question, replace(self.pref)))
        if automatic:
            self._last_automatic_snapshot = (self.assembler.generation, raw, identity)
        if self.busy:
            self.window.set_status(f"当前回答仍在生成，新问题已排队（{len(self._requests)}题）。")
        else:
            self._next_answer()

    def _next_answer(self):
        if self._closed or self.busy or not self._requests:
            return
        identity, question, prefs = self._requests.popleft()
        prefs = replace(prefs,
                        cloud_consent=bool(prefs.cloud_consent and self.pref.cloud_consent),
                        share_evidence=bool(prefs.share_evidence and self.pref.share_evidence))
        self._active = identity
        self._active_question_en = question
        self._active_prefs = prefs
        self._cancel = threading.Event()
        cancel = self._cancel
        self.busy = True
        self._first_answer = False
        self._stream = {"answer_en": "", "answer_zh": "", "question_zh": ""}
        self._stream_pairs = []
        self._answer_started = time.perf_counter()
        timing = self._answer_request_timing.get(identity, {})
        self.last_answer_timing = {key: value for key, value in timing.items() if key != 'requested_at'}
        self.last_answer_timing['answer_queue_ms'] = max(0.,
            (self._answer_started-timing.get('requested_at',self._answer_started))*1000)
        self.window.set_metrics(self.last_answer_timing)
        self.window.set_busy(True)
        packet = self._question_packets.get(identity)
        self.window.set_answer(sources=[], streaming=True, question_en=question,
            question_preview_en=packet.answer_question_en if packet else question,
            request_id=identity, question_raw_en=self._question_raw_by_request.get(identity, question))
        self.window.set_status("正在回答：" + question[:110])
        self.window.set_answer_model(prefs.ollama_model if prefs.provider == "ollama" else prefs.answer_model or "连接后确认")
        self.service.preferences = prefs
        history = list(self._history)
        packet=self._question_packets.get(identity)
        raw_question=self._question_raw_by_request.get(identity,question)
        prepared = self._matching_answer_prefetch(raw_question, question)
        if prepared and prepared['id'] == identity:
            prepared['promoted'] = True
            self._cancel = prepared['cancel']
            self.last_answer_timing.update({'answer_prefetch_reused': True,
                'answer_prefetch_lead_ms': max(0.,
                    (self._answer_started-prepared['started_at'])*1000)})
            self.window.set_metrics(self.last_answer_timing)
            for kind, payload in prepared['events']:
                self.post(kind, payload)
            prepared['events'].clear()
            if prepared['finished']:
                self._answer_prefetch = None
            return
        def work():
            stage_started = time.perf_counter()
            provider_started = None
            first_english = False
            def delta(field, value):
                nonlocal first_english
                if field in ('answer_en', 'answer_pair') and value and not first_english:
                    first_english = True
                    self.post('answer_stage', (identity, {
                        'answer_provider_ttft_ms': max(0., (time.perf_counter()-(provider_started or stage_started))*1000)}))
                self.post('answer_delta', (identity, field, value))
            try:
                from .quick_reply import quick_coordination_reply
                quick = quick_coordination_reply(question,packet=packet,request_id=identity)
                if quick is not None:
                    if cancel.is_set():
                        raise AnswerCancelled('已取消回答。')
                    provider_started = time.perf_counter()
                    self.post('answer_stage', (identity, {'answer_retrieval_ms': 0., 'answer_local_reply': True}))
                    quick.pairs = [{"id": "local-coordination", "en": quick.answer_en, "zh": quick.answer_zh,
                                    "en_complete": True, "zh_complete": True, "citations": list(quick.citations)}]
                    delta('question_zh', quick.question_zh)
                    delta('answer_pair', json.dumps(quick.pairs[0], ensure_ascii=False))
                    self.post('answer',(identity,question,quick))
                    return
                hits = self._retrieve_question(question,packet,prefs.top_k,prefs=prefs)
                coverage=dict(self.last_retrieval_coverage)
                coverage_warning=''
                if coverage.get('uncovered_subquestion_indices'):
                    missing=', '.join(map(str,coverage['uncovered_subquestion_indices']))
                    coverage_warning=f'第 {missing} 个子问未分配到检索资料；请区分一般推理与已有证据，不要补造个人结果。'
                if packet:
                    packet_for_answer=replace(packet,retrieval_coverage=coverage,
                        warnings=packet.warnings+((coverage_warning,) if coverage_warning else ()))
                else:
                    packet_for_answer=None
                provider_started = time.perf_counter()
                self.post('answer_stage', (identity, {
                    'answer_retrieval_ms': max(0., (provider_started-stage_started)*1000)}))
                result = self.service.answer(
                    question, hits,
                    on_delta=delta,
                    cancel_event=cancel, history=history, request_id=identity, preferences=prefs,
                    raw_question=raw_question,
                    interview_packet=packet_for_answer.to_prompt_data(compact=prefs.provider=='ollama',
                        max_chars=1400 if prefs.provider=='ollama' else 4200) if packet_for_answer else None)
                result.diagnostics['retrieval_coverage']=coverage
                if coverage_warning and coverage_warning not in result.warnings:
                    result.warnings.append(coverage_warning)
                if not first_english and result.answer_en:
                    self.post('answer_stage', (identity, {
                        'answer_provider_ttft_ms': max(0., (time.perf_counter()-provider_started)*1000)}))
                self.post("answer", (identity, question, result))
            except Exception as exc:
                self.post("answer_error", (identity, str(exc), cancel.is_set()))
        self.background(work, "PrismAnswer")

    def _retrieve_question(self, question, packet, top_k, prefs=None, *, return_coverage=False):
        """Reserve evidence for each actual ask before adding broad context.

        Coverage means a local search candidate was selected, not that its
        scientific content answers the question. Missing/budget-limited asks
        stay explicit in the answer packet and content-free diagnostics.
        """
        top_k=max(0,min(8,int(top_k)))
        from .candidate_profile import retrieval_query
        prefs = prefs or self.pref
        from .evidence_gap import split_answer_subquestions
        # Use the same full-source checklist as the final answer request. The
        # UI-oriented packet can group several asks into one readable paragraph.
        subquestions=list(dict.fromkeys(split_answer_subquestions(question)))
        searched=subquestions[:24] if top_k else []
        expand=lambda q:retrieval_query(q,candidate_profile=prefs.candidate_profile,research_profile=prefs.research_profile)
        primary=expand(question)
        queries=[primary]+[expand(q) for q in searched]
        found={q:self.preparation.search(self.kb,q,top_k=top_k)
               for q in dict.fromkeys(queries) if top_k and q.strip()}
        whole=found.get(primary,[])
        groups=[found.get(expand(q),[]) for q in searched]
        hits=[];seen=set()
        key=lambda p:(p.source,p.location,p.text)
        def add(p):
            if len(hits)<top_k and key(p) not in seen:
                seen.add(key(p));hits.append(p)
        if len(subquestions)>1:
            # The old full-question top-(k-1) reservation starved later asks.
            for group in groups:
                if group:add(group[0])
        else:
            # Retain full-context definitions for a single/generic child ask.
            for p in whole[:max(1,top_k-1)]:add(p)
            for group in groups:
                if group:add(group[0])
        for p in whole:add(p)
        for rank in range(1,top_k):
            for group in groups:
                if rank<len(group):add(group[rank])
        covered=[index+1 for index,group in enumerate(groups) if any(key(p) in seen for p in group)]
        all_indices=list(range(1,len(subquestions)+1))
        coverage={
            'subquestion_count':len(subquestions),'searched_subquestion_count':len(searched),
            'covered_subquestion_indices':covered,
            'uncovered_subquestion_indices':[i for i in all_indices if i not in covered],
            'no_hit_subquestion_indices':[i+1 for i,g in enumerate(groups) if not g],
            'unsearched_subquestion_indices':all_indices[len(searched):],
            'passage_budget':top_k,'selected_passage_count':len(hits),
            'coverage_is_retrieval_only':True}
        if return_coverage:
            return hits, coverage
        self.last_retrieval_coverage = coverage
        return hits

    def cancel_answer(self):
        self._cancel_answer_prefetch()
        if self.busy:
            self._cancel.set()
            self.window.set_status("正在取消当前回答；待排队问题会在取消完成后继续。")

    def _finish_answer(self):
        self._question_raw_by_request.pop(self._active,None)
        self._question_packets.pop(self._active,None)
        self._answer_request_timing.pop(self._active, None)
        self._automatic_ids.discard(self._active)
        self._sealed_automatic_ids.discard(self._active)
        self.last_answer_timing['answer_total_ms'] = max(0.,
            (time.perf_counter()-self._answer_started)*1000)
        self.window.set_metrics(self.last_answer_timing)
        self.busy = False
        self._active = ""
        self._active_prefs = None
        self.window.set_busy(False)
        if not self._closed:
            QTimer.singleShot(0, self._next_answer)

    def _discard_unfinished_answer(self):
        self._stream = {"answer_en": "", "answer_zh": "", "question_zh": ""}
        self._stream_pairs = []
        self.window.set_answer(sources=[], streaming=False, question_en="",
                               request_id=self._active, cancelled=True)

    def _make_translator(self, cache, engine=None):
        if (engine or self.pref.translation_engine) == "qwen":
            from .translation_qwen import QwenCaptionTranslator
            return QwenCaptionTranslator(replace(self.pref))
        if (engine or self.pref.translation_engine) == "gemma":
            from .translation_gemma import GemmaTranslator
            return GemmaTranslator(cache)
        if (engine or self.pref.translation_engine) == "nllb":
            from .translation_nllb import NllbTranslator
            return NllbTranslator(Path(cache) / "translation-nllb")
        return ArgosTranslator(Path(cache) / "translation")

    def prepare_translation(self):
        self.save_settings(self.window.get_settings())
        cache = Path(self.pref.model_cache_dir)
        engine = self.pref.translation_engine
        def work():
            translator = self._make_translator(cache, engine)
            if not translator.ready:
                if engine == "nllb":
                    translator.install(on_status=lambda message: self.post("status_global", message))
                else:
                    translator.install_english_chinese(lambda message: self.post("status_global", message))
            # Load and validate, rather than equate a downloaded file with readiness.
            translator.translate("Could you explain your method?")
            label = "千问听译连接已验证" if engine == "qwen" else "本地英译中模型已就绪"
            self.post("status_global", label + "；正在收音时请停止并重新开始以载入。")
        self.background(work, "PrismPrepareTranslation")

    def prepare_speaker(self):
        self.save_settings(self.window.get_settings())
        cache = Path(self.pref.model_cache_dir)
        def work():
            from .speaker import prepare_speaker_model, OnlineSpeakerDiarizer
            prepare_speaker_model(cache, lambda message: self.post("status_global", message))
            checker = OnlineSpeakerDiarizer(cache)
            if not checker.prepare():
                raise RuntimeError(checker.failure)
            self.post("status_global", "本地说话人模型已就绪；停止并重新监听后启用匿名声纹分组。")
        self.background(work, "PrismPrepareSpeaker")

    def check_models(self):
        self.save_settings(self.window.get_settings())
        if self._model_check_running:
            return
        if self.busy:
            self.window.set_status("请在当前回答结束后检查或切换模型。", "warning")
            return
        prefs = replace(self.pref)
        if prefs.provider!='ollama' and not prefs.cloud_consent:
            self.window.set_status('尚未授权云端连接；可先使用本地听译。')
            return
        self._model_check_running=True
        def work():
            if prefs.provider == "codex":
                catalog = self.service.available_models(prefs)
                self.post("answer_models", catalog)
                names = [str(item.get("model") or item.get("id") or item.get("displayName")) for item in catalog]
                self.post("status_global", "Codex 订阅连接已预热；模型可用性以实际回答为准。目录：" + ", ".join(names[:5]))
            elif prefs.provider == "ollama":
                from urllib.request import urlopen
                with urlopen(prefs.ollama_endpoint.rstrip("/") + "/api/tags", timeout=5) as response:
                    models = json.load(response).get("models", [])
                self.post("models", [x["name"] for x in models])
                self.post("status_global", "Ollama 已连接。请确认所选模型已下载。")
            else:
                catalog = self.service.available_models(prefs)
                self.post("answer_models", catalog)
                self.post("status_global", "已读取服务商模型目录；这不代表免费额度或回答权限已经核验。")
        def checked():
            try:work()
            finally:self.post('model_check_done')
        self.background(checked, "PrismModelCheck")

    def _conversation_snapshot(self):
        return build_snapshot(self._saved_session_id, self._session_created_at,
                              self._transcripts, self._answers,
                              current_question_en=self._last_question_display)

    def set_auto_save_sessions(self, enabled):
        self.pref.auto_save_sessions = enabled is True
        self.pref.save()
        if hasattr(self.window, "set_auto_save"):
            self.window.set_auto_save(self.pref.auto_save_sessions)
        self.window.set_status("已开启本机自动保存；只保存确认字幕与完成回答，不保存录音。"
                               if self.pref.auto_save_sessions else
                               "已关闭自动保存；已有本机记录仍可在历史中查看。")

    def _auto_save_conversation(self):
        if not self._closed and self.pref.auto_save_sessions is True and self._session_dirty:
            self.save_conversation(automatic=True)

    def save_conversation(self, automatic=False):
        """Explicit local persistence, separate from provider/model state."""
        snapshot = self._conversation_snapshot()
        if not snapshot["transcripts"] and not snapshot["answers"]:
            if not automatic:
                self.window.set_status("尚无已确认对话或已完成回答可保存。", "warning")
            return None
        try:
            destination = self.session_store.save(snapshot)
        except (OSError, ValueError, TypeError) as exc:
            # No transcript or model request is copied into diagnostic output.
            self.window.set_status("本机会话保存失败：" + str(exc), "error")
            return None
        self._session_dirty = False
        if not automatic:
            self.window.set_status("已保存到本机会话历史；可查看、导出或删除。", "success")
        return destination

    def show_session_history(self):
        """Read-only archive; opening it never restores a capture/model session."""
        dialog = QDialog(self.window)
        dialog.setWindowTitle("Prism Interview · 本机会话历史")
        dialog.resize(1100, 760)
        layout = QVBoxLayout(dialog)
        note = QLabel("记录仅在本机。保存确认字幕与完成回答；查看历史不会监听、连接模型或继续回答。")
        note.setWordWrap(True)
        layout.addWidget(note)
        split = QSplitter(Qt.Orientation.Horizontal)
        sessions = QListWidget()
        sessions.setMinimumWidth(260)
        preview = QPlainTextEdit()
        preview.setReadOnly(True)
        preview.setPlaceholderText("选择左侧会话，查看完整的中英对话和回答。")
        split.addWidget(sessions)
        split.addWidget(preview)
        split.setSizes([300, 760])
        layout.addWidget(split, 1)
        location = QLabel("本机目录：" + str(self.session_store.root))
        location.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(location)
        controls = QHBoxLayout()
        export = QPushButton("导出选中记录…")
        remove = QPushButton("删除选中记录…")
        close = QPushButton("关闭")
        controls.addWidget(export)
        controls.addWidget(remove)
        controls.addStretch()
        controls.addWidget(close)
        layout.addLayout(controls)
        state = {"snapshot": None}

        def refresh():
            sessions.clear()
            for entry in self.session_store.list_sessions():
                title = entry["created_at"] + f'  ·  {entry["transcript_count"]}段 / {entry["answer_count"]}题'
                label = QListWidgetItem(title + "\n" + (entry["preview"][:85] or "会话记录"))
                label.setData(Qt.ItemDataRole.UserRole, entry["id"])
                sessions.addItem(label)
            if sessions.count():
                sessions.setCurrentRow(0)
            else:
                preview.setPlainText("尚无已保存会话。点击“保存对话”，或按需开启自动保存。")
                state["snapshot"] = None
                export.setEnabled(False)
                remove.setEnabled(False)

        def selected(current, previous=None):
            state["snapshot"] = None
            export.setEnabled(False)
            remove.setEnabled(False)
            if current is None:
                return
            try:
                snapshot = self.session_store.load(current.data(Qt.ItemDataRole.UserRole))
                preview.setPlainText(render_text(snapshot))
            except (OSError, ValueError, TypeError):
                preview.setPlainText("无法读取此记录。文件可能已移走、损坏或格式不受支持。")
                return
            state["snapshot"] = snapshot
            export.setEnabled(True)
            remove.setEnabled(True)

        def export_selected():
            snapshot = state["snapshot"]
            if snapshot is None:
                return
            filename, chosen_filter = QFileDialog.getSaveFileName(dialog, "导出本机会话记录",
                "Prism-" + snapshot["id"][:8] + ".md",
                "Markdown (*.md);;JSON conversation (*.json);;UTF-8 text (*.txt)")
            if not filename:
                return
            suffix = ".json" if "*.json" in chosen_filter else ".txt" if "*.txt" in chosen_filter else ".md"
            destination = Path(filename)
            if not destination.suffix:
                destination = destination.with_suffix(suffix)
            try:
                self.session_store.export(snapshot, destination)
                self.window.set_status("选中的历史会话已导出。", "success")
            except (OSError, ValueError, TypeError) as exc:
                QMessageBox.warning(dialog, "导出失败", str(exc))

        def delete_selected():
            snapshot = state["snapshot"]
            if snapshot is None:
                return
            choice = QMessageBox.question(dialog, "删除本机记录", "永久删除选中的本机会话记录？已导出的副本不受影响。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if choice != QMessageBox.StandardButton.Yes:
                return
            try:
                self.session_store.delete(snapshot["id"])
            except OSError as exc:
                QMessageBox.warning(dialog, "删除失败", str(exc))
            refresh()

        sessions.currentItemChanged.connect(selected)
        export.clicked.connect(export_selected)
        remove.clicked.connect(delete_selected)
        close.clicked.connect(dialog.accept)
        refresh()
        dialog.exec()

    def export_text(self):
        file, _ = QFileDialog.getSaveFileName(self.window, "手动导出本次会话", "Prism-session.txt", "UTF-8 text (*.txt)")
        if not file:
            return
        parts = ["Prism Interview · 用户手动导出", datetime.now().isoformat(timespec="seconds"), ""]
        for entry in self._transcripts:
            parts += [entry["at"] + " · " + entry.get("speaker_label", "说话人未确认"),
                      '整理英文：'+entry["en"], '中文：'+entry["zh"],
                      '识别原话：'+entry.get('raw_en',entry['en']), ""]
        for entry in self._answers:
            result = entry["result"]
            parts += ["QUESTION: " + entry["question"], result.question_zh, result.answer_en,
                      result.answer_zh, "\n".join(result.sources), ""]
        try:
            Path(file).write_text("\n".join(parts), encoding="utf-8")
            self.window.set_status("会话已导出至你指定的位置。", "success")
        except OSError as exc:
            self.window.set_status("导出失败：" + str(exc), "error")

    def _mark_capture_failure(self):
        self._capture_error_handled = True
        self._question_uncertain = True
        self._cancel_answer_prefetch()
        self._requests = deque(item for item in self._requests if item[0] not in self._automatic_ids)
        if self.busy and self._active in self._automatic_ids:
            self._cancel.set()
            self._discard_unfinished_answer()

    def pump(self):
        if self._closed:
            return
        error_event = getattr(self.audio, "error_event", None)
        if (error_event is not None and error_event.is_set() and not self._capture_error_handled
                and not self.stopping):
            self._mark_capture_failure()
            self.stop_listening(drain=True)
            self.window.set_status("输入设备已断开或收音失败，已停止采集；请检查两端设备后重新开始。", "error")
        for _ in range(250):
            try:
                kind, data = self.events.get_nowait()
            except queue.Empty:
                return
            if kind == "devices":
                if isinstance(data, dict):
                    if data.get("generation") != self._device_refresh_generation:
                        continue
                    source, devices = data["input_source"], data["devices"]
                else:
                    source, devices = "loopback", data
                selected_id = self.pref.microphone_id if source == "microphone" else self.pref.speaker_id
                self.window.set_devices(devices, selected_id, source=source)
                if source == "loopback" and not self.pref.speaker_id and devices:
                    self.pref.speaker_id = devices[0][0]
                if self.preparing or self.stopping:
                    self.window.start_button.setEnabled(False)
            elif kind == "asr_ready":
                self._start_capture(data)
            elif kind == 'capture_ready':
                if data==self._session and self.preparing and self._capture_started and not self.stopping:
                    self.listening,self.preparing=True,False
                    self.window.set_listening(True)
            elif kind == "translation_ready":
                if data == self._session:
                    self._translation_ready = True
                    self._start_capture(data)
            elif kind == 'restoration_ready':
                if data==self._session:
                    self._restoration_ready=True
                    self._start_capture(data)
            elif kind == 'preparation':
                self.window.set_preparation_status(str(data))
                self._show_question_packet()
            elif kind == 'grammar':
                if (not self.pref.grammar_refinement or not data.accepted
                        or self._grammar_versions.get(data.turn_id)!=data.revision
                        or self._caption_sources.get(data.turn_id)!=data.baseline):
                    continue
                _,display=self._restored(data.clean)
                row=next((r for r in reversed(self._transcripts) if r['id']==data.turn_id),None)
                if row:
                    row['en']=display
                    self._session_dirty = True
                    self.window.update_message(data.turn_id,display,row['zh'],raw_en=row.get('raw_en',data.raw),
                        translation_pending=row.get('translation_pending',False),
                        **self._caption_metadata.get(data.turn_id,{}))
                if self._live_data.get('entry_id')==data.turn_id and self._live_data.get('source')==data.baseline:
                    self._live_data.update(en=display,state='已成句 · AI 语法整理')
                self._publish_live()
            elif kind in ("voice", "level"):
                session = data if kind == "voice" else data[0]
                if session != self._session:
                    continue
                if kind == "voice":
                    self.assembler.voice()
                else:
                    self.window.set_metrics({"audio_level": data[1]})
            elif kind == "transcript":
                session, subtype, english = data[:3]
                context = data[3] if len(data) > 3 else None
                if session == self._session:
                    self.on_transcript(subtype, english, context)
            elif kind == "translation":
                session, key, zh = data
                if (self._listen_pref.translation_engine == "qwen"
                        and not self.pref.translation_cloud_consent):
                    continue
                identity, source = key if isinstance(key, tuple) else (key, None)
                if session == self._session:
                    current = self._caption_sources.get(identity)
                    if (not isinstance(source, str) or not str(zh).strip()
                            or not self._caption_translation_prefix(source, current)):
                        continue
                    watermark = self._caption_translation_sources.get((session, identity))
                    if (watermark and self._caption_translation_prefix(watermark, current)
                            and len(source) <= len(watermark)):
                        continue
                    # A slower MT response may still cover a genuine confirmed
                    # prefix. Show that progress without claiming that the later
                    # confirmed suffix has been translated, and never regress
                    # to an older/shorter response after a longer accepted one.
                    pending = source != current
                    self._caption_translation_sources[(session, identity)] = source
                    if pending:
                        self.window.update_translation(identity, zh, translation_pending=True)
                    else:
                        self.window.update_translation(identity, zh)
                    for entry in self._transcripts:
                        if entry["id"] == identity:
                            entry["zh"] = zh
                            entry["translation_pending"] = pending
                            self._session_dirty = True
                            break
                    live=self._live_data
                    if (live.get('entry_id')==identity
                            and self._caption_translation_prefix(source, live.get('source',''))):
                        live['zh']=zh
                        if not pending and live.get('source')==source:
                            live['state']='已成句'
                    self._publish_live()
            elif kind == "translation_preview":
                session,identity,source,zh=data
                live=self._live_data
                if (self.pref.live_translation and session==self._session and identity==('live',self._phase_epoch,self._live_token)
                        and live.get('source','').startswith(source)):
                    live['zh']=zh
                    self._publish_live()
            elif kind == "capture_drained":
                self._finish_listening_drain(data)
            elif kind in ("listening_drained", "drain_failed"):
                session, result = data
                if session == self._session:
                    self.stop_listening(drain=False)
                    if kind == "drain_failed":
                        self.window.set_status(result, "warning")
                    elif not result:
                        self.window.set_status("停止收音时翻译排空超时，请核对最后一条文字。", "warning")
            elif kind == "status":
                session, message = data
                if session != self._session:
                    continue
                failure = message.startswith(("Whisper 加载失败：", "Whisper 处理失败：", "音频设备错误：", "翻译模型准备失败："))
                if failure:
                    if message.startswith("音频设备错误："):
                        self._mark_capture_failure()
                    self.stop_listening()
                self.window.set_status(message, "error" if failure else "info")
            elif kind in ("status_global", "error"):
                self.window.set_status(str(data), "error" if kind == "error" else "info")
            elif kind == "metrics":
                metrics = dict(data)
                if "asr_seconds" in metrics:
                    metrics["asr_ms"] = metrics["asr_seconds"] * 1000
                self.window.set_metrics(metrics)
            elif kind == "documents":
                self.refresh_docs()
                count, issues = data
                self.window.set_status(f"本地资料索引已更新：{len(self.kb.document_names)}份，共{len(self.kb.passages)}段。" + ("；" + "; ".join(issues[:2]) if issues else ""), "warning" if issues else "success")
            elif kind == "models":
                self.window.set_models(data)
            elif kind == "answer_models":
                self.window.set_answer_models(data)
            elif kind == 'model_check_done':
                self._model_check_running=False
            elif kind == 'answer_prefetch_event':
                identity, forwarded_kind, payload = data
                slot = self._answer_prefetch
                if not slot or slot['id'] != identity:
                    continue
                terminal = forwarded_kind in ('answer', 'answer_error')
                if terminal:
                    slot['finished'] = True
                if not slot.get('promoted') and slot['scope'] != self._answer_prefetch_scope():
                    self._cancel_answer_prefetch()
                if slot['cancel'].is_set() and not slot.get('promoted'):
                    if terminal:
                        self._answer_prefetch = None
                    continue
                if slot.get('promoted'):
                    # Re-enter the existing identity, pair, citation and
                    # cancellation validation paths; never render here.
                    self.post(forwarded_kind, payload)
                    if terminal:
                        self._answer_prefetch = None
                elif forwarded_kind == 'answer_error':
                    # A speculative failure is invisible. The real endpoint
                    # may still make its ordinary fresh answer request.
                    self._cancel_answer_prefetch()
                else:
                    if forwarded_kind == 'answer_delta':
                        slot['bytes'] += len(str(payload[2]).encode('utf-8'))
                    if slot['bytes'] > 262144:
                        self._cancel_answer_prefetch()
                    else:
                        slot['events'].append((forwarded_kind, payload))
            elif kind == 'answer_stage':
                identity, metrics = data
                if identity == self._active and not self._cancel.is_set():
                    # Numeric stage timings only: no speech, document excerpts,
                    # prompts, provider credentials or personal facts.
                    self.last_answer_timing.update(metrics)
                    self.window.set_metrics(metrics)
            elif kind == "answer_delta":
                identity, field, delta = data
                if identity != self._active or self._cancel.is_set():
                    continue
                if field == "answer_pair":
                    try:
                        pair = json.loads(delta)
                        if (not isinstance(pair, dict) or not all(isinstance(pair.get(key), str) and pair[key].strip()
                                for key in ('id', 'en', 'zh')) or not pair.get('en_complete') or not pair.get('zh_complete')):
                            continue
                    except (ValueError, TypeError):
                        continue
                    if any(item['id'] == pair['id'] for item in self._stream_pairs):
                        continue
                    self._stream_pairs.append(pair)
                    self._stream['answer_en'] = '\n\n'.join(item['en'] for item in self._stream_pairs)
                    self._stream['answer_zh'] = '\n\n'.join(item['zh'] for item in self._stream_pairs)
                if field in self._stream:
                    self._stream[field] += delta
                if field == 'question_zh' and delta:
                    self.window.set_current_question_translation(self._active_question_en,
                                                                 self._stream['question_zh'])
                if field in ("answer_en", "answer_pair") and delta and not self._first_answer:
                    self._first_answer = True
                    self.window.set_metrics({"answer_ms": (time.perf_counter()-self._answer_started)*1000})
                self.window.set_answer(self._stream["answer_en"], self._stream["answer_zh"],
                                       self._stream["question_zh"], streaming=True, request_id=identity,
                                       answer_pairs=self._stream_pairs or None)
            elif kind == "answer":
                identity, question, result = data
                if identity != self._active:
                    continue
                if not self._cancel.is_set():
                    guarded_gap = bool(result.diagnostics.get('guarded_evidence_gap'))
                    self.window.set_answer_model(('资料不足提示 · ' if guarded_gap else '') + result.model)
                    self.window.set_answer(result.answer_en, result.answer_zh, result.question_zh,
                                           sources=result.evidence, streaming=False, question_en=question,
                                           question_preview_en=(self._question_packets[identity].answer_question_en
                                               if identity in self._question_packets else question),
                                           request_id=identity,
                                           answer_pairs=result.pairs or None,
                                           question_raw_en=self._question_raw_by_request.get(identity, question))
                    self._history.append({"question": question, "answer_en": result.answer_en})
                    self._answers.append({"question": question, "result": result,
                        "question_raw_en": self._question_raw_by_request.get(identity, question),
                        "at": utc_now()})
                    self._session_dirty = True
                    label = '本题个人资料不足 · 受约束提示' if guarded_gap else '回答已生成'
                    self.window.set_status(f"{label} · {result.provider} / {result.model}" + (" · " + "; ".join(result.warnings) if result.warnings else ""), "warning" if guarded_gap else "success")
                else:
                    self._discard_unfinished_answer()
                    self.window.set_status("当前回答已取消。")
                self._finish_answer()
            elif kind == "answer_error":
                identity, message, cancelled = data
                if identity != self._active:
                    continue
                self._discard_unfinished_answer()
                self.window.set_status("回答已取消。" if cancelled else "回答失败：" + message,
                                       "info" if cancelled else "error")
                self._finish_answer()

    def close(self):
        if self._closed:
            return
        self.session_save_timer.stop()
        if self.pref.auto_save_sessions is True and self._session_dirty:
            self.save_conversation(automatic=True)
        self._cancel_answer_prefetch()
        self._closed = True
        self.grammar.close()
        self.timer.stop()
        self.auto_timer.stop()
        self._requests.clear()
        self._cancel.set()
        self.stop_listening(drain=False)
        self.pref.save()
        # The child is isolated to this app, not the user's existing Codex app.
        self.background(self.service.close, "PrismCloseModel")


def main():
    load_runtime()
    parser = argparse.ArgumentParser(description="Prism Interview")
    parser.add_argument("--smoke", help="Render a real empty application and close; no capture/inference")
    parser.add_argument("--self-check", help="Write a dependency/device preflight JSON; no inference")
    parser.add_argument("--verify-runtime", metavar="TEST_WAV", help="Explicit offline ASR/translation/PDF verification; no capture or cloud")
    parser.add_argument("--verification-report", help="Destination JSON for --verify-runtime")
    parser.add_argument("--verification-translation-engine", choices=["argos", "gemma"], default="argos")
    parser.add_argument("--video-demo", action="store_true", help="Start local listening in manual-answer mode for a user-controlled video")
    parser.add_argument("--choose-model", action="store_true", help="Open the answer model chooser after startup; no answer request")
    parser.add_argument("--verification-speaker", action="store_true", help="Include existing local speaker model in offline runtime verification")
    parser.add_argument("--verification-live", action="store_true", help="Verify already installed streaming and punctuation models offline")
    parser.add_argument('--prepare-facts', metavar='FACT_JSON', help='Rebind and validate portable fact cards against the current local index; no UI or cloud')
    parser.add_argument('--facts-report', metavar='REPORT_JSON', help='Destination JSON for --prepare-facts')
    args, qt_args = parser.parse_known_args()
    if args.prepare_facts:
        if not args.facts_report:
            parser.error('--prepare-facts requires --facts-report')
        from .research_facts import rebind_fact_library
        report = rebind_fact_library(Path(args.prepare_facts), KnowledgeBase(app_dir() / 'knowledge.json'))
        destination = Path(args.facts_report)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        return 0 if report['ok'] else 2
    if args.verify_runtime:
        if not args.verification_report:
            parser.error("--verify-runtime requires --verification-report")
        from .diagnostics import verify_runtime
        if args.verification_speaker or args.verification_live:
            return verify_runtime(args.verify_runtime, args.verification_report,
                                  args.verification_translation_engine, verify_speaker=args.verification_speaker,
                                  verify_live=args.verification_live)
        if args.verification_translation_engine == "gemma":
            return verify_runtime(args.verify_runtime, args.verification_report, "gemma")
        return verify_runtime(args.verify_runtime, args.verification_report)
    if args.self_check:
        from .runtime import configure_cuda_dlls
        configure_cuda_dlls()
        import ctranslate2, faster_whisper, soundcard, numpy
        import PySide6
        report = {"numpy": numpy.__version__, "qt": PySide6.__version__,
                  "ctranslate2": ctranslate2.__version__, "faster_whisper": faster_whisper.__version__,
                  "cuda_devices": ctranslate2.get_cuda_device_count(),
                  "loopback_count": len(speaker_loopbacks()), "capture_started": False,
                  "model_requests": 0}
        Path(args.self_check).write_text(json.dumps(report, indent=2), encoding="utf-8")
        if not args.smoke:
            return 0
    app = QApplication.instance() or QApplication([sys.argv[0], *qt_args])
    app.setApplicationName("Prism Interview")
    app.setFont(QFont(FONT, 10))
    from .instance_guard import InstanceGuard, interactive_instance_required
    instance_guard = None
    if interactive_instance_required(args):
        instance_guard = InstanceGuard(app_dir(), app)
        try:
            if not instance_guard.acquire_or_notify("choose-model" if args.choose_model else "activate"):
                instance_guard.close()
                return 0
        except RuntimeError as exc:
            instance_guard.close()
            QMessageBox.warning(None, "Prism 已在运行", str(exc))
            return 1
    try:
        return _run_interactive_window(app, args, instance_guard)
    finally:
        # Include window/controller construction in the lock lifetime. A
        # startup exception may display an error dialog before process exit.
        if instance_guard is not None:
            instance_guard.close()


def _run_interactive_window(app, args, instance_guard=None):
    window = CopilotWindow()
    from . import __version__
    demo_mode = args.video_demo and not (args.smoke or args.self_check)
    if demo_mode:
        import soundcard as sc
        demo_preferences = Preferences.load()
        demo_preferences.auto_answer = False
        demo_preferences.interview_phase = "presentation"
        default_speaker = sc.default_speaker()
        if default_speaker is not None:
            demo_preferences.speaker_id = default_speaker.id
        controller = CopilotController(window, demo_preferences)
        window.setWindowTitle(f"Prism {__version__} · 视频听译体验")
    else:
        controller = CopilotController(window)
        window.setWindowTitle(f"Prism Interview {__version__}")
    app.aboutToQuit.connect(controller.close)
    if instance_guard is not None:
        instance_guard.bind_window(window, controller.choose_answer_model)
    window.show()
    if args.choose_model and not (args.smoke or args.self_check):
        QTimer.singleShot(900, controller.choose_answer_model)
    if (not demo_mode and not (args.smoke or args.self_check)
            and controller.pref.provider == "codex" and controller.pref.cloud_consent):
        # Establish the official connection while the user prepares/presents.
        # Catalog loading consumes no answer turn and sends no paper content.
        QTimer.singleShot(700, controller.check_models)
    if demo_mode:
        # The HTML player opens paused. Capture starts only for this explicit
        # demo flag; answers remain user-triggered and consume no cloud turn
        # until the user asks to generate one.
        def start_demo_capture(attempt=0):
            # Device enumeration runs off the UI thread. Do not overwrite the
            # requested default with the combobox's initially empty value.
            if window.get_settings().get("speaker_id"):
                controller.start_listening()
            elif attempt < 40:
                QTimer.singleShot(250, lambda: start_demo_capture(attempt + 1))
            else:
                window.set_status("未能自动选择输出设备；请选好耳机／扬声器后点击开始监听。", "warning")
        QTimer.singleShot(400, start_demo_capture)
        if controller.pref.provider == "codex" and controller.pref.cloud_consent:
            QTimer.singleShot(700, controller.check_models)
    if args.smoke:
        def screenshot():
            Path(args.smoke).parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(args.smoke)
            app.quit()
        QTimer.singleShot(1000, screenshot)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

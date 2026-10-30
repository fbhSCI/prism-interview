"""App preferences and application storage locations."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import math
import os
from pathlib import Path


APP_NAME = "PrismInterview"


def app_dir() -> Path:
    override = os.getenv("PRISM_INTERVIEW_HOME")
    if override:
        location = Path(override).expanduser().resolve()
        location.mkdir(parents=True, exist_ok=True)
        return location
    if os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA") or (Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.getenv("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))
    location = base / APP_NAME
    location.mkdir(parents=True, exist_ok=True)
    return location


@dataclass
class Preferences:
    speaker_id: str = ""
    # Keep independent choices so changing source never substitutes one device
    # class for another or forgets the user's selected headset.
    input_source: str = "loopback"
    microphone_id: str = ""
    whisper_size: str = "large-v3-turbo"
    asr_device: str = "auto"
    compute_type: str = "int8_float16"
    translation_engine: str = "qwen"
    # Independent of answer consent: continuous confirmed speech is a separate
    # data flow and must not inherit permission from an answer provider switch.
    translation_cloud_consent: bool = False
    translation_qwen_model: str = "qwen3.8-flash"
    translation_qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    translation_timeout: float = 20.0
    model_cache_dir: str = ""
    ollama_model: str = "qwen2.5:7b"
    ollama_endpoint: str = "http://127.0.0.1:11434"
    provider: str = "qwen"
    answer_model: str = ""
    answer_speed: str = "fast"
    cloud_consent: bool = False
    share_evidence: bool = True
    api_base_url: str = ""
    # Session-only. Never serialize a credential to preferences.json.
    api_key: str = field(default="", repr=False, compare=False)
    # A separate in-memory key per provider prevents a dialog switch from sending
    # a DeepSeek key to Qwen (or vice versa). Neither field is persisted.
    session_api_keys: dict[str, str] = field(default_factory=dict, repr=False, compare=False)
    qwen_region: str = "cn-beijing"
    free_quota_confirmed: bool = False
    # Only Qwen may use a separately authorized paid quota; other providers keep
    # their existing free-quota gate. No API credential is persisted here.
    qwen_paid_usage_authorized: bool = False
    codex_executable: str = ""
    answer_timeout: float = 90.0
    # Optional domain/role context; never prefill achievements or a research specialty.
    research_profile: str = ""
    # User-confirmed personal facts, independent of a paper's authorship.
    candidate_profile: str = ""
    prefer_prepared_answers: bool = True
    num_ctx: int = 4096
    max_answer_words: int = 220
    brief_answer: bool = False
    speech_threshold: float = 0.007
    # Confirm stable cards sooner; the independent streaming draft is optional.
    # The measured 3.6 s preset preserved speaker grouping better than 3.2 s,
    # while removing the former 6 s confirmation wait. 3.2 s remains optional.
    caption_window_seconds: float = 3.6
    # Faster confirmation is opt-in, and applies only on actual CUDA with a
    # single remote speaker and the standard 3.6 s context. New/legacy settings
    # retain original cadence; explicit valid choices are preserved.
    caption_update_mode: str = "original"
    speaker_diarization: bool = True
    # An explicit source assumption for a two-person call, never a voiceprint
    # identity claim. The loopback contains the selected output, not the mic.
    caption_speaker_mode: str = "single_remote"
    auto_save_sessions: bool = False
    follow_latest: bool = True
    streaming_captions: bool = True
    live_translation: bool = False
    clean_speech: bool = True
    grammar_refinement: bool = False
    auto_answer: bool = True
    # Capture already waits for 1.1 s of VAD silence before final recognition.
    # Do not stack another multi-second endpoint delay after that final.
    auto_endpoint_seconds: float = 1.0
    interview_phase: str = "qa"
    topmost: bool = False
    top_k: int = 5

    def __post_init__(self):
        # Provider-specific defaults must not be attached to a user who chose
        # Codex, DeepSeek, Ollama or a custom endpoint instead.
        if self.provider == "qwen":
            if not self.answer_model:
                self.answer_model = "qwen3.8-flash"
            if not self.api_base_url:
                self.api_base_url = {
                    "cn-beijing": "https://dashscope.aliyuncs.com/compatible-mode/v1",
                    "ap-southeast-1": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
                }.get(self.qwen_region, "")
        if self.input_source not in ("loopback", "microphone", "both"):
            self.input_source = "loopback"
        for name in ("speaker_id", "microphone_id"):
            if not isinstance(getattr(self, name), str):
                setattr(self, name, "")
        # Only literal booleans may authorize persistence, sharing or cloud billing.
        # Strings such as "false"/"yes" and numeric JSON values are not consent.
        for name in ("auto_save_sessions", "cloud_consent", "translation_cloud_consent",
                     "free_quota_confirmed", "qwen_paid_usage_authorized", "share_evidence"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                setattr(self, name, False)
        if self.translation_engine not in ("argos", "nllb", "gemma", "qwen"):
            self.translation_engine = "argos"
        if not isinstance(self.translation_cloud_consent, bool):
            self.translation_cloud_consent = False
        self.translation_qwen_model = str(self.translation_qwen_model or "qwen3.8-flash").strip() or "qwen3.8-flash"
        self.translation_qwen_base_url = str(self.translation_qwen_base_url or "https://dashscope.aliyuncs.com/compatible-mode/v1").strip().rstrip("/")
        try:
            timeout = float(self.translation_timeout)
            self.translation_timeout = max(3.0, min(60.0, timeout)) if math.isfinite(timeout) else 20.0
        except (TypeError, ValueError, OverflowError):
            self.translation_timeout = 20.0
        if self.interview_phase not in ("presentation", "qa"):
            self.interview_phase = "qa"
        if self.answer_speed not in ("standard", "fast"):
            self.answer_speed = "standard"
        if self.caption_speaker_mode not in ("acoustic", "single_remote"):
            self.caption_speaker_mode = "single_remote"
        if self.caption_update_mode not in ("responsive", "original"):
            self.caption_update_mode = "original"
        try:
            self.auto_endpoint_seconds = max(.6, min(2.5, float(self.auto_endpoint_seconds)))
        except (TypeError, ValueError):
            self.auto_endpoint_seconds = 1.0
        try:
            self.max_answer_words = max(60, min(360, int(self.max_answer_words)))
        except (TypeError, ValueError, OverflowError):
            self.max_answer_words = 220

    @classmethod
    def load(cls) -> "Preferences":
        file = app_dir() / "preferences.json"
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return cls()
            # Prior releases used local translation when this field was absent.
            # Recommend Qwen for fresh installs without silently changing an
            # existing partial settings file into a cloud workflow.
            data.setdefault("translation_engine", "argos")
            allowed = {item.name for item in fields(cls)} - {"api_key", "session_api_keys"}
            return cls(**{key: val for key, val in data.items() if key in allowed})
        except (OSError, ValueError, TypeError):
            return cls()

    def save(self) -> None:
        dest = app_dir() / "preferences.json"
        temp = dest.with_suffix(".json.tmp")
        data = asdict(self)
        data.pop("api_key", None)
        data.pop("session_api_keys", None)
        temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        temp.replace(dest)

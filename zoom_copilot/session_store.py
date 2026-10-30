"""Explicit, local-only conversation snapshots; never audio or app credentials.

The on-disk format deliberately whitelists visible conversation fields. It is
not a dump of controller state, preferences, model requests or diagnostic logs.
"""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PureWindowsPath
import re
import tempfile
import uuid


SCHEMA_VERSION = 1
MAX_SESSION_BYTES = 32 * 1024 * 1024
_SESSION_ID = re.compile(r"^[a-f0-9]{32}$")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_session_id() -> str:
    return uuid.uuid4().hex


def _text(value) -> str:
    return value if isinstance(value, str) else ""


def _row(value) -> dict:
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    return value if isinstance(value, dict) else {}


def _texts(value) -> list[str]:
    return [v for v in value if isinstance(v, str)] if isinstance(value, (list, tuple)) else []


def _filename(value) -> str:
    # Portable exports need provenance, not the source machine's home path.
    return PureWindowsPath(_text(value)).name


def build_snapshot(session_id: str, created_at: str, transcripts=(), answers=(),
                   *, current_question_en: str = "", updated_at: str | None = None) -> dict:
    """Copy a strict allowlist; model settings and arbitrary nested fields vanish."""
    if not _SESSION_ID.fullmatch(session_id):
        raise ValueError("Invalid conversation identifier")
    captions = []
    for value in transcripts:
        item = _row(value)
        if not (_text(item.get("en")).strip() or _text(item.get("zh")).strip()):
            continue
        caption = {key: _text(item.get(key)) for key in
                   ("id", "at", "en", "raw_en", "zh", "turn_id", "speaker_label", "speaker_status")}
        caption["raw_en"] = caption["raw_en"] or caption["en"]
        sid = item.get("speaker_id")
        caption["speaker_id"] = sid if isinstance(sid, (str, int)) and not isinstance(sid, bool) else None
        caption["translation_pending"] = item.get("translation_pending") is True
        captions.append(caption)
    completed = []
    for value in answers:
        item = _row(value)
        result = _row(item.get("result", item))
        evidence = []
        for source in result.get("evidence", []) if isinstance(result.get("evidence"), (list, tuple)) else []:
            source = _row(source)
            evidence.append({"id": _text(source.get("id")), "source": _filename(source.get("source")),
                             "location": _text(source.get("location")), "text": _text(source.get("text"))})
        pairs = []
        for pair in result.get("pairs", []) if isinstance(result.get("pairs"), (list, tuple)) else []:
            pair = _row(pair)
            if _text(pair.get("en")).strip() or _text(pair.get("zh")).strip():
                pairs.append({key: _text(pair.get(key)) for key in ("id", "en", "zh")})
        answer = {key: _text(result.get(key)) for key in
                  ("question_zh", "answer_en", "answer_zh", "request_id", "provider", "model")}
        answer.update(question_en=_text(item.get("question", item.get("question_en"))),
                      question_raw_en=_text(item.get("question_raw_en")),
                      at=_text(item.get("at")), evidence=evidence, pairs=pairs,
                      sources=_texts(result.get("sources")), warnings=_texts(result.get("warnings")))
        if answer["answer_en"].strip() or answer["answer_zh"].strip():
            completed.append(answer)
    return {"schema_version": SCHEMA_VERSION, "application": "Prism Interview", "id": session_id,
            "created_at": _text(created_at), "updated_at": updated_at or utc_now(),
            "current_question_en": _text(current_question_en),
            "transcripts": captions, "answers": completed}


def validate_snapshot(value) -> dict:
    """Reject foreign formats and normalize loaded files through the allowlist."""
    if (not isinstance(value, dict) or type(value.get("schema_version")) is not int
            or value.get("schema_version") != SCHEMA_VERSION):
        raise ValueError("Unsupported conversation format")
    if not isinstance(value.get("transcripts"), list) or not isinstance(value.get("answers"), list):
        raise ValueError("Invalid conversation records")
    return build_snapshot(_text(value.get("id")), _text(value.get("created_at")),
                          value["transcripts"], value["answers"],
                          current_question_en=_text(value.get("current_question_en")),
                          updated_at=_text(value.get("updated_at")))


def render_text(snapshot: dict, *, markdown: bool = False) -> str:
    """A complete human-readable transcript, without HTML or abbreviation."""
    data = validate_snapshot(snapshot)
    h = "## " if markdown else ""
    parts = [("# " if markdown else "") + "Prism Interview · 本机会话记录",
             "开始时间：" + data["created_at"], "保存时间：" + data["updated_at"],
             "已确认字幕与已完成回答；不含录音。AI 建议请自行核对。", ""]
    if data["transcripts"]:
        parts += [h + "双语对话", ""]
    for item in data["transcripts"]:
        parts += [f'{item["at"]} · {item["speaker_label"] or "发言者"}',
                  "EN: " + item["en"], "中文：" + item["zh"]]
        if item["translation_pending"]:
            parts += ["（本段中文尚未全部生成）"]
        if item["raw_en"] != item["en"]:
            parts += ["识别原话：" + item["raw_en"]]
        parts += [""]
    if data["current_question_en"]:
        parts += [h + "保存时的当前问题（可能尚未完成）", data["current_question_en"], ""]
    for index, item in enumerate(data["answers"], 1):
        parts += [h + f"回答建议 {index}", "问题 EN: " + item["question_en"],
                  "问题中文：" + item["question_zh"],
                  f'模型：{item["provider"]} / {item["model"]}', ""]
        if item["pairs"]:
            for pair in item["pairs"]:
                parts += [pair["en"], pair["zh"], ""]
        else:
            parts += [item["answer_en"], item["answer_zh"], ""]
        if item["question_raw_en"] and item["question_raw_en"] != item["question_en"]:
            parts += ["问题原话：" + item["question_raw_en"], ""]
        for evidence in item["evidence"]:
            parts += [f'出处：{evidence["source"]} · {evidence["location"]}', evidence["text"], ""]
        if not item["evidence"] and item["sources"]:
            parts += item["sources"] + [""]
        if item["warnings"]:
            parts += ["提示：" + "；".join(item["warnings"]), ""]
    return "\n".join(parts)


class SessionStore:
    """One atomic JSON per local session. No indexing service or network calls."""

    def __init__(self, root: Path):
        self.root = Path(root).expanduser().resolve()

    def _path(self, identity: str) -> Path:
        if not _SESSION_ID.fullmatch(identity):
            raise ValueError("Invalid conversation identifier")
        candidate = self.root / (identity + ".json")
        if candidate.resolve().parent != self.root:
            raise ValueError("Conversation path is outside its local directory")
        return candidate

    @staticmethod
    def _atomic_write(destination: Path, text: str) -> None:
        if len(text.encode("utf-8")) > MAX_SESSION_BYTES:
            raise ValueError("Conversation exceeds the 32 MB local file limit")
        destination.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".session-", suffix=".tmp", dir=destination.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def save(self, snapshot: dict) -> Path:
        data = validate_snapshot(snapshot)
        path = self._path(data["id"])
        self._atomic_write(path, json.dumps(data, ensure_ascii=False, indent=2))
        return path

    def load(self, identity: str) -> dict:
        path = self._path(identity)
        if path.stat().st_size > MAX_SESSION_BYTES:
            raise ValueError("Conversation exceeds the 32 MB local file limit")
        data = validate_snapshot(json.loads(path.read_text(encoding="utf-8")))
        if data["id"] != identity:
            raise ValueError("Conversation identifier does not match its file")
        return data

    def list_sessions(self) -> list[dict]:
        if not self.root.exists():
            return []
        rows = []
        for path in self.root.glob("*.json"):
            try:
                data = self.load(path.stem)
            except (OSError, ValueError, TypeError):
                continue
            preview = next((row["en"] for row in data["transcripts"] if row["en"]), "")
            if not preview and data["answers"]:
                preview = data["answers"][0]["question_en"]
            rows.append({"id": data["id"], "created_at": data["created_at"],
                         "updated_at": data["updated_at"], "preview": preview,
                         "transcript_count": len(data["transcripts"]), "answer_count": len(data["answers"])})
        return sorted(rows, key=lambda row: row["updated_at"], reverse=True)

    def delete(self, identity: str) -> None:
        self._path(identity).unlink()

    def export(self, snapshot: dict, destination: Path) -> Path:
        destination = Path(destination)
        extension = destination.suffix.lower()
        if extension == ".json":
            text = json.dumps(validate_snapshot(snapshot), ensure_ascii=False, indent=2)
        elif extension in (".txt", ".md"):
            text = render_text(snapshot, markdown=extension == ".md")
        else:
            raise ValueError("Choose a JSON, Markdown or TXT file")
        self._atomic_write(destination, text)
        return destination

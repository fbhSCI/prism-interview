"""Official Codex app-server stdio integration; credentials stay inside Codex.

Protocol: https://learn.chatgpt.com/docs/app-server
No private HTTP endpoints, token files, or desktop chat automation are used.
"""
from __future__ import annotations

import json
from collections import deque
import math
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Callable


class AnswerCancelled(RuntimeError):
    pass


class CodexError(RuntimeError):
    pass


def require_included_quota(payload: dict) -> None:
    """Fail closed unless the official read confirms ordinary included usage.

    Credit balances and earned reset credits never grant permission here. This
    is a fresh preflight, not a reservation or an account-wide spending switch;
    another client or the current turn can still consume the remaining quota.
    Missing/null fields are unknown, never an inferred zero-percent usage.
    """
    unknown = ("无法确认 Codex 订阅内额度；为避免动用已购 credits，本次未发送问题。"
               "请检查 Codex 额度或更新 CLI 后重试，或切换本地模式。")
    exhausted = ("Codex 订阅内额度当前不可用或已到限；本次未发送问题。"
                 "本程序不会为继续回答而购买 credits 或兑换额度重置次数；"
                 "请等待额度恢复，或切换本地模式。")
    if not isinstance(payload, dict):
        raise CodexError(unknown)
    allowed = payload.get("ordinaryUsageAllowed")
    if allowed is False:
        raise CodexError(exhausted)
    buckets = payload.get("rateLimitsByLimitId")
    if buckets is not None and not isinstance(buckets, dict):
        raise CodexError(unknown)
    if buckets:
        snapshot = buckets.get("codex")
    else:
        snapshot = payload.get("rateLimits")
    if not isinstance(snapshot, dict) or snapshot.get("limitId") not in (None, "codex"):
        raise CodexError(unknown)
    # A backend reached-state remains authoritative even past a displayed reset
    # time. Never infer recovery merely because a timestamp has elapsed.
    if snapshot.get("rateLimitReachedType") is not None or snapshot.get("spendControlReached") is True:
        raise CodexError(exhausted)
    known_windows = 0
    for name in ("primary", "secondary"):
        if name not in snapshot:
            raise CodexError(unknown)
        window = snapshot[name]
        if window is None:
            continue
        if not isinstance(window, dict):
            raise CodexError(unknown)
        percent = window.get("usedPercent")
        if isinstance(percent, bool) or not isinstance(percent, (int, float)) or not math.isfinite(percent) or percent < 0:
            raise CodexError(unknown)
        if percent >= 100:
            raise CodexError(exhausted)
        known_windows += 1
    # This backend flag is account-validated in the installed CLI schema. Its
    # absence/null does not establish included availability, even with low usage.
    if allowed is not True or not known_windows:
        raise CodexError(unknown)


def find_codex(explicit: str = "") -> str:
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        raise CodexError("配置的 Codex CLI 路径不存在。")
    found = shutil.which("codex.exe") or shutil.which("codex")
    if found:
        return found
    # The Windows app bundles the official CLI; inspect executable names only.
    base = Path(os.getenv("LOCALAPPDATA", "")) / "OpenAI" / "Codex" / "bin"
    if base.is_dir():
        candidates = sorted(base.glob("*/codex.exe"), key=lambda p: p.stat().st_mtime, reverse=True)
        if candidates:
            return str(candidates[0])
    raise CodexError("未找到官方 Codex CLI。安装 Codex 并运行 codex login，或切换本地 Ollama。")


class CodexClient:
    """One isolated child server; one ephemeral text-only thread per answer.

    The CLI owns authentication. Only ChatGPT-authenticated accounts are accepted,
    preventing an accidental switch to separately billed API-key usage.
    """

    def __init__(self, executable: str = "", model: str = "", timeout: float = 90,
                 service_tier: str = "default"):
        if service_tier not in ("default", "priority"):
            raise CodexError("未知 Codex 速度档位；只支持标准 default 或经目录确认的 Fast priority。")
        self.executable = executable
        self.model = model
        self.timeout = timeout
        self.service_tier = service_tier
        self._process = None
        self._messages: queue.Queue = queue.Queue()
        self._pending_events = deque()
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._next_id = 0
        self._workdir = None
        self._model_catalog: list[dict] = []
        self._tool_config: dict = {}
        self.last_model = ""
        self.last_effort = ""
        self.last_service_tier = ""
        self.last_usage: dict = {}
        self.last_timings: dict = {}

    def close(self) -> None:
        process, self._process = self._process, None
        if process is not None:
            try:
                if process.stdin:
                    process.stdin.close()
                process.terminate()
                process.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                    process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            for pipe in (process.stdout, process.stdin):
                try:
                    if pipe:
                        pipe.close()
                except OSError:
                    pass
        if self._workdir is not None:
            self._workdir.cleanup()
            self._workdir = None

    def _send(self, message: dict) -> None:
        with self._write_lock:
            if self._process is None or self._process.poll() is not None:
                raise CodexError("Codex 服务已退出；请重试或检查登录。")
            try:
                self._process.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise CodexError("Codex 连接已断开；可重试。") from exc

    def _next(self, deadline: float, cancel_event=None, pending: bool = True) -> dict:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise AnswerCancelled("已取消回答。")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Codex 回答超时；请重试、缩短问题或切换本地模型。")
            if pending and self._pending_events:
                return self._pending_events.popleft()
            try:
                message = self._messages.get(timeout=min(0.1, remaining))
            except queue.Empty:
                if self._process is None or self._process.poll() is not None:
                    raise CodexError("Codex 服务意外退出；请重试。")
                continue
            if message is None:
                raise CodexError("Codex 服务关闭了连接；请重试。")
            # A server request is never authorized to act on this computer.
            if "method" in message and "id" in message:
                self._send({"id": message["id"], "error": {
                    "code": -32601, "message": "Prism Interview exposes no tools or approvals."}})
                raise CodexError("模型请求调用工具，已中止本次回答。")
            return message

    def _rpc(self, method: str, params: dict, deadline: float, cancel_event=None) -> dict:
        self._next_id += 1
        request_id = self._next_id
        self._send({"id": request_id, "method": method, "params": params})
        while True:
            message = self._next(deadline, cancel_event, pending=False)
            if message.get("id") != request_id:
                if message.get("method") in ("item/agentMessage/delta", "item/started", "item/completed", "turn/completed", "error"):
                    self._pending_events.append(message)
                continue
            if message.get("error"):
                error = message["error"]
                # Protocol errors contain service diagnostics, never credentials.
                raise CodexError(f"Codex {method}: {str(error.get('message', error))[:1000]}")
            return message.get("result", {})

    def _start(self, deadline: float, cancel_event=None) -> None:
        if self._process is not None and self._process.poll() is None:
            return
        self.close()
        executable = find_codex(self.executable)
        self._workdir = tempfile.TemporaryDirectory(prefix="research-copilot-codex-")
        self._messages = queue.Queue()
        self._pending_events.clear()
        # These overrides apply only to this child process, never the user's config.
        overrides = {
            "approval_policy": '"never"', "sandbox_mode": '"read-only"',
            "model_provider": '"openai"', "web_search": '"disabled"',
            "project_doc_max_bytes": "0", "mcp_servers": "{}",
            "features.shell_tool": "false", "features.unified_exec": "false",
            "features.apps": "false", "features.plugins": "false",
            "features.hooks": "false", "features.multi_agent": "false",
            "features.computer_use": "false", "features.browser_use": "false",
            "features.image_generation": "false", "features.view_image": "false",
            "features.memories": "false", "features.sleep_tool": "false",
            "features.shell_snapshot": "false",
        }
        command = [executable, "app-server", "--listen", "stdio://"]
        for key, value in overrides.items():
            command.extend(["-c", f"{key}={value}"])
        environment = os.environ.copy()
        # Never let unrelated API keys select a metered fallback.
        for name in ("OPENAI_API_KEY", "AZURE_OPENAI_API_KEY"):
            environment.pop(name, None)
        try:
            self._process = subprocess.Popen(
                command, cwd=self._workdir.name, env=environment,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except OSError as exc:
            self.close()
            raise CodexError("无法启动官方 Codex CLI。请检查安装及路径。") from exc
        process = self._process
        target_queue = self._messages

        def read_stdout():
            try:
                for line in process.stdout:
                    try:
                        message = json.loads(line)
                        if isinstance(message, dict):
                            target_queue.put(message)
                    except ValueError:
                        continue
            except (OSError, ValueError):
                pass
            finally:
                target_queue.put(None)

        threading.Thread(target=read_stdout, daemon=True, name="codex-stdio-reader").start()
        self._rpc("initialize", {"clientInfo": {
            "name": "prism_interview", "title": "Prism Interview", "version": "0.4.0"},
            "capabilities": {"experimentalApi": True}}, deadline, cancel_event)
        self._send({"method": "initialized", "params": {}})
        account = self._rpc("account/read", {"refreshToken": False}, deadline, cancel_event)
        if (account.get("account") or {}).get("type") != "chatgpt":
            self.close()
            raise CodexError("此功能只使用已有 ChatGPT 订阅。请在终端运行 codex login 登录 ChatGPT；不接受 API Key 计费模式。")
        self._read_model_catalog(deadline, cancel_event)
        # config/read returns effective settings through the official protocol.
        # Inspect names only; do not persist/log settings or credential values.
        settings = self._rpc("config/read", {"includeLayers": False}, deadline, cancel_event).get("config", {})
        server_names = list(settings.get("mcp_servers") or {})
        if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in server_names):
            raise CodexError("存在无法安全隔离的 MCP 服务名，已停止。请使用纯本地回答或配置独立 CLI。")
        self._tool_config = {f"mcp_servers.{name}.enabled": False for name in server_names}

    def _read_model_catalog(self, deadline, cancel_event=None):
        self._model_catalog = []
        cursor, seen_cursors, seen_models = None, set(), set()
        for _ in range(30):
            params = {"includeHidden": False}
            if cursor:
                params["cursor"] = cursor
            page = self._rpc("model/list", params, deadline, cancel_event)
            for item in page.get("data", []):
                if not isinstance(item, dict):
                    continue
                identity = item.get("model") or item.get("id")
                if identity and identity not in seen_models:
                    self._model_catalog.append(item)
                    seen_models.add(identity)
            cursor = page.get("nextCursor")
            if not cursor:
                break
            if cursor in seen_cursors:
                raise CodexError("Codex 模型目录分页重复，未发送问题。")
            seen_cursors.add(cursor)
        else:
            raise CodexError("Codex 模型目录分页过多，请更新 CLI 后重试。")

    def models(self) -> list[dict]:
        with self._lock:
            deadline = time.monotonic() + min(self.timeout, 30)
            try:
                running = self._process is not None and self._process.poll() is None
                self._start(deadline)
                if running:
                    self._read_model_catalog(deadline)
                return list(self._model_catalog)
            except Exception:
                self.close()
                raise

    def read_rate_limits(self) -> dict:
        """Official read-only quota snapshot; no model turn or credit mutation."""
        with self._lock:
            deadline = time.monotonic() + min(self.timeout, 30)
            try:
                self._start(deadline)
                return self._rpc("account/rateLimits/read", {}, deadline)
            except Exception:
                self.close()
                raise

    def complete(self, system: str, user: str, schema: dict,
                 on_delta: Callable[[str], None] | None = None, cancel_event=None) -> str:
        with self._lock:
            started = time.monotonic()
            self.last_timings = {}
            deadline = time.monotonic() + self.timeout
            thread_id = turn_id = ""
            try:
                if self.service_tier not in ("default", "priority"):
                    raise CodexError("未知 Codex 速度档位；本次未发送问题。")
                self._start(deadline, cancel_event)
                self.last_timings["startup"] = time.monotonic() - started
                phase = time.monotonic()
                try:
                    limits = self._rpc("account/rateLimits/read", {}, deadline, cancel_event)
                except AnswerCancelled:
                    raise
                except Exception as exc:
                    raise CodexError("无法读取 Codex 订阅内额度；本次未发送问题，以避免动用已购 credits。"
                                     "请检查连接或更新 CLI 后重试，或切换本地模式。") from exc
                require_included_quota(limits)
                self.last_timings["quota_check"] = time.monotonic() - phase
                selected = self.model
                if not selected:
                    from .model_catalog import choose_default_gpt
                    selected = choose_default_gpt(self._model_catalog)
                if not selected:
                    raise CodexError("当前账号没有可用 Codex 模型，请检查登录和订阅额度。")
                metadata = next((m for m in self._model_catalog
                                 if selected in (m.get("model"), m.get("id"))), {})
                if not metadata:
                    raise CodexError("所选 GPT 模型不在当前订阅的官方目录中；请打开模型窗口刷新并重新选择。本次未发送问题。")
                supported = {item.get("reasoningEffort") for item in metadata.get("supportedReasoningEfforts", [])}
                effort = next((e for e in ("none", "minimal", "low") if e in supported), "low")
                tiers = metadata.get("serviceTiers") or []
                has_priority = isinstance(tiers, list) and any(
                    isinstance(item, dict) and item.get("id") == "priority" for item in tiers)
                if self.service_tier == "priority" and not has_priority:
                    raise CodexError("所选 Codex 模型的官方目录未确认支持 Fast；本次未发送问题。"
                                     "请检查模型目录或手动选择标准速度。")
                self.last_effort, self.last_service_tier = effort, self.service_tier
                phase = time.monotonic()
                thread = self._rpc("thread/start", {
                    "model": selected, "modelProvider": "openai", "ephemeral": True,
                    "cwd": self._workdir.name, "approvalPolicy": "never", "sandbox": "read-only",
                    "environments": [], "dynamicTools": [],
                    "baseInstructions": system,
                    "developerInstructions": "Return the requested JSON only. No tools, filesystem, shell, web, plugins, or external actions are available or permitted.",
                    "config": self._tool_config,
                    "serviceName": "research_copilot",
                }, deadline, cancel_event)
                self.last_timings["thread_start"] = time.monotonic() - phase
                if not thread.get("thread", {}).get("ephemeral"):
                    raise CodexError("Codex 未确认临时会话，已停止发送问题。请升级 CLI。")
                thread_id = thread["thread"]["id"]
                self.last_model = thread.get("model") or selected
                phase = time.monotonic()
                result = self._rpc("turn/start", {
                    "threadId": thread_id, "input": [{"type": "text", "text": user}],
                    "environments": [], "approvalPolicy": "never", "outputSchema": schema,
                    "effort": self.last_effort, "serviceTierForTurn": self.last_service_tier,
                }, deadline, cancel_event)
                self.last_timings["turn_start"] = time.monotonic() - phase
                turn_id = result.get("turn", {}).get("id", "")
                text_parts = []
                final_text = ""
                while True:
                    message = self._next(deadline, cancel_event)
                    params = message.get("params") or {}
                    if params.get("threadId") not in (None, thread_id):
                        continue
                    method = message.get("method")
                    if method == "item/agentMessage/delta":
                        delta = params.get("delta", "")
                        if delta:
                            self.last_timings.setdefault("first_delta", time.monotonic() - started)
                            text_parts.append(delta)
                            if on_delta:
                                on_delta(delta)
                    elif method in ("item/started", "item/completed"):
                        item = params.get("item") or {}
                        kind = item.get("type", "")
                        if kind not in ("userMessage", "agentMessage", "reasoning", "plan", "contextCompaction"):
                            raise CodexError(f"本次回答出现非文本工具项目（{kind}），已中止。")
                        if method == "item/completed" and kind == "agentMessage":
                            final_text = item.get("text", "") or final_text
                    elif method == "thread/tokenUsage/updated":
                        self.last_usage = params.get("tokenUsage") or {}
                    elif method == "turn/completed":
                        turn = params.get("turn") or {}
                        if turn_id and turn.get("id") != turn_id:
                            continue
                        if turn.get("status") != "completed":
                            error = turn.get("error") or {}
                            raise CodexError("Codex 回答未完成：" + str(error.get("message") or turn.get("status"))[:1000])
                        raw = final_text or "".join(text_parts)
                        if not raw.strip():
                            raise CodexError("Codex 返回了空回答，请重试。")
                        self.last_timings["total"] = time.monotonic() - started
                        return raw
                    elif method == "error":
                        error = params.get("error") or {}
                        if not params.get("willRetry", False):
                            raise CodexError("Codex：" + str(error.get("message", "请求失败"))[:1000])
            except Exception:
                if thread_id and turn_id:
                    try:
                        self._send({"id": "cancel", "method": "turn/interrupt", "params": {
                            "threadId": thread_id, "turnId": turn_id}})
                    except Exception:
                        pass
                self.close()
                raise

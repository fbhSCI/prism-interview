"""HTTP providers. Credentials stay out of settings and diagnostics."""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .config import Preferences
from .codex_client import AnswerCancelled
from .credential_store import load_key


QWEN_REGION_ENDPOINTS = {
    "cn-beijing": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "ap-southeast-1": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
}


def qwen_can_disable_thinking(model: str) -> bool:
    """Known hybrid text families; never claim thinking-only models are fast."""
    name = model.lower()
    thinking_only = {"qwen3.8-2.4t-a95b", "qwen3.7-max-preview", "qwen3.7-max-2026-05-17"}
    if name in thinking_only or "-thinking" in name or "omni" in name:
        return False
    return name.startswith(("qwen3.8-max", "qwen3.8-flash", "qwen3.8-27b", "qwen3.7-max", "qwen3.7-plus",
                            "qwen3.7-flash", "qwen3.6-plus", "qwen3.5-plus", "qwen3-", "qwen-plus", "qwen-flash"))


def provider_connection(preferences: Preferences, *, require_quota: bool = True):
    """Resolve explicit provider credentials, without persisting or logging them.

    Model listing is read-only. Inference requires cloud consent and either
    confirmed free quota or the separate explicit Qwen paid-balance consent.
    """
    provider = preferences.provider.lower()
    if provider == "openai_compatible":
        provider = "compatible"
    if not preferences.cloud_consent:
        raise ValueError("尚未授权云端模型。")
    paid_qwen = provider == "qwen" and getattr(preferences, "qwen_paid_usage_authorized", False)
    if require_quota and not (preferences.free_quota_confirmed or paid_qwen):
        raise ValueError("此 API 尚未确认可用免费额度及禁止超额计费；本次未发送请求。千问也可在设置中明确允许付费调用，或使用已有 Codex 订阅／本地 Ollama。")
    defaults = {"qwen": QWEN_REGION_ENDPOINTS.get(preferences.qwen_region, ""),
                "deepseek": "https://api.deepseek.com/v1"}
    base = (preferences.api_base_url.strip() or defaults.get(provider, "")).rstrip("/")
    parsed = urlparse(base)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.port not in (None, 443)):
        raise ValueError("云 API 需要有效的 HTTPS 官方地址，地址中不能包含凭据、查询参数或片段。")
    host = parsed.hostname.lower()
    if provider == "deepseek" and host != "api.deepseek.com":
        raise ValueError("DeepSeek 请选择 api.deepseek.com 官方地址；其他服务请使用兼容 API。")
    if provider == "qwen" and not (host in {
            "dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com", "dashscope-us.aliyuncs.com",
            "cn-hongkong.dashscope.aliyuncs.com"} or host.endswith(".maas.aliyuncs.com")):
        raise ValueError("千问请选择千问AI平台或对应地域的阿里云百炼官方域名。")
    env_name = {"qwen": "DASHSCOPE_API_KEY", "deepseek": "DEEPSEEK_API_KEY"}.get(provider, "PRISM_INTERVIEW_API_KEY")
    # An existing per-provider map takes precedence and prevents a stale single
    # input field from becoming a cross-provider credential fallback.
    keys = getattr(preferences, "session_api_keys", {}) or {}
    key = keys.get(provider, "") if keys else preferences.api_key
    key = key or os.getenv(env_name, "")
    key = key or load_key(provider, base)
    if not key:
        raise ValueError(f"未设置 API Key。请填写会话密钥、在模型窗口保存本机默认密钥，或设置环境变量 {env_name}。")
    return provider, base, key


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("模型地址发生重定向，已停止以保护 API Key。请填写最终官方 API 地址。")


def _open(request, timeout):
    return build_opener(_NoRedirect()).open(request, timeout=timeout)


def _service_billing_error(code) -> str:
    if code == "Arrearage":
        return ("服务商账户欠费或计费不可用（Arrearage）。请到这把密钥所属账号的费用中心"
                "检查余额和欠费状态；本次未生成完整回答。")
    return ""


def _http_error_message(exc: HTTPError) -> str:
    # Inspect only a bounded error code. Never echo provider messages/bodies:
    # an error body may contain input excerpts or even credentials.
    code = None
    try:
        body = json.loads(exc.read(32768))
        error = body.get("error", body) if isinstance(body, dict) else {}
        code = error.get("code") if isinstance(error, dict) else None
    except (ValueError, OSError, TypeError):
        pass
    billing = _service_billing_error(code)
    if billing:
        return billing
    label = {401: "密钥无效或已过期", 403: "无访问权限", 404: "模型或地址不存在",
             429: "额度不足或请求过于频繁"}.get(exc.code, "服务请求失败")
    return f"模型服务 HTTP {exc.code}：{label}。"


def stream_http(request: Request, timeout: float, protocol: str, on_delta=None,
                cancel_event=None, opener=None, on_stats=None) -> str:
    """Cancellable at the caller even while the socket is connecting/reading.

    Background socket work is bounded by a finite timeout and closes on cancellation.
    Successful streaming requires a provider completion marker, not just socket EOF.
    """
    if cancel_event is not None and cancel_event.is_set():
        raise AnswerCancelled("已取消回答。")
    events: queue.Queue = queue.Queue(maxsize=256)
    stop = threading.Event()
    holder: list = []
    deadline = time.monotonic() + max(0.1, timeout)
    opener = opener or _open

    def put(kind, value):
        while not stop.is_set():
            try:
                events.put((kind, value), timeout=0.1)
                return
            except queue.Full:
                pass

    def worker():
        stat_names = ("prompt_eval_count", "eval_count", "load_duration",
                      "prompt_eval_duration", "eval_duration", "done_reason")
        try:
            with opener(request, timeout=max(timeout, 1)) as response:
                holder.append(response)
                if on_delta is None:
                    data = json.load(response)
                    if protocol == "ollama":
                        if data.get("error"):
                            raise RuntimeError("Ollama 返回错误，请检查模型名称与内存。")
                        text = data.get("message", {}).get("content", "")
                        put("stats", {name: data[name] for name in stat_names if name in data})
                    else:
                        text = data["choices"][0]["message"]["content"]
                        if isinstance(data.get("model"), str):
                            put("stats", {"reported_model": data["model"]})
                    put("done", text)
                    return
                completed = False
                while not stop.is_set():
                    line = response.readline()
                    if not line:
                        break
                    decoded = line.decode("utf-8", "strict").strip()
                    if not decoded:
                        continue
                    if protocol == "compatible":
                        if not decoded.startswith("data:"):
                            continue
                        decoded = decoded[5:].strip()
                        if decoded == "[DONE]":
                            completed = True
                            break
                    item = json.loads(decoded)
                    if item.get("error"):
                        error = item['error']
                        billing = _service_billing_error(error.get('code') if isinstance(error, dict) else None)
                        raise RuntimeError(billing or "模型服务返回错误，请检查额度、模型名称和连接。")
                    if protocol == "ollama":
                        delta = item.get("message", {}).get("content", "")
                        if delta:
                            put("delta", delta)
                        if item.get("done"):
                            if item.get("done_reason") == "length":
                                raise RuntimeError("本地回答达到输出长度上限，未完整生成。")
                            completed = True
                            put("stats", {name: item[name] for name in stat_names if name in item})
                            break
                    else:
                        if isinstance(item.get("model"), str):
                            put("stats", {"reported_model": item["model"]})
                        for choice in item.get("choices", []):
                            delta = choice.get("delta", {}).get("content") or ""
                            if delta:
                                put("delta", delta)
                            reason = choice.get("finish_reason")
                            if reason:
                                if reason != "stop":
                                    raise RuntimeError("模型回答未正常结束：" + str(reason))
                                completed = True
                if not stop.is_set():
                    if not completed:
                        raise RuntimeError("回答连接提前断开，内容不完整；请重试。")
                    put("done", "")
        except HTTPError as exc:
            put("error", RuntimeError(_http_error_message(exc)))
        except (URLError, TimeoutError, OSError) as exc:
            put("error", RuntimeError("模型连接失败或超时；请检查服务、网络和地址。"))
        except Exception as exc:
            put("error", exc)

    threading.Thread(target=worker, daemon=True, name="model-http-stream").start()
    parts = []
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise AnswerCancelled("已取消回答。")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("模型回答超时，请重试或切换模型。")
            try:
                kind, value = events.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            if kind == "error":
                raise value
            if kind == "stats":
                if on_stats:
                    on_stats(value)
                continue
            if kind == "done":
                raw = value or "".join(parts)
                if not raw.strip():
                    raise RuntimeError("模型返回空回答，请重试。")
                return raw
            parts.append(value)
            if on_delta:
                on_delta(value)
    finally:
        stop.set()
        # close() may block waiting for another thread's buffered read lock.
        # Keep the UI's cancellation path independent of that lock.
        if holder and hasattr(holder[0], "close"):
            threading.Thread(target=holder[0].close, daemon=True, name="model-http-close").start()


class OpenAICompatibleClient:
    def __init__(self, preferences: Preferences):
        self.preferences = preferences
        self.provider, base, self._key = provider_connection(preferences)
        self.endpoint = base + "/chat/completions"
        self.model = preferences.answer_model.strip()
        self.last_model = ""
        self.last_metrics = {}
        if not self.model:
            raise ValueError("请填写该服务的确切模型名称，程序不会自动选择计费模型。")

    def complete(self, system: str, user: str, on_delta=None, cancel_event=None):
        payload = {
            "model": self.model, "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "stream": on_delta is not None, "temperature": 0.15,
            # A full English answer, its Chinese counterpart and question
            # translation share one response; the old ceiling could truncate
            # the second language when a multipart question needs detail.
            "max_tokens": (1000 if self.preferences.brief_answer
                           and self.preferences.max_answer_words <= 100 else 2000),
            "response_format": {"type": "json_object"},
        }
        if on_delta is not None:
            payload["stream_options"] = {"include_usage": True}
        if self.provider == "deepseek":
            payload["thinking"] = {"type": "disabled"}
        elif self.provider == "qwen" and qwen_can_disable_thinking(self.model):
            payload["enable_thinking"] = False
        req = Request(self.endpoint, data=json.dumps(payload).encode("utf-8"), method="POST",
                      headers={"Content-Type": "application/json", "Authorization": "Bearer " + self._key})
        def stats(values):
            self.last_metrics.update(values)
            self.last_model = values.get("reported_model") or self.last_model
        return stream_http(req, self.preferences.answer_timeout, "compatible", on_delta, cancel_event, on_stats=stats)

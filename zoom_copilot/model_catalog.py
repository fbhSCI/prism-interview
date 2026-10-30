"""Read-only model discovery. A catalog never proves access or free inference.

Official catalogs are queried only through Codex app-server, provider /models, or
local Ollama /api/tags. No answer/test prompt is sent to discover a model.
"""
from __future__ import annotations

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request

from .config import Preferences


TESTED_FAST_GPT = "gpt-5.6-luna"
TESTED_FAST_NOTE = "候选速度选项；实际可用模型、延迟与额度以你的账号和实测为准"


def model_id(item: dict) -> str:
    return str(item.get("model") or item.get("id") or "").strip()


def supports_fast(item: dict) -> bool:
    tiers = item.get("serviceTiers") or []
    return isinstance(tiers, list) and any(isinstance(tier, dict) and tier.get("id") == "priority" for tier in tiers)


def is_text_answer_model(item: dict, provider: str = "codex") -> bool:
    identity = model_id(item).lower()
    if not identity:
        return False
    modalities = item.get("outputModalities")
    if isinstance(modalities, list) and modalities and "text" not in modalities:
        return False
    # Model providers sometimes include embedding/audio endpoints in /models.
    # Text-capable multimodal chat models are permitted; dedicated media and
    # embedding endpoints cannot satisfy this app's JSON answer contract.
    excluded = ("embedding", "rerank", "realtime", "-tts", "-asr", "-image", "wanx", "transcription")
    if any(word in identity for word in excluded):
        return False
    if provider == "qwen" and not identity.startswith("qwen"):
        return False
    return True


def choose_default_gpt(catalog: list[dict]) -> str:
    """Use the prior measured fast choice only when it exists in this catalog."""
    choices = [item for item in catalog if is_text_answer_model(item)]
    for item in choices:
        if model_id(item) == TESTED_FAST_GPT:
            return model_id(item)
    luna = [item for item in choices if "luna" in model_id(item).lower()]
    if luna:
        preferred = next((item for item in luna if supports_fast(item)), luna[0])
        return model_id(preferred)
    default = next((item for item in choices if item.get("isDefault")), None)
    return model_id(default or next(iter(choices), {}))


def fetch_provider_models(preferences: Preferences, *, opener=None) -> list[dict]:
    """Read official provider catalog with the explicitly selected session key.

    The free-quota confirmation gates answers, not this non-inference GET.
    Some DashScope regions/accounts do not expose /models: report that failure
    and let the user enter an exact console model ID rather than invent access.
    """
    from .providers import _open, provider_connection
    provider = preferences.provider.lower()
    if provider == "ollama":
        from urllib.parse import urlparse
        endpoint = preferences.ollama_endpoint.rstrip("/")
        parsed = urlparse(endpoint)
        if parsed.scheme not in ("http", "https") or parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("本地 Ollama 目录地址必须为 loopback。")
        request = Request(endpoint + "/api/tags", method="GET")
    else:
        provider, base, key = provider_connection(preferences, require_quota=False)
        request = Request(base + "/models", method="GET", headers={"Authorization": "Bearer " + key,
                                                                  "Accept": "application/json"})
    try:
        with (opener or _open)(request, timeout=min(15, max(1, preferences.answer_timeout))) as response:
            payload = json.load(response)
    except HTTPError as exc:
        label = {401: "密钥无效或地域不匹配", 403: "目录无访问权限", 404: "该地域未提供 /models 目录",
                 429: "目录请求过于频繁"}.get(exc.code, "目录服务失败")
        raise RuntimeError(f"读取模型目录 HTTP {exc.code}：{label}。请在控制台核对确切模型标识；未发送回答请求。") from None
    except (URLError, TimeoutError, OSError, ValueError):
        raise RuntimeError("模型目录连接失败或格式无效；未发送回答请求。") from None
    if not isinstance(payload, dict):
        raise ValueError("模型目录格式无效；未发送回答请求。")
    items = payload.get("models" if provider == "ollama" else "data")
    if not isinstance(items, list):
        raise ValueError("模型目录未返回模型列表；未发送回答请求。")
    result, seen = [], set()
    for raw in items:
        if not isinstance(raw, dict):
            continue
        item = dict(raw)
        if provider == "ollama":
            item["model"] = str(raw.get("model") or raw.get("name") or "")
        identity = model_id(item)
        if identity in seen or not is_text_answer_model(item, provider):
            continue
        seen.add(identity)
        item.setdefault("model", identity)
        item["catalogSource"] = "local-tags" if provider == "ollama" else "official-models-endpoint"
        result.append(item)
    return result

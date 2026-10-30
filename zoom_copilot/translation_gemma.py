"""Local TranslateGemma 4B via loopback Ollama; no cloud fallback or credentials."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil
import re
import subprocess
import threading
import time
from urllib.request import Request, urlopen

MODEL = "translategemma:4b"
BASE = "http://127.0.0.1:11434"
PROMPT = ("You are a professional English (en) to Chinese (zh-Hans) translator. Your goal is to "
          "accurately convey the meaning and nuances of the original English text while adhering "
          "to Chinese grammar, vocabulary, and cultural sensitivities.\n"
          "Produce only the Chinese translation, without any additional explanations or commentary. "
          "Please translate the following English text into Chinese:\n\n\n")

TERMS = {"sensing operator": "测量算子", "material reflectance": "材料反射特性",
         "analyzer intensities": "检偏通道强度",
         "phrasal or two-word verbs": "短语动词（双词动词）", "inverse rendering": "逆渲染"}


def translation_prompt(text):
    # Controlled translation terminology only. Nothing is passed into Whisper's
    # acoustic decoder, and no words absent from the source are injected there.
    matched = [f'{en} = {zh}' for en, zh in TERMS.items()
               if re.search(r'(?<!\w)' + re.escape(en) + r'(?!\w)', text, re.I)]
    prompt = PROMPT
    if re.search(r'\bmodels?\b',text,re.I):
        prompt = prompt.replace('Produce only',
            'When the source explicitly means a computational or scientific model, use 模型. '
            'Use 模特 only for an explicitly human fashion model and 型号 only for an explicit product identifier. '
            'Do not invent expansions of source abbreviations.\nProduce only')
    if matched:
        prompt = prompt.replace('Produce only', 'Use this terminology for phrases in the source: ' + '; '.join(matched) + '.\nProduce only')
    return prompt + text


class GemmaTranslator:
    # Prior sentences disambiguate words such as "model" and resolve pronouns.
    # Unlike Argos, sentence-by-sentence caching loses useful source context.
    context_sensitive = True

    def __init__(self, model_dir):
        self.model_dir = Path(model_dir)
        self._lock = threading.Lock()
        self.last_metrics = {}

    @property
    def ready(self):
        try:
            with urlopen(BASE + "/api/tags", timeout=1) as response:
                return any(m.get("name") == MODEL for m in json.load(response).get("models", []))
        except (OSError, ValueError):
            return False

    def _ensure_server(self):
        try:
            with urlopen(BASE + "/api/version", timeout=1):
                return
        except OSError:
            pass
        portable = self.model_dir.parent / "ollama" / "ollama.exe"
        executable = str(portable) if portable.is_file() else shutil.which("ollama")
        if not executable:
            raise RuntimeError("TranslateGemma 需要本地 Ollama。请先安装 Ollama 或使用包含 Ollama 的运行包。")
        env = {**os.environ, "OLLAMA_HOST": "127.0.0.1:11434", "OLLAMA_MODELS": str(self.model_dir / "ollama")}
        subprocess.Popen([executable, "serve"], env=env, stdin=subprocess.DEVNULL,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for _ in range(30):
            try:
                with urlopen(BASE + "/api/version", timeout=.4):
                    return
            except OSError:
                time.sleep(.2)
        raise RuntimeError("本地 Ollama 未启动；请检查端口 11434。")

    def install_english_chinese(self, on_status=None):
        status = on_status or (lambda _: None)
        self._ensure_server()
        if self.ready:
            return "TranslateGemma 4B 已安装。"
        status("下载本地 TranslateGemma 4B（约 3.3GB，Gemma 使用条款）；无云端推理费用。")
        request = Request(BASE + "/api/pull", data=json.dumps({"model": MODEL, "stream": True}).encode(),
                          headers={"Content-Type": "application/json"})
        last = 0
        with urlopen(request, timeout=60) as response:
            for line in response:
                row = json.loads(line)
                if row.get("error"):
                    raise RuntimeError("本地翻译模型下载失败。")
                if time.monotonic() - last > 4:
                    percent = 100 * row.get("completed", 0) / max(1, row.get("total", 1))
                    status(f"TranslateGemma：{row.get('status', '')} {percent:.0f}%")
                    last = time.monotonic()
        if not self.ready:
            raise RuntimeError("TranslateGemma 下载后未出现在本地模型列表。")
        return "TranslateGemma 4B 已安装。"

    def translate(self, text):
        if not text.strip():
            return ""
        if len(text) > 12000:
            raise ValueError("本段文本过长，未静默截断。请缩短语音分段。")
        with self._lock:
            start = time.monotonic()
            payload = {"model": MODEL, "messages": [{"role": "user", "content": translation_prompt(text)}],
                       "stream": False, "keep_alive": "30m", "options": {"temperature": 0,
                       "num_ctx": 4096, "num_predict": 1024}}
            request = Request(BASE + "/api/chat", data=json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json"})
            try:
                with urlopen(request, timeout=35) as response:
                    result = json.load(response)
            except OSError as exc:
                raise RuntimeError("本地 TranslateGemma 连接超时／失败。没有改用云端。") from exc
            output = result.get("message", {}).get("content", "").strip()
            if not output or not result.get("done") or result.get("done_reason") == "length":
                raise RuntimeError("TranslateGemma 未完整生成译文。")
            if re.search(r'\bend-to-end\b', text, re.I):
                # Exact technical idiom normalization; avoid an extra prompt
                # instruction that induced a longer, unsupported explanation.
                output = output.replace("从头到尾", "端到端")
            self.last_metrics = {"translation_seconds": time.monotonic() - start,
                                 "translation_engine": MODEL,
                                 "eval_count": result.get("eval_count"),
                                 "eval_duration_ns": result.get("eval_duration"),
                                 "load_duration_ns": result.get("load_duration")}
            return output

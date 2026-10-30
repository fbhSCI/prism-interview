"""Independent offline EN→ZH translation, using an official Argos model.

Loads the model directly through CTranslate2 and SentencePiece: no torch,
Stanza downloads, cloud translation or paid service is required.
"""
from __future__ import annotations
from collections import deque, OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import re
import threading
import time
from urllib.request import urlopen
import zipfile
from .runtime import default_model_cache
from .translation_protection import protect_translation_source

INDEX_URL = "https://raw.githubusercontent.com/argosopentech/argospm-index/main/index.json"
_ARGOS_MAX_DECODING_TOKENS = 384

# Exact phrases across interview and technical language. These are lexical
# translations, never a candidate profile, retrieval prior or claim of accuracy.
RESEARCH_TERMS = {
    "spectral polarimetric ghost imaging": "光谱偏振鬼成像",
    "incident stokes vectors": "入射斯托克斯向量",
    "forward instrument matrix": "前向仪器矩阵",
    "inverse demodulation": "逆解调",
    "analyzer intensities": "检偏通道强度",
    "subsurface scattering": "次表面散射",
    "multi illumination data set": "多光照数据集",
    "multi illumination dataset": "多光照数据集",
    "ground truth albedo": "反照率参考真值",
    "ground truth": "参考真值",
    "light stages": "光照采集装置",
    "light stage": "光照采集装置",
    "anisotropic": "各向异性",
    "relighting": "重光照",
    "albedo": "反照率",
    "pseudoinverse": "伪逆",
    "end to end": "端到端",
    "back propagation": "反向传播",
    "back propagated": "反向传播",
    "back propagate": "反向传播",
    "backpropagation": "反向传播",
    "backpropagated": "反向传播",
    "backpropagate": "反向传播",
    "hyperspectral polarimetric reflectance": "高光谱偏振反射",
    "scattering induced background interference": "散射引起的背景干扰",
    "wavelength dependent attenuation": "波长相关衰减",
    "stokes calibration matrix": "斯托克斯标定矩阵",
    "recovered stokes parameters": "反演得到的斯托克斯参数",
    "polarization reconstruction": "偏振重建",
    "circular polarization": "圆偏振",
    "linear polarization": "线偏振",
    "complex valued": "复值",
    "real valued": "实值",
    "polarimetric reflectance": "偏振反射",
    "spectral reconstruction": "光谱重建",
    "calibration uncertainty": "标定不确定性",
    "novel view synthesis": "新视角合成",
    "material reflectance": "材质反射特性",
    "material properties": "材质属性",
    "calibration matrix": "标定矩阵",
    "stokes parameters": "斯托克斯参数",
    "sensing operator": "感知算子",
    "inverse rendering": "逆渲染",
    "uniquely identify": "唯一确定",
    "uniquely determine": "唯一确定",
    "speckle modulation": "散斑调制",
    "stokes fields": "斯托克斯场",
    "ghost imaging": "鬼成像",
    "hyperspectral": "高光谱",
    "ill conditioned": "病态的",
    "normal map": "法线图",
    "full rank": "满秩",
    "identifiability": "可辨识性",
    "people person": "善于与人打交道的人",
    "transferable skills": "可迁移技能",
    "career goals": "职业目标",
    "team player": "善于团队合作的人",
    "conflict resolution": "冲突解决",
    "project management": "项目管理",
    "stakeholder management": "利益相关方管理",
    "customer experience": "客户体验",
    "supply chain": "供应链",
    "cash flow": "现金流",
    "informed consent": "知情同意",
    "clinical trial": "临床试验",
    "statistical significance": "统计显著性",
    "confidence interval": "置信区间",
    "root cause analysis": "根因分析",
}
_TERM_ITEMS = sorted(RESEARCH_TERMS.items(), key=lambda pair: len(pair[0]), reverse=True)
_TERM_PATTERN = re.compile("|".join(
    f"(?P<T{i}>(?<![A-Za-z])" + r"[-\s]+".join(re.escape(w) for w in en.split()) + r"(?![A-Za-z]))"
    for i, (en, _) in enumerate(_TERM_ITEMS)), re.I)
_MARKER = re.compile(r"\[\s*TERM\s*\d+\s*\]|\bTERM\s*\d+\b", re.I)


@dataclass(frozen=True)
class ProtectedTerm:
    marker: str
    source: str
    chinese: str


def protect_research_terms(sentence: str) -> tuple[str, list[ProtectedTerm]]:
    """Replace only complete listed terms; collisions/overload opt out.

    No fuzzy recovery is allowed. A decoder must return every marker exactly
    once or the original, unmodified English sentence is translated instead.
    """
    matches = list(_TERM_PATTERN.finditer(sentence))
    if _MARKER.search(sentence) or len(matches) > 32:
        return sentence, []
    bindings = []

    def replace(match):
        _, chinese = _TERM_ITEMS[int(match.lastgroup[1:])]
        marker = f"[TERM{100 + len(bindings)}]"
        bindings.append(ProtectedTerm(marker, match.group(), chinese))
        return marker

    return _TERM_PATTERN.sub(replace, sentence), bindings


def restore_research_terms(output: str, bindings: list[ProtectedTerm]) -> str | None:
    """Return None on any dropped, mutated, repeated or invented marker."""
    if any(output.count(binding.marker) != 1 for binding in bindings):
        return None
    restored = output
    for binding in bindings:
        restored = restored.replace(binding.marker, binding.chinese)
    if _MARKER.search(restored):
        return None
    return restored


def _correct_known_terms(source: str, output: str) -> str:
    """Small source-conditioned corrections; never append a missing claim."""
    corrections = {
        "sensing operator": {"感知操作员": "感知算子", "传感操作员": "感知算子", "感应操作员": "感知算子",
                             "传感器操作员": "感知算子", "感应操作器": "感知算子"},
        "ghost imaging": {"幽灵成像": "鬼成像"},
        "hyperspectral": {"超光谱": "高光谱"},
        "polarimetric": {"光谱极化": "光谱偏振", "极度反射": "偏振反射", "极化反射": "偏振反射"},
        "inverse rendering": {"反向渲染": "逆渲染"},
        "material properties": {"物质属性": "材质属性"},
        "stokes parameters": {"检索斯托克斯参数": "求解斯托克斯参数"},
        "novel view synthesis": {"新颖的视图合成": "新视角合成"},
        "scattering induced background interference": {"斑点引起的背景干扰": "散射引起的背景干扰"},
    }
    for term, replacements in corrections.items():
        pattern = r"(?<![A-Za-z])" + r"[-\s]+".join(re.escape(w) for w in term.split()) + r"(?![A-Za-z])"
        if re.search(pattern, source, re.I):
            for original, preferred in replacements.items():
                output = output.replace(original, preferred)
    # ASR may spell this learning term as one word, two words or a hyphenated
    # word. Correct an actual mistranslation only when the source contains it;
    # never append a claim if the decoder omitted the phrase.
    if re.search(r"\bback[-\s]?propagat(?:e|ed|ion)\b", source, re.I):
        for original in ("背面传播", "背部传播", "向后传播", "后向传播"):
            output = output.replace(original, "反向传播")
    # Weight is also a physical quantity. Require plural weights explicitly
    # attached to a model/network plus a learning/training context, and reject
    # body-mass contexts. Do not turn a learned predictor's physical weights
    # into neural coefficients merely because a model is mentioned nearby.
    neural_weights = bool(re.search(
        r"\b(?:(?:neural(?:\s+network)?|network|model)\s+weights|"
        r"weights\s+(?:in|of|for)\s+(?:the\s+)?[^.!?]{0,100}\b(?:network|model))\b",
        source, re.I))
    trained = bool(re.search(r"\b(?:learn(?:ed|t|ing)?|train(?:ed|ing)?|back[-\s]?propagat(?:e|ed|ion))\b",
                             source, re.I))
    physical = bool(re.search(r"\b(?:body|mass|kilograms?|kg|pounds?|patients?|participants?|subjects?)\b",
                              source, re.I))
    if neural_weights and trained and not physical:
        output = output.replace("重量", "权重")
    return output


def _translation_protected_spans(source: str) -> list[tuple[int, int]]:
    """Source spans across which an internal inference cut is unsafe."""
    spans = [(match.start(), match.end()) for match in _TERM_PATTERN.finditer(source)]
    spans.extend((match.start(), match.end()) for match in re.finditer(
        r'\b(?:not|no|never|without|rather\s+than|instead\s+of)\s+\S+|'
        r'(?<!\w)[+-]?\d[\d,.]*(?:[eE][+-]?\d+)?\s+'
        r'(?:nm|um|μm|mm|cm|m|ms|s|Hz|kHz|MHz|GHz|kg|g|degrees?|percent)\b',
        source, re.I))
    # Use the same exact spans as formula protection rather than guessing
    # equation boundaries from whitespace. Never split an equation in half.
    cursor = 0
    for _, literal in protect_translation_source(source).slots:
        start = source.find(literal, cursor)
        if start >= 0:
            spans.append((start, start + len(literal)))
            cursor = start + len(literal)
    return spans


def _bounded_translation_blocks(sentence: str, max_words: int = 120,
                                max_chars: int = 1000) -> list[str]:
    """Bound MT input without inventing English sentence boundaries.

    Complete words, equations and known multiword terms stay intact. Existing clause
    punctuation is preferred near the budget; otherwise only whitespace is
    used. A single indivisible word/term may exceed the character budget.
    This is an internal inference boundary, never a displayed/source sentence.
    """
    sentence = str(sentence).strip()
    words = list(re.finditer(r'\S+', sentence))
    if not words:
        return []
    if len(words) <= max_words and len(sentence) <= max_chars:
        return [sentence]
    protected = _translation_protected_spans(sentence)
    blocked = {index for index in range(1, len(words))
               if any(start < words[index].start() < end for start, end in protected)}
    blocks = []
    begin = 0
    while begin < len(words):
        end = min(len(words), begin + max_words)
        while end > begin + 1 and words[end - 1].end() - words[begin].start() > max_chars:
            end -= 1
        while end > begin and end in blocked:
            end -= 1
        if end == begin:
            # No legal boundary inside the budget: emit this whole protected
            # unit, preserving spelling/numbers instead of slicing a word.
            end = begin + 1
            while end in blocked:
                end += 1
        if end < len(words):
            minimum = begin + max(1, (end - begin) // 2)
            preferred = [index for index in range(minimum, end + 1)
                         if index not in blocked and words[index - 1].group().endswith((',', ';', ':'))]
            if preferred:
                end = preferred[-1]
        blocks.append(sentence[words[begin].start():words[end - 1].end()])
        begin = end
    return blocks


_CONTINUATION_WORDS = frozenset((
    "a an the this that these those my your his her its our their each every "
    "other another any some all both either neither and or but because although "
    "while whether if than as of to for from with without in on at by into onto "
    "through between among about is are was were be been being am have has had "
    "do does did can could may might must shall should will would not mean"
).split())


def _coherent_translation_units(text: str, *, target_words: int = 24,
                                max_words: int = 100, max_chars: int = 1000) -> list[str]:
    """Group consecutive confirmed snippets, without rewriting their wording.

    ASR punctuation often marks an audio cut rather than a sentence. Translating
    ``are. Completely back-propagated.`` independently loses that relationship.
    Small snippets therefore share a bounded input. Once a unit reaches the
    soft target, its unchanged prefix can be cached; only its growing tail is
    reconsidered. No waiting for another audio segment is introduced here.
    """
    from .transcript_text import split_sentences
    snippets = split_sentences(text) or ([str(text).strip()] if str(text).strip() else [])
    output, current = [], []
    word_count, char_count = 0, 0
    for snippet in snippets:
        for part in _bounded_translation_blocks(snippet, max_words, max_chars):
            words = len(part.split())
            if current and (word_count + words > max_words
                            or char_count + 1 + len(part) > max_chars):
                output.append(" ".join(current))
                current, word_count, char_count = [], 0, 0
            current.append(part)
            word_count += words
            char_count += len(part) + int(len(current) > 1)
            tokens = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)?", part)
            continuation = bool(tokens and tokens[-1].casefold() in _CONTINUATION_WORDS)
            if ((word_count >= target_words and not continuation)
                    or word_count >= max_words or char_count >= max_chars):
                output.append(" ".join(current))
                current, word_count, char_count = [], 0, 0
    if current:
        output.append(" ".join(current))
    return output


def _bounded_preceding_context(text: str, max_chars: int = 600) -> str:
    """Keep the nearest source-only context on whole word/formula boundaries."""
    text = str(text).strip()
    if len(text) <= max_chars:
        return text
    minimum = len(text) - max_chars
    protected = _translation_protected_spans(text)
    for word in re.finditer(r"\S+", text):
        start = word.start()
        if start >= minimum and not any(a < start < b for a, b in protected):
            return text[start:]
    # One indivisible formula/word can exceed the context budget. Context is
    # optional; omit it instead of uploading a broken suffix of that formula.
    return ""


class ArgosTranslator:
    def __init__(self, model_dir: str | Path | None = None):
        self.model_dir = Path(model_dir) if model_dir else default_model_cache() / "translation"
        self._translator = None
        self._tokenizer = None
        self._lock = threading.RLock()
        self._model_path = None
        self.last_error = ""
        self.last_warnings: list[str] = []
        self.last_metrics: dict = {}
        self.refresh()

    @property
    def ready(self) -> bool:
        return self._model_path is not None

    def refresh(self) -> bool:
        self._model_path = None
        if self.model_dir.exists():
            for metadata in self.model_dir.rglob("metadata.json"):
                try:
                    data = json.loads(metadata.read_text(encoding="utf-8"))
                    root = metadata.parent
                    if (data.get("from_code"), data.get("to_code")) == ("en", "zh") and (root/"model"/"model.bin").exists() and (root/"sentencepiece.model").exists():
                        self._model_path = root
                        break
                except (OSError, ValueError):
                    continue
        return self.ready

    def _load(self):
        if self._translator is not None and self._tokenizer is not None:
            return
        if not self.ready:
            raise RuntimeError("本地英译中模型尚未安装；请点击检查／准备模型。")
        import ctranslate2
        import sentencepiece
        # SentencePiece's native filename loader cannot open some Windows
        # Unicode paths. Python opens the file correctly; pass the serialized
        # model instead, allowing portable packages in Chinese-named folders.
        tokenizer = sentencepiece.SentencePieceProcessor(
            model_proto=(self._model_path/"sentencepiece.model").read_bytes())
        translator = ctranslate2.Translator(
            str(self._model_path/"model"), device="cpu", compute_type="int8",
            inter_threads=1, intra_threads=3)
        # Publish only a fully initialized pair so a failed load can be retried.
        self._translator, self._tokenizer = translator, tokenizer

    def translate(self, text: str) -> str:
        if not text.strip():
            return ""
        with self._lock:
            started = time.perf_counter()
            self.last_warnings = []
            try:
                self._load()
                from .transcript_text import split_sentences
                source_sentences = split_sentences(text) or [text.strip()]
                groups = [_bounded_translation_blocks(sentence) for sentence in source_sentences]
                # Do not undo the worker's coherent grouping by splitting at
                # every ASR full stop a second time inside the local engine.
                sentences = _coherent_translation_units(text, max_words=120)
                protected = [protect_research_terms(sentence) for sentence in sentences]
                tokens = [self._tokenizer.encode(item[0], out_type=str) for item in protected]
                batches = self._translator.translate_batch(
                    tokens, beam_size=3, max_decoding_length=_ARGOS_MAX_DECODING_TOKENS,
                    replace_unknowns=True, length_penalty=0.2)
                if len(batches) != len(sentences):
                    raise RuntimeError("翻译模型返回句数不匹配，未输出不完整译文。")

                def decode_complete(result):
                    hypothesis = result.hypotheses[0]
                    if len(hypothesis) >= _ARGOS_MAX_DECODING_TOKENS:
                        raise RuntimeError("译文达到模型长度上限，未显示可能截断的中文；请核对英文或重试较短片段。")
                    return self._tokenizer.decode(hypothesis).replace("▁", " ")

                outputs, fallback_count, term_count = [], 0, 0
                for index, (sentence, (_, bindings), result) in enumerate(zip(sentences, protected, batches), 1):
                    raw = decode_complete(result)
                    restored = restore_research_terms(raw, bindings) if bindings else raw
                    if restored is None:
                        # Re-run this whole original bounded input, never a
                        # guessed repair of malformed/missing placeholder text.
                        fallback_count += 1
                        original_tokens = self._tokenizer.encode(sentence, out_type=str)
                        baseline = self._translator.translate_batch(
                            [original_tokens], beam_size=3, max_decoding_length=_ARGOS_MAX_DECODING_TOKENS,
                            replace_unknowns=True, length_penalty=0.2)[0]
                        restored = decode_complete(baseline)
                        self.last_warnings.append(f"第 {index} 个翻译片段术语保护未通过完整性校验，已从原始英文重新翻译；请核对英语原文。")
                    else:
                        term_count += len(bindings)
                    outputs.append(_correct_known_terms(sentence, restored).strip())
                output = " ".join(outputs)
                self.last_error = ""
                self.last_metrics = {"translation_seconds": round(time.perf_counter() - started, 4),
                                     "terms_restored": term_count, "term_fallback_sentences": fallback_count,
                                     "sentences": len(source_sentences),
                                     "translation_blocks": len(sentences),
                                     "long_sentences_split": sum(len(group) > 1 for group in groups)}
                return output.strip()
            except Exception as exc:
                self.last_error = str(exc)
                raise RuntimeError("本地翻译失败：" + str(exc)) from exc

    def install_english_chinese(self, on_status=None) -> str:
        status = on_status or (lambda message: None)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        status("读取官方 Argos 模型目录…")
        with urlopen(INDEX_URL, timeout=30) as response:
            pairs = json.load(response)
        pair = next((p for p in pairs if p.get("from_code") == "en" and p.get("to_code") == "zh"), None)
        if pair is None:
            raise RuntimeError("官方目录没有 en→zh 翻译包。")
        url = next((u for u in pair["links"] if u.startswith("https://argos-net.com/")), None)
        if not url:
            raise RuntimeError("翻译包未提供受支持的官方下载地址。")
        target = self.model_dir / "english-chinese.argosmodel"
        partial = target.with_suffix(".download")
        status("下载离线英译中模型（约80MB）…")
        with urlopen(url, timeout=90) as response, partial.open("wb") as dest:
            total = 0
            while chunk := response.read(1024*1024):
                total += len(chunk)
                if total > 500*1024*1024:
                    raise RuntimeError("翻译包大小异常，已停止。")
                dest.write(chunk)
        partial.replace(target)
        self.install_from_path(target)
        return "离线英译中模型已安装。"

    def install_from_path(self, archive: str | Path):
        self.model_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as package:
            base = self.model_dir.resolve()
            for item in package.infolist():
                dest = (base / item.filename).resolve()
                if not dest.is_relative_to(base):
                    raise ValueError("翻译模型包包含越界路径。")
            # Only weights, tokenizer, metadata and license/readme; no executable code.
            for item in package.infolist():
                if item.is_dir() or "/stanza/" in item.filename:
                    continue
                if not item.filename.endswith((".bin", ".json", ".model", ".md", ".txt")):
                    continue
                package.extract(item, base)
        self._translator = None
        self._tokenizer = None
        if not self.refresh():
            raise RuntimeError("翻译包安装后未发现有效模型。")


class TranslationWorker:
    """Finals have FIFO priority; one replaceable preview is throttled.

    The current native translation cannot be interrupted. A final can wait for
    that one running preview, but never for a backlog of obsolete previews.
    Preview callbacks carry the English source so a UI can reject already-posted
    results when its current ASR hypothesis changes.
    """
    def __init__(self, translator, on_result, on_status=None, on_metrics=None, *,
                 on_preview=None, preview_interval=0.8, allow_prefix_preview=False):
        self.translator, self.on_result = translator, on_result
        self.on_status = on_status or (lambda message: None)
        self.on_metrics = on_metrics or (lambda data: None)
        self.on_preview = on_preview or (lambda identity, source, translated: None)
        self.preview_interval = max(0.0, float(preview_interval))
        self.allow_prefix_preview = bool(allow_prefix_preview)
        self._preview_sources = {}
        self._jobs = deque()
        self._preview = None
        self._revisions = {}
        self._finalized = set()
        self._last_preview_started = float("-inf")
        self._condition = threading.Condition()
        self._stopped = False
        self._working = False
        # Confirmed speaker cards grow cumulatively. Re-translating every old
        # sentence on every 3-second update caused increasing delay and changed
        # already readable Chinese. Cache exact coherent source units (and their
        # bounded context when supported), never arbitrary tiny ASR fragments.
        self._sentence_cache = OrderedDict()
        self._maximum_cached_sentences = 192
        self._last_final_warnings = []
        self.thread = None

    @property
    def pending(self):
        with self._condition:
            return len(self._jobs) + int(self._preview is not None) + int(self._working)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self._stopped = False
        self.thread = threading.Thread(target=self._run, name="PrismTranslation", daemon=True)
        self.thread.start()

    def submit(self, entry_id, text):
        with self._condition:
            if self._stopped:
                return False
            revision = self._revisions.get(entry_id, 0) + 1
            self._revisions[entry_id] = revision
            self._finalized.add(entry_id)
            if isinstance(entry_id, tuple):
                # A growing caption includes all earlier confirmed text. Keep
                # just its newest pending translation, rather than translate
                # obsolete prefixes on a slower local generative model.
                self._jobs = deque(job for job in self._jobs if not
                    (isinstance(job[0], tuple) and job[0][0] == entry_id[0]))
            if self._preview and self._preview[0] == entry_id:
                self._preview = None
            self._jobs.append((entry_id, text, revision, time.perf_counter()))
            self._condition.notify_all()
            return True

    def submit_preview(self, entry_id, text):
        if getattr(self.translator, "confirmed_only", False):
            return False
        if not str(text).strip():
            return False
        with self._condition:
            if self._stopped or entry_id in self._finalized:
                return False
            if self._preview and self._preview[:2] == (entry_id, text):
                return False
            revision = self._revisions.get(entry_id, 0) + 1
            self._revisions[entry_id] = revision
            # Replace rather than queue intermediate English hypotheses.
            self._preview_sources[entry_id] = text
            while len(self._preview_sources)>256:
                self._preview_sources.pop(next(iter(self._preview_sources)))
            self._preview = (entry_id, text, revision, time.perf_counter())
            self._condition.notify_all()
            return True

    def stop(self):
        with self._condition:
            self._stopped = True
            self._jobs.clear()
            self._preview = None
            self._sentence_cache.clear()
            self._condition.notify_all()
        cancel = getattr(self.translator, "cancel", None)
        if callable(cancel):
            cancel()

    def _translate_confirmed(self, entry_id, text):
        """Reuse unchanged coherent units; a revised active tail is new.

        A cache hit requires identical source text within the same speaker card.
        No guessed target-prefix alignment, hidden truncation or disk cache is
        used. Formula protection applies to each translated unit independently.
        """
        self._last_final_warnings = []
        if getattr(self.translator, "context_sensitive", False):
            protected = protect_translation_source(text)
            translated = protected.restore(self.translator.translate(protected.text))
            self._last_final_warnings.extend(getattr(self.translator, "last_warnings", []))
            return translated, 0, 1
        sentences = _coherent_translation_units(text)
        card = entry_id[0] if isinstance(entry_id, tuple) and entry_id else entry_id
        if (callable(getattr(self.translator, "translate_batch", None))
                or callable(getattr(self.translator, "translate_batch_with_context", None))):
            return self._translate_batched(card, sentences)
        output, hits, calls = [], 0, 0
        for sentence in sentences:
            key = (card, sentence)
            with self._condition:
                if self._stopped:
                    return None, hits, calls
                cached = self._sentence_cache.get(key)
                if cached is not None:
                    self._sentence_cache.move_to_end(key)
            if cached is not None:
                output.append(cached)
                hits += 1
                continue
            protected = protect_translation_source(sentence)
            translated = protected.restore(self.translator.translate(protected.text))
            self._last_final_warnings.extend(getattr(self.translator, "last_warnings", []))
            calls += 1
            output.append(translated)
            with self._condition:
                if not self._stopped:
                    self._sentence_cache[key] = translated
                    while len(self._sentence_cache) > self._maximum_cached_sentences:
                        self._sentence_cache.popitem(last=False)
        return " ".join(output), hits, calls

    def _translate_batched(self, card, sentences):
        """Translate only uncached coherent units in bounded cloud batches.

        Preserve unchanged Chinese; growing source clauses are new inputs.
        Batch index/count validation is the translator's contract. Formula
        restoration must succeed for the whole batch before anything is cached.
        """
        output = [None] * len(sentences)
        missing, hits, calls = [], 0, 0
        contextual = callable(getattr(self.translator, "translate_batch_with_context", None))
        preceding = []
        with self._condition:
            if self._stopped:
                return None, 0, 0
            for index, sentence in enumerate(sentences):
                context = _bounded_preceding_context(" ".join(preceding)) if contextual else ""
                key = (card, context, sentence) if contextual else (card, sentence)
                preceding.append(sentence)
                cached = self._sentence_cache.get(key)
                if cached is None:
                    missing.append((index, sentence, protect_translation_source(sentence), context, key))
                else:
                    output[index] = cached
                    hits += 1
                    self._sentence_cache.move_to_end(key)
        while missing:
            batch, size = [], 0
            while missing and (not batch or (len(batch) < 8 and size + len(missing[0][2].text)
                                            + len(missing[0][3]) <= 2000)):
                item = missing.pop(0)
                batch.append(item)
                size += len(item[2].text) + len(item[3])
            with self._condition:
                if self._stopped:
                    return None, hits, calls
            if contextual:
                translated = self.translator.translate_batch_with_context(
                    [item[2].text for item in batch], [item[3] for item in batch])
            else:
                translated = self.translator.translate_batch([item[2].text for item in batch])
            if len(translated) != len(batch):
                raise ValueError("翻译分段数量不符，未显示错位译文。")
            restored = [item[2].restore(zh) for item, zh in zip(batch, translated)]
            calls += 1
            with self._condition:
                if self._stopped:
                    return None, hits, calls
                for (index, _, _, _, key), zh in zip(batch, restored):
                    output[index] = zh
                    self._sentence_cache[key] = zh
                while len(self._sentence_cache) > self._maximum_cached_sentences:
                    self._sentence_cache.popitem(last=False)
        return " ".join(output), hits, calls

    def _run(self):
        while True:
            with self._condition:
                while True:
                    if self._stopped:
                        return
                    if self._jobs:
                        entry_id, text, revision, submitted = self._jobs.popleft()
                        kind = "final"
                        break
                    if self._preview:
                        delay = self._last_preview_started + self.preview_interval - time.monotonic()
                        if delay > 0:
                            self._condition.wait(timeout=delay)
                            continue
                        entry_id, text, revision, submitted = self._preview
                        self._preview = None
                        self._last_preview_started = time.monotonic()
                        kind = "preview"
                        break
                    self._condition.wait(timeout=.5)
                self._working = True
            start = time.perf_counter()
            try:
                if kind == "final":
                    translated, cache_hits, model_calls = self._translate_confirmed(entry_id, text)
                else:
                    protected = protect_translation_source(text)
                    translated = protected.restore(self.translator.translate(protected.text))
                    cache_hits, model_calls = 0, 1
                with self._condition:
                    compatible_prefix = (kind == "preview" and self.allow_prefix_preview
                                         and self._preview_sources.get(entry_id, "").startswith(text))
                    current = (translated is not None and not self._stopped and (self._revisions.get(entry_id) == revision or compatible_prefix)
                               and (kind == "final" or entry_id not in self._finalized))
                if current:
                    if kind == "preview":
                        self.on_preview(entry_id, text, translated)
                    else:
                        self.on_result(entry_id, translated)
                        for warning in self._last_final_warnings:
                            self.on_status("翻译提示：" + warning)
                self.on_metrics({"translation_seconds": round(time.perf_counter()-start, 3),
                                 "translation_queue_seconds": round(start-submitted, 3),
                                 "translation_cached_sentences": cache_hits,
                                 "translation_model_calls": model_calls,
                                 "translation_kind": kind, "translation_result_current": current})
            except Exception as exc:
                if kind == "final":
                    label = getattr(self.translator, "engine_label", "本地翻译")
                    self.on_status(label+'未完成或未通过校验，请参照英文原话：'+str(exc))
                else:
                    self.on_metrics({"translation_kind": "preview", "translation_preview_error": True})
            finally:
                with self._condition:
                    self._working = False
                    self._condition.notify_all()

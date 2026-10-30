"""Local interview intent, question structure, and precomputed document lookup.

No cloud request, training, inferred authorship or generated paper summary.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import re
import threading

_SELF_EXPLANATION = re.compile(r"^(?:I(?:'m| am)\s+not\s+sure|I\s+(?:think|used|use|would\s+start)|"
                               r"we\s+(?:used|use|can)|yes\b|no\b|got\s+it\b|that\s+(?:sounds|makes))",re.I)
_ANSWER_CUE=re.compile(r"\bI(?:'m| am)\s+not\s+sure\b|\bbut\s+I\s+think\s+we\s+can\b",re.I)
_CONFIRMATION_TAG = re.compile(r"(?:,\s*(?:right|okay|ok)|^(?:okay|ok|right|all right|got it|understood))\s*\?$", re.I)
_NEW_QUESTION = re.compile(
    r"^(?:(?:okay|ok|right|well|so)[, .:]+)*(?:now[, ]+)?"
    r"(?:(?:(?:a|the|my|our)\s+)?(?:different|next|another)\s+question\b|"
    r"let(?:['’]s|\s+us)\s+(?:move|go)\s+on\s+to\s+(?:another|a\s+different|the\s+next)\s+question\b)", re.I)


def starts_new_question(text: str) -> bool:
    """Only an explicit topic boundary; ordinary 'and/how about' is continuation."""
    return bool(_NEW_QUESTION.search(str(text).strip()))


def is_interview_request(text: str) -> bool:
    """Shared local gate for explicit requests, without model classification.

    A period is common in polite requests. Requiring a question mark would miss
    'I would be interested in your limitations' and 'Talk us through ...'.
    This is still a conservative language heuristic, not complete understanding.
    """
    from .transcript_text import repair_fragment_boundaries, split_sentences
    from .turns import starts_direct_question, is_non_request_context
    value = repair_fragment_boundaries(str(text).strip())
    return bool(value and len(value) >= 4 and not is_non_request_context(value) and (
        '?' in value or any(starts_direct_question(part.strip()) for part in split_sentences(value))
        or _shared_explicit_parts(value)))


def _shared_explicit_parts(text: str) -> tuple[str, ...]:
    from .turns import explicit_question_parts
    return explicit_question_parts(text)


def _confirmation_only(sentence: str) -> bool:
    """A statement's conversational tag, never an explicit direct question."""
    from .turns import starts_direct_question
    return not starts_direct_question(sentence.strip()) and bool(_CONFIRMATION_TAG.search(sentence.strip()))


def _question_sentences(sentences) -> list[str]:
    from .turns import starts_direct_question, is_non_request_context
    asks = [part for part in sentences if not is_non_request_context(part) and
            (starts_direct_question(part.strip()) or '?' in part)]
    # "We take turns, okay?" supplies setup for the later addressed ask; it
    # should not become an additional item on the model's question checklist.
    # Without that later direct ask, retain technical confirmation questions.
    if any(starts_direct_question(part.strip()) for part in asks):
        asks = [part for part in asks if not _confirmation_only(part)]
    return asks


def question_parts(text: str) -> list[str]:
    """A deterministic checklist, not an assertion of semantic completeness."""
    from .transcript_text import repair_fragment_boundaries, split_sentences
    from .turns import starts_direct_question, is_non_request_context
    text = repair_fragment_boundaries(text)
    if is_non_request_context(text):
        return []
    sentences = [sentence for line in text.splitlines()
                 for sentence in split_sentences(line) if sentence.strip()]
    asks = _question_sentences(sentences)
    if (asks and all(_confirmation_only(part) for part in asks)
            and not _RESEARCH_TOPIC.search(text)):
        return []
    items=[]
    for sentence in asks:
        # A second addressed question may arrive without restored punctuation.
        # Split only an explicit discourse conjunction + question opening;
        # ordinary technical conjunctions and decimal values stay intact.
        boundaries=list(re.finditer(
            r"\s+(?:and|also|so)\s+(?=(?:how|why|which|what|where|when|who|whose)\b|"
            r"(?:can|could|would|do|did|have)\s+you\b|"
            r"(?:please\s+)?(?:explain|describe|discuss|clarify|compare|tell\s+(?:me|us)|"
            r"(?:walk|talk)\s+(?:me|us)\s+through)\b)",sentence,flags=re.I))
        # 'How and why ...' is one coordinated opening, not an orphan 'How'.
        cuts=[match for match in boundaries if len(sentence[:match.start()].split())>=3
              and not re.search(r'\b(?:how|why|which|what|where|when|who|whose)\s*$',sentence[:match.start()],re.I)]
        parts=[];start=0
        for match in cuts:
            parts.append(sentence[start:match.start()]);start=match.end()
        parts.append(sentence[start:])
        items.extend(s.strip() for s in parts if starts_direct_question(s.strip()) or '?' in s)
    return items or ([text.strip()] if text.strip() else [])


@dataclass(frozen=True)
class QuestionStructure:
    """A local routing/display aid, never a replacement for the source speech."""
    background_en: str
    answer_question_en: str
    interaction_kind: str


_TURN_CHOICE = re.compile(
    r"^(?:(?:so|okay|ok|well|then)[, ]+)*"
    r"(?:(?:do|would) you (?:want|like|prefer) to (?:start|begin|go first),? or "
    r"(?:(?:do|would) you (?:want|like|prefer) me to (?:start|begin|go first)|"
    r"(?:should|shall|can) I (?:start|begin|go first))|"
    r"(?:should|shall|can) I (?:start|begin|go first),? or "
    r"(?:do|would) you (?:want|like|prefer) to (?:start|begin|go first)|"
    r"(?:do|would) you (?:want|like|prefer) me to (?:start|begin|go first),? or "
    r"(?:do|would) you (?:want|like|prefer) to (?:start|begin|go first))"
    r"[.?!]*$", re.I)
_RESEARCH_TOPIC = re.compile(
    r"\b(?:spectral|polarimetric|polarization|Stokes|ghost imaging|inverse rendering|"
    r"reflectance|scattering|attenuation|calibrat\w*|instrument matrix|sensing operator|"
    r"reconstruct\w*|neural network|machine learning|training data|loss function|"
    r"experiment\w*|hypothes\w*|uncertainty|optical|algorithm\w*|doctoral|research|"
    r"object detection|model performance|metrics?|benchmark\w*|precision|recall|"
    r"accuracy|computational|imaging|projects?|teamwork|teams?|collaborat\w*|"
    r"clients?|customers?|budgets?|leadership|lead\w*|management|manag\w*|"
    r"sales|marketing|operations?|supply chain|finance|revenue|costs?|"
    r"patients?|clinical|healthcare|teaching|curriculum|students?|"
    r"skills?|experience|responsibilit\w*|priorit\w*|deadlines?|delivery|"
    r"design|engineering|product\w*|quality|risks?|strategy|process\w*)\b", re.I)


def analyze_question(text: str) -> QuestionStructure:
    """Separate explicit asks from setup; only narrow turn choices route locally.

    Research content is not discarded: ``text`` remains the complete question
    in the packet, and background is supplied alongside the focus to the model.
    Without an explicit ask the whole utterance remains the focus. This avoids
    pretending a heuristic understands every indirect interview question.
    """
    from .transcript_text import repair_fragment_boundaries, split_sentences
    from .turns import is_non_request_context
    text = repair_fragment_boundaries(text)
    if is_non_request_context(text):
        return QuestionStructure(text, "", "setup")
    sentences = split_sentences(text)
    asks = _question_sentences(sentences)
    if (asks and all(_confirmation_only(part) for part in asks)
            and not _RESEARCH_TOPIC.search(text)):
        return QuestionStructure(text, "", "setup")
    if not asks:
        return QuestionStructure("", text,
                                 "research" if _RESEARCH_TOPIC.search(text) else "general")
    focus = " ".join(asks)
    background = " ".join(part for part in sentences if part not in asks)
    kind = ("coordination" if len(asks) == 1 and _TURN_CHOICE.fullmatch(focus)
            else "research" if _RESEARCH_TOPIC.search(text) else "general")
    return QuestionStructure(background, focus, kind)


class InterviewIntent:
    def __init__(self):
        self.roles={}
        self._question_speakers=set()
        self._responding_speakers=set()
        self._response_active=False
        self._previous_tail=''

    def reset(self):
        self.roles.clear(); self._question_speakers.clear();self._responding_speakers.clear()
        self._response_active=False
        self._previous_tail=''

    def begin_question(self):
        """An explicit phase/new-question boundary retires inferred response state."""
        self._response_active=False
        self._previous_tail=''
        self._responding_speakers.clear()

    def set_role(self,speaker_id,role):
        if speaker_id and role in ('auto','interviewer','ignore'):
            self.roles[str(speaker_id)]=role

    def accepts(self,text,speaker_id=None):
        from .turns import starts_direct_question, next_addressed_question_offset, is_non_request_context
        boundary_text=self._previous_tail+' '+text
        boundary=len(self._previous_tail)+1
        crossed=any(m.start()<boundary and m.end()>boundary for m in _ANSWER_CUE.finditer(boundary_text))
        self._previous_tail=text[-80:]
        identity=str(speaker_id or '')
        role=self.roles.get(identity,'auto')
        if role=='ignore':return False
        if role=='interviewer':return True
        if is_non_request_context(text):return False
        # A candidate explaining their answer can contain quoted/interior
        # 'what' questions. Do not promote those into a new interviewer turn.
        if (_SELF_EXPLANATION.search(text.strip())
                or _ANSWER_CUE.search(text) or crossed):
            if identity:self._responding_speakers.add(identity)
            self._response_active=True
            return False
        if self._response_active:
            if next_addressed_question_offset(text.strip()) == 0:
                # An unmistakably addressed new question can recover after an
                # acoustic cluster collision. Generic interior 'what's the
                # probability' fragments cannot change an answer into a query.
                self._response_active=False
                self._responding_speakers.discard(identity)
            elif (not identity or identity in self._responding_speakers
                  or identity not in self._question_speakers):
                return False
        if identity in self._question_speakers:return True
        if identity and identity in self._responding_speakers:return False
        if starts_direct_question(text.strip()):
            if identity:self._question_speakers.add(identity)
            return True
        # Anonymous/changed acoustic labels are not a semantic boundary. Keep
        # ordinary fragments of the active question until an answer cue or an
        # explicit role says otherwise; short crops often lack usable identity.
        return not self._response_active


class InterviewPreparation:
    # Neutral cache seeds. A user's actual question and supplied profile drive retrieval.
    TOPICS=('experience responsibilities contribution project results',
            'methods decisions alternatives limitations validation',
            'collaboration challenges goals motivation transferable skills')

    def __init__(self):
        self._lock=threading.RLock();self._cache=OrderedDict()
        self.overview='';self.document_count=0;self.passage_count=0
        self._index_signature=None
        self._documents=()
        self._state='empty'

    def snapshot(self):
        """Only bounded navigation metadata; no paper text, path or evidence.

        Prepared means the local index/cache is ready. It never means a cloud
        model has learned the documents or verified the candidate's authorship.
        """
        with self._lock:
            return {'documents':list(self._documents),
                    'document_count':self.document_count,
                    'passage_count':self.passage_count,'state':self._state}

    @staticmethod
    def _signature(kb):
        return (id(kb), getattr(kb,'_revision',None), id(kb.passages), len(kb.passages))

    @staticmethod
    def _key(question,top_k):
        return (re.sub(r'\W+',' ',question.casefold()).strip(),int(top_k))

    def prepare(self,kb):
        signature=self._signature(kb)
        names=list(kb.document_names)
        count=len(kb.passages)
        # Filenames are navigation context, never evidence of personal results.
        documents=tuple(Path(name.replace('\\','/')).name[:160] for name in names)
        overview='\n'.join(documents)[:1600]
        cache=OrderedDict()
        for topic in self.TOPICS if count else ():
            cache[self._key(topic,5)]=list(kb.search(topic,top_k=5))
        with self._lock:
            if signature != self._signature(kb):
                return '资料正在更新 · 下次提问将使用最新索引'
            self._cache=cache;self.overview=overview
            self._index_signature=signature
            self.document_count=len(names);self.passage_count=count
            self._documents=documents[:30]
            self._state='ready' if count else 'empty'
        return f'已准备 {len(names)} 份资料 · {count} 段可检索内容'

    def search(self,kb,question,top_k=5):
        key=self._key(question,top_k)
        signature=self._signature(kb)
        with self._lock:
            if signature!=self._index_signature:
                self._cache.clear();self._index_signature=signature
                self._state='updating'
            if key in self._cache:
                self._cache.move_to_end(key)
                return list(self._cache[key])
        result=list(kb.search(question,top_k=top_k))
        with self._lock:
            if signature==self._signature(kb):
                self._cache[key]=result
            while len(self._cache)>24:self._cache.popitem(last=False)
        return result

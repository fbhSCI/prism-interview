"""Question assembly independent of caption segmentation and GUI edit state."""
from __future__ import annotations
import time
import uuid
import re


_ORAL_REQUEST_BODY = (
    r"(?:I\s+(?:was\s+)?(?:(?:just|actually|kind\s+of)\s+)*(?:wondering|wonder)\s+"
    r"(?:like[, ]+)?(?:if|whether|how|why|what|which|where|when|about)|"
    r"I(?:['’]m|\s+am)\s+(?:just\s+)?curious\s+(?:about|if|whether|how|why|what|which|where|when)|"
    r"I(?:['’]d|\s+would)\s+be\s+interested\s+in\s+(?:your|how|why|what|whether)|"
    r"I(?:['’]m|\s+am)\s+(?:particularly\s+)?interested\s+in\s+(?:your|how|why|what|whether)|"
    r"I\s+(?:wanted|want)\s+to\s+(?:ask|know|understand)|"
    r"I(?:['’]d|\s+would)\s+like\s+to\s+(?:ask|know|understand|hear)|"
    r"I(?:['’]d|\s+would)\s+(?:like|appreciate)\s+your\s+"
    r"(?:thoughts|comments|suggestions|insights)\s+(?:on|about)|"
    r"(?:any|your)\s+(?:thoughts|comments|suggestions|insights)\s+(?:on|about))\b")
_ORAL_REQUEST = re.compile(
    r"^(?:(?:so|and|also|then|well|okay|ok|now|but)[, :]+)*" + _ORAL_REQUEST_BODY, re.I)
_STATED_REQUEST = re.compile(
    r"^(?:(?:so|and|also|then|well|okay|ok|now|but)[, :]+)*"
    r"(?:the|my|our)\s+question\s+(?:here\s+)?is\s+"
    r"(?:how|why|what|which|whether|if|where|when)\b", re.I)
_REQUEST_FRAME = re.compile(
    r"^(?:(?:so|and|also|then|well|okay|ok|now|but)[, :]+)*(?:"
    r"(?:please\s+)?(?:explain|describe|discuss|clarify|compare|tell\s+(?:me|us)(?:\s+about)?|"
    r"(?:walk|talk)\s+(?:me|us)\s+through)|"
    r"I(?:['’]d|\s+would)\s+like\s+to\s+(?:ask|know|understand|hear)|"
    r"I(?:['’]d|\s+would)\s+(?:like|appreciate)\s+your\s+"
    r"(?:thoughts|comments|suggestions|insights)\s+(?:on|about)|"
    r"I(?:['’]d|\s+would)\s+be\s+interested\s+in|"
    r"I(?:['’]m|\s+am)\s+(?:(?:just|particularly)\s+)?(?:curious(?:\s+about)?|interested\s+in)|"
    r"I\s+(?:was\s+)?(?:just\s+)?(?:wondering|wonder)(?:\s+like)?|"
    r"(?:the|my|our)\s+question\s+(?:here\s+)?is)\b", re.I)
_EMBEDDED_REQUEST = re.compile(r"\b(?:" + _ORAL_REQUEST_BODY +
    r"|(?:the|my|our)\s+question\s+(?:here\s+)?is\s+(?:how|why|what|which|whether|if))", re.I)
_TOPIC_FOLLOWUP = re.compile(
    r"^(?:(?:so|and|also|then|well|okay|ok|now|but)[, :]+)*(?:"
    r"(?:what|how)\s+about|(?:any|your)\s+(?:thoughts|comments|suggestions|insights)\s+(?:on|about))\b", re.I)
_SHORT_COMPLETE_REQUEST = re.compile(
    r"^(?:(?:so|and|also|then|well|okay|ok|now|but)[, :]+)*(?:"
    r"(?:(?:why|how|when|where)\s+)?(?:do|does|did|can|could|would|will|should|might)\s+"
    r"(?:it|they|this|that|these|those|the\s+\w+)\s+"
    r"(?:work|generalize|converge|scale|fail|matter|differ|help|hold|apply)|"
    r"what\s+(?:do|did|would)\s+you\s+(?:think|suggest|recommend|mean))\s*[?!.]*$", re.I)

# Source-preserving turn cues. These are deliberately narrower than the UI's
# question detector: a tag such as "right?" is not enough to close a turn.
_DIRECT_ASK = re.compile(
    r"^(?:(?:so|and|also|then|well|okay|ok|now|but)[, :]+)*(?:"
    r"(?:do|did|does|can|could|would|will|should|have|has|are|was|were|may|might|must)\s+"
    r"(?:you|we|I|he|she|it|they|the|this|that|these|those|a|an|our|your|their|each|both)\b|"
    r"(?:is|are|does|has)\s+(?!really\b|very\b|quite\b|also\b|still\b)\w+\b|"
    r"(?:what|why|how|which|where|when|who|whose)\b|"
    r"(?:please\s+)?(?:explain|describe|discuss|clarify|compare|tell\s+(?:me|us)|"
    r"(?:walk|talk)\s+(?:me|us)\s+through)\b|"
    r"I(?:['’]d|\s+would)\s+like\s+to\s+(?:know|understand|hear)\b)", re.I)
_MARKED_AUXILIARY_ASK = re.compile(
    r"^(?:(?:so|and|also|then|well|okay|ok|now|but)[, :]+)*"
    r"(?:do|did|does|can|could|would|will|should|have|has|is|are|was|were|may|might|must)\s+\w+\b", re.I)
_EMBEDDED_INVERTED_ASK = re.compile(
    r"\b(?:what|why|how|which|where|when|who)\s+"
    r"(?:do|did|does|can|could|would|will|should|have|has|is|are|was|were|may|might|must)\b", re.I)
_REPORTED_ASK = re.compile(r"\b(?:asks|asked|question\s+is)\s*[:,]?\s+", re.I)
_THANKS_FOR_TALK = re.compile(
    r"\bthank\s+you\s+for\s+(?:the|your|this|a)\s+"
    r"(?:(?:great|good|fantastic|wonderful|interesting|excellent|amazing)\s+)*"
    r"(?:talk|presentation|seminar|lecture)[, :;\s]*$", re.I)
_MODERATOR_NAME = r"[A-ZÀ-ÖØ-Þ][^\W\d_]*(?:[-’'][A-ZÀ-ÖØ-Þ]?[^\W\d_]*)*"
_MODERATOR_PREFIX = re.compile(
    r"^(?:(?i:so|and|then)[, ]+)?(?:(?:" + _MODERATOR_NAME + r"[, ]+){1,8}"
    r"(?i:asks|asked)[:, ]+)?(?:(?i:first)[, ]+)?"
    r"(?:(?i:thank\s+you\s+for\s+(?:the|your|this|a)\s+"
    r"(?:(?:great|good|fantastic|wonderful|interesting|excellent|amazing)\s+)*"
    r"(?:talk|presentation|seminar|lecture))[.!?,;:\s]*)?$")
_QUESTION_INTRO = re.compile(
    r"^(?:(?:okay|ok|right|well|so|now|yes|yeah|yep)[, .:]+)*"
    r"(?:[A-Z][a-z]{1,24},\s*)?(?:I(?:['’]ve|\s+have)\s+(?:got\s+)?|I\s+got\s+)?"
    r"(?:(?:one|a|another|my|the\s+next)\s+(?:(?:quick|brief|related|follow-up)\s+)?question\b|"
    r"(?:my|a|the)\s+(?:next|different)\s+question\b)", re.I)
_QUESTION_INTRO_PREFIX = re.compile(
    r"^(?:(?:okay|ok|right|well|so|now|yes|yeah|yep)[, .:]+)*"
    r"I(?:['’]ve|\s+have)?(?:\s+got)?(?:\s+(?:a|one|another))?"
    r"(?:\s+(?:quick|brief|related|follow-up))?[, ]*$", re.I)
_ACK = re.compile(
    r"^(?:(?:yes|yeah|yep|sure|certainly|absolutely|okay|ok|right|"
    r"all\s+right|I\s+see|got\s+it|understood|makes\s+sense|that\s+makes\s+sense)"
    r"[.,!\s]*)+$", re.I)
_RESPONSE_START = re.compile(
    r"^(?:(?:oh|well)[, ]+)*(?:(?:yes|yeah|yep|sure|certainly|absolutely)[, .!]+|"
    r"(?:(?:it['’]s|it\s+is|that['’]s|that\s+is)\s+)+(?:a\s+)?(?:great|good|interesting)\s+question\b)", re.I)
_SELF_CORRECTION = re.compile(r"^(?:(?:no|yes|yeah|sure|sorry|actually)[, .!]+)*"
    r"(?:I\s+(?:mean|meant)|let\s+me\s+(?:rephrase|correct|clarify)|"
    r"what\s+I\s+(?:mean|meant))\b", re.I)
# A greeting directed back to the questioner is stronger than a bare yes/yeah.
# ASR hard cuts can omit BOTH the question stop and the stop before this reply.
# Require the entire social formula; do not split ordinary acknowledgments,
# generic 'hi', academic keywords, names, or quoted examples by themselves.
_SOCIAL_RESPONSE = re.compile(
    r"(?<!\S)(?:(?:oh|well)[, ]+)*(?:(?:yes|yeah|yep|sure|certainly|absolutely)[, .!]+)*"
    r"(?:hi|hello)(?:[, ]+[\w’'-]+){0,4}[, ]+"
    r"(?:it(?:['’]s|\s+is)\s+)?(?:good|nice|great|lovely)\s+to\s+"
    r"(?:see|meet|hear)(?:\s+from)?\s+you\b", re.I)
_QUOTED_SPEECH = re.compile(r'''"[^"]*"|“[^”]*”|‘[^’]*’|(?<!\w)'[^\n]*?'(?!\w)''')
_REPORTED_CONTEXT = re.compile(
    r"\b(?:said|says|wrote|writes|quoted|quotes|quoting|read\s+out|the\s+phrase|"
    r"ask(?:ed)?\s+(?:myself|ourselves))\b", re.I)
_SELF_ANSWER_CUE = re.compile(
    r"^(?:(?:so|and|well)[, ]+)*(?:"
    r"(?:this|that)\s+question\s+(?:motivated|inspired|guided|drove|led\s+to)\s+(?:my|our|the)|"
    r"(?:the|my|our)\s+answer\s+is\b|"
    r"(?:I|we)\s+(?:will|shall)\s+(?:now\s+)?(?:show|explain|answer|demonstrate)\b|"
    r"let\s+me\s+answer\s+(?:that|this|my\s+own)\s+question\b)", re.I)
# In an answer, 'what you need is ...' and 'when you do ...' are often
# declarative clauses, unlike inverted asks ('what do you need?'). A fresh
# addressed question must not be inferred merely from their pronoun 'you'.
_DECLARATIVE_WH = re.compile(
    r"^(?:(?:so|and|also|then|well|okay|ok|now|but)[, :]+)*"
    r"(?:what|how|when|where|why|which)\s+(?:you|we|I|he|she|they|it|"
    r"(?:the|your|our|this|that)\s+\w+)\b", re.I)
# A deictic + stance marker is only a tentative reply onset. It becomes a
# boundary after a later explicit first-person explanatory clause confirms it.
# Bare 'honestly', punctuation, timing or topical nouns cannot seal a question.
_TENTATIVE_EXPLANATION = re.compile(
    r"(?<!\S)(?:this|that)[, ]+(?:honestly|frankly|basically|actually)\b", re.I)
_EXPLANATION_CLAUSE = re.compile(
    r"\b(?:what\s+(?:we|I)\s+(?:have|do|use|need)\s+(?:here\s+)?is|"
    r"(?:we|I)\s+(?:(?:already|actually|currently|normally|typically|usually|"
    r"kind\s+of|sort\s+of)\s+)*(?:have|use|do|compute|estimate|measure|model))"
    r"(?=\s+\w+)", re.I)
_QUALIFYING_CONTINUATION = re.compile(
    r"\b(?:if|whether|when|where|how|why|while|unless|until|before|after|because|since|"
    r"given|suppose|assuming|imagine|I\s+(?:mean|meant)|"
    r"let\s+me\s+(?:rephrase|correct|clarify))\b", re.I)

_INVITE_COUNT = r"(?:(?:any|some|a|another|one|more|further|other|additional)\s+)*"
_INVITE_SUFFIX = (r"(?:\s+(?:for|from|about|on|to)\s+[^.!?]+)?"
                  r"(?:\s+(?:please|now))?(?:[,;]\s*(?:thank\s+you|thanks))?[.!?,;:\s]*")
_INVITATION = re.compile(
    r"^(?:(?:okay|ok|so|now|well|right|then|and|yes)[, :]+)*(?:"
    r"questions?|any\s+" + _INVITE_COUNT + r"questions?|"
    r"(?:do|does|did)\s+(?:you|we|anyone|anybody|someone|somebody|the\s+audience)\s+"
        r"have\s+" + _INVITE_COUNT + r"questions?|"
    r"(?:have|has)\s+(?:you|anyone|anybody)\s+got\s+" + _INVITE_COUNT + r"questions?|"
    r"(?:are|is)\s+there\s+" + _INVITE_COUNT + r"questions?|"
    r"(?:would|does|will)\s+(?:anyone|anybody|someone|somebody)\s+(?:like|want)\s+to\s+"
        r"ask\s+" + _INVITE_COUNT + r"questions?|"
    r"(?:can|could|may|shall)\s+(?:I|we)\s+(?:ask|take|have|hear)\s+" + _INVITE_COUNT + r"questions?|"
    r"(?:let(?:['’]s|\s+us)\s+)?open\s+the\s+floor\s+(?:for|to)\s+questions|"
    r"(?:we\s+have\s+)?time\s+for\s+" + _INVITE_COUNT + r"questions?|"
    r"(?:I|we)\s+(?:will|can)\s+(?:now\s+)?take\s+" + _INVITE_COUNT + r"questions?|"
    r"I(?:['’]ll|\s+will|\s+would)\s+be\s+happy\s+to\s+take\s+" + _INVITE_COUNT + r"questions?|"
    r"(?:I(?:['’]m|\s+am)?|we(?:['’]re|\s+are)?)\s+(?:just\s+)?(?:going\s+to|gonna)\s+"
        r"(?:forward|read|relay)\s+(?:(?:the|some|any|a|one|YouTube|chat|online|audience|submitted)\s+)*questions?|"
    r"(?:who\s+has|who\s+would\s+like\s+to\s+ask)\s+" + _INVITE_COUNT + r"questions?|"
    r"(?:can|could|would)\s+you\s+(?:ask|repeat|state)\s+your\s+question|"
    r"(?:I\s+think[, ]+)?(?:[A-Z][a-z]{1,24}[, ]+)?you\s+have\s+" + _INVITE_COUNT + r"questions?|"
    r"(?:please\s+)?(?:feel\s+free\s+to\s+)?ask\s+" + _INVITE_COUNT + r"questions?"
    r")" + _INVITE_SUFFIX + r"$", re.I)
_INVITATION_PREFIX = re.compile(
    r"^(?:(?:okay|ok|so|now|well|right|then|yes)[, :]+)*(?:"
    r"(?:do|does|did)\s+(?:you|we|anyone|anybody|someone|somebody|the\s+audience)(?:\s+have)?|"
    r"(?:have|has)\s+(?:you|anyone|anybody)\s+got|(?:are|is)\s+there|"
    r"(?:can|could|may)\s+(?:I|we)\s+(?:ask|take|have|hear)|any|"
    r"(?:I(?:['’]m|\s+am)?|we(?:['’]re|\s+are)?)\s+(?:just\s+)?(?:going(?:\s+to)?|gonna)"
        r"(?:\s+(?:forward|read|relay)(?:\s+(?:the|some|any|a|one|YouTube|chat|online|audience|submitted))*)?)"
    r"(?:\s+(?:any|some|a|another|one|more|further|other|additional))*[,.?!\s]*$", re.I)
_INVITATION_CUE = re.compile(r"\b(?:any\s+(?:(?:more|other|further)\s+)?questions?|"
    r"(?:do|does|did)\s+(?:you|we|anyone|anybody)\s+have|(?:are|is)\s+there|"
    r"(?:would|does)\s+(?:anyone|anybody)\s+(?:like|want)|"
    r"(?:can|could|may)\s+(?:I|we)\s+(?:ask|take)|I['’]ll\s+be\s+happy\s+to\s+take)\b", re.I)


def is_question_invitation(text: str) -> bool:
    """A request to take/ask questions, rather than the technical question.

    Narrow language families preserve substantive uses of 'question', such as
    'Which scientific questions remain open?' and 'Do you have any evidence?'.
    Automatic answering ignores invitations; the manual button remains usable.
    """
    return bool(_INVITATION.fullmatch(str(text).strip()))


def invitation_prefix_offset(text: str) -> int | None:
    spans = _speech_sentences(str(text))
    return spans[-1].start() if spans and _INVITATION_PREFIX.fullmatch(spans[-1].group().strip()) else None


def invitation_segments(text: str) -> list[tuple[bool, str]]:
    result = []
    start = 0
    for match in _speech_sentences(str(text)):
        sentence = match.group()
        begin = match.start() if is_question_invitation(sentence) else None
        if begin is None:
            # Whisper can miss the stop between a compliment and an invitation.
            # Match only a complete invitation suffix, never arbitrary words
            # preceding a technical question or a fragment like 'any evidence'.
            begin = next((match.start() + cue.start() for cue in _INVITATION_CUE.finditer(sentence)
                          if is_question_invitation(sentence[cue.start():])), None)
        if begin is not None:
            if text[start:begin].strip():
                result.append((False, text[start:begin].strip()))
            result.append((True, text[begin:match.end()].strip()))
            start = match.end()
    if text[start:].strip():
        result.append((False, text[start:].strip()))
    return result


def announces_question(text: str) -> bool:
    """An explicit named question introduction; do not infer a new topic."""
    return bool(_QUESTION_INTRO.search(str(text).strip()))


def question_introduction_prefix(text: str) -> str:
    """Hold only a bounded incomplete 'I have/got a ...' across ASR crops."""
    sentences = _speech_sentences(str(text))
    tail = sentences[-1].group().strip() if sentences else ''
    return tail if len(tail) <= 100 and (_QUESTION_INTRO_PREFIX.fullmatch(tail) or
        re.fullmatch(r"(?:the|my|our)\s+question(?:\s+here)?(?:\s+is)?[,:. ]*", tail, re.I)) else ''


def starts_direct_question(text: str) -> bool:
    value = str(text).strip()
    # Explicit question punctuation admits arbitrary subjects ('Can spectral
    # measurements ...?'), while unmarked fragments still use narrower cues.
    return bool(_DIRECT_ASK.search(value) or _ORAL_REQUEST.search(value) or _STATED_REQUEST.search(value) or
                (value.endswith('?') and _MARKED_AUXILIARY_ASK.search(value)))


def acknowledgment_only(text: str) -> bool:
    return bool(_ACK.fullmatch(str(text).strip()))


def _speech_sentences(text: str):
    # Preserve exact source offsets. Decimal points are not sentence boundaries.
    sentences = list(re.finditer(
        r'\S[\s\S]*?(?:[?!]+|(?<!\d)\.(?!\d)|[,;:](?=\s+'
        r'(?:do|did|can|could|would|will|should|have)\s+you\b)|'
        r'(?=\s+(?:and|also)\s+' + _ORAL_REQUEST_BODY + r'))(?=\s|$)|\S[\s\S]*$', text, re.I))
    result = []
    for sentence in sentences:
        value = sentence.group()
        # ASR can join a moderator's attribution/thanks to the actual inverted
        # ask. Separate only this explicit reporting form; never an arbitrary
        # interior 'which/what' in scientific background or an indirect request.
        offset = next((cue.start() for cue in _EMBEDDED_INVERTED_ASK.finditer(value)
                       if cue.start() and (_THANKS_FOR_TALK.search(value[:cue.start()])
                           or _REPORTED_ASK.search(value[:cue.start()]))), None)
        if offset is None and not starts_direct_question(value):
            # An explicit polite request can follow scientific background even
            # when ASR misses the stop. A quotation/report is not a fresh ask.
            quotes = tuple(_QUOTED_SPEECH.finditer(value))
            offset = next((cue.start() for cue in _EMBEDDED_REQUEST.finditer(value)
                if cue.start() and not _REPORTED_CONTEXT.search(value[:cue.start()])
                and not any(q.start() <= cue.start() < q.end() for q in quotes)), None)
        if offset is None or starts_direct_question(value):
            result.append(sentence)
        else:
            cut = sentence.start() + offset
            result.extend((re.compile(r'\S[\s\S]*').match(text, sentence.start(), cut),
                           re.compile(r'\S[\s\S]*').match(text, cut, sentence.end())))
    return result


def _non_request_sentence_starts(text: str) -> set[int]:
    """Explicit quote/self-answer evidence, not inferred speaker identities."""
    spans = _speech_sentences(text)
    quotes = tuple(_QUOTED_SPEECH.finditer(text))
    suppressed = {m.start() for m in spans if any(
        q.start() <= m.start() < q.end() for q in quotes)}
    for index, match in enumerate(spans):
        if index and _SELF_ANSWER_CUE.search(match.group().strip()):
            # Suppress the immediately preceding ask only. A later independent
            # addressed question still reaches the model with all source kept.
            previous = spans[index - 1]
            if starts_direct_question(previous.group()):
                suppressed.add(previous.start())
        if _REPORTED_CONTEXT.search(match.group()):
            # A reported/quoted request stays background even if its text is
            # displayed without quotation marks after a colon.
            marker = _REPORTED_CONTEXT.search(match.group())
            if marker and not starts_direct_question(match.group()[:marker.start()]):
                suppressed.add(match.start())
    return suppressed


def is_non_request_context(text: str) -> bool:
    """True only when explicit quote/self-answer cues explain every ask.

    Automatic callers may withhold these narrated examples. Manual answering
    can still use the untouched source; uncertain discourse is not classified.
    """
    source = str(text).strip()
    spans = _speech_sentences(source)
    suppressed = _non_request_sentence_starts(source)
    candidates = [m for m in spans if starts_direct_question(m.group()) or '?' in m.group()
        or _EMBEDDED_REQUEST.search(m.group())]
    return bool(candidates and all(m.start() in suppressed for m in candidates))


def _request_has_content(sentence: str) -> bool:
    # Preserve the old minimum for arbitrary fragments. The shorter forms are
    # explicit topic requests or bounded complete predicates, not merely a
    # 3-word opening such as 'Do you think?' / 'Could you tell me?'.
    frame = _REQUEST_FRAME.search(sentence)
    if frame:
        complement = sentence[frame.end():].strip(' ,:;.!?')
        if (not complement or re.fullmatch(
                r"(?:how|why|what|which|where|when|whether|if)(?:\s+(?:you|we|I|he|she|they|it))?|"
                r"(?:you|we|I|he|she|they|it|this|that|these|those|your\s+thoughts)", complement, re.I)):
            return False
        return bool(re.search(r'[A-Za-z0-9]', complement))
    return bool(len(sentence.split()) >= 5 or
                (len(sentence.split()) >= 3 and
                 (_TOPIC_FOLLOWUP.search(sentence) or _SHORT_COMPLETE_REQUEST.fullmatch(sentence))))


def has_complete_direct_question(text: str, *, require_terminal: bool = True) -> bool:
    source = str(text)
    if QuestionAssembler.incomplete_text(source):
        return False
    suppressed = _non_request_sentence_starts(source)
    for match in _speech_sentences(source):
        sentence = match.group().strip()
        if (match.start() not in suppressed and not is_question_invitation(sentence)
                and starts_direct_question(sentence) and _request_has_content(sentence)
                and (not require_terminal or sentence.endswith(('?', '.', '!')))
                and not QuestionAssembler.incomplete_text(sentence)):
            return True
    return False


def question_after_moderation_offset(text: str) -> int | None:
    """Locate a new ask after ONLY names/attribution/thanks, not scientific setup.

    Used solely after a confirmed invitation/handoff. An incomplete opening
    such as 'Which' starts the question's own recognition-integrity interval;
    a rejection after that word may never be cleared by later question text.
    Unknown/technical prefixes deliberately do not grant a new boundary.
    """
    for sentence in _speech_sentences(str(text)):
        if starts_direct_question(sentence.group()) or announces_question(sentence.group()):
            before = text[:sentence.start()].strip()
            if not before or _MODERATOR_PREFIX.fullmatch(before):
                return sentence.start()
            return None
    return None


def confirmed_question_boundary(text: str) -> int | None:
    """Return a source offset only after a complete explicit ask AND a cue.

    A question mark alone cannot trigger. Short noun complements ("The video?")
    stay with the ask; an acknowledgement/answer opening or explicit next-topic
    introduction closes it. This works with continuing audio and ASR queues.
    """
    boundaries = []
    for cue in _SOCIAL_RESPONSE.finditer(str(text)):
        if any(quote.start() <= cue.start() < quote.end() for quote in _QUOTED_SPEECH.finditer(text)):
            continue
        prefix = text[:cue.start()].strip()
        if (has_complete_direct_question(prefix, require_terminal=False)
                and not QuestionAssembler.incomplete_text(prefix)):
            boundaries.append(cue.start())
    for cue in _TENTATIVE_EXPLANATION.finditer(str(text)):
        if any(quote.start() <= cue.start() < quote.end() for quote in _QUOTED_SPEECH.finditer(text)):
            continue
        prefix, reply = text[:cue.start()].strip(), text[cue.start():]
        explanation = _EXPLANATION_CLAUSE.search(reply, cue.end() - cue.start())
        if (explanation and explanation.start() <= 240
                and not _QUALIFYING_CONTINUATION.search(reply[:explanation.start()])
                and '?' not in reply
                and not QuestionAssembler.incomplete_text(reply)
                and has_complete_direct_question(prefix, require_terminal=False)
                and not QuestionAssembler.incomplete_text(prefix)):
            boundaries.append(cue.start())
    seen_question = False
    suppressed = _non_request_sentence_starts(str(text))
    for match in _speech_sentences(str(text)):
        sentence = match.group().strip()
        if seen_question and not _SELF_CORRECTION.search(sentence) and (acknowledgment_only(sentence)
                or _RESPONSE_START.search(sentence) or announces_question(sentence)):
            boundaries.append(match.start())
            break
        if (match.start() not in suppressed and not is_question_invitation(sentence)
                and starts_direct_question(sentence) and _request_has_content(sentence)
                and sentence.endswith(('?', '.', '!'))
                and not QuestionAssembler.incomplete_text(sentence)):
            seen_question = True
    return min(boundaries) if boundaries else None



def speculative_question_snapshot(text: str) -> str:
    """Return confirmed source for hidden preparation, NEVER a turn endpoint.

    A complete-looking question can acquire later conditions.  Likewise a
    stance/restart such as ``this honestly`` may belong to its questioner.
    Callers may prepare an answer for this exact snapshot, but must not seal
    the question, publish the draft, or discard subsequent words because of
    this helper.  Only an independent endpoint and an exact snapshot match
    permit reuse.  Missing/rejected speech must be handled by the caller.

    Keep offsets and every earlier condition.  Do not summarize, infer a new
    speaker, or clip by character/word count.  A tentative response prefix is
    useful only for removing possible reply speech from hidden computation;
    the full ongoing source must remain in the question ledger meanwhile.
    """
    source = str(text).strip()
    if not source:
        return ''
    boundary = confirmed_question_boundary(source)
    if boundary is not None:
        return source[:boundary].strip()
    for cue in _TENTATIVE_EXPLANATION.finditer(source):
        if any(quote.start() <= cue.start() < quote.end()
               for quote in _QUOTED_SPEECH.finditer(source)):
            continue
        prefix = source[:cue.start()].strip()
        tail = source[cue.end():]
        if (_QUALIFYING_CONTINUATION.search(tail)
                or re.search(r"\b(?:I\s+(?:need|want|require)|my\s+question|"
                             r"under|with|without|not\s+the)\b", tail, re.I)):
            continue
        if (has_complete_direct_question(prefix, require_terminal=False)
                and not QuestionAssembler.incomplete_text(prefix)):
            return prefix
    # Require an explicit complete final ask here. A fragment or a trailing
    # explanation is not a reason to start a model on every ASR word/window.
    sentences = _speech_sentences(source)
    if sentences:
        tail = sentences[-1].group().strip()
        if (tail.endswith(('?', '.', '!'))
                and has_complete_direct_question(tail)
                and not QuestionAssembler.incomplete_text(source)):
            return source
    return ''


def announced_question_offset(text: str) -> int | None:
    return next((m.start() for m in _speech_sentences(str(text))
                 if announces_question(m.group())), None)


def next_addressed_question_offset(text: str) -> int | None:
    """Recover a new ask after reply speech without accepting quoted self-asks."""
    suppressed = _non_request_sentence_starts(str(text))
    return next((m.start() for m in _speech_sentences(str(text))
                 if m.start() not in suppressed and (announces_question(m.group()) or (starts_direct_question(m.group())
                    and not _DECLARATIVE_WH.search(m.group())
                    and (re.search(r'\byou(?:r)?\b', ' '.join(m.group().split()[:20]), re.I)
                         or (_REQUEST_FRAME.search(m.group()) and not re.search(
                             r"^I(?:(?:['’]m|\s+am)\s+(?:just\s+)?curious\b|"
                             r"\s+(?:was\s+)?(?:just\s+)?(?:wonder|wondering)\b)", m.group(), re.I))
                         or re.search(r"^(?:please\s+)?(?:explain|describe|discuss|clarify|compare|"
                            r"tell|walk|talk)\b|^I(?:['’]d|\s+would)\s+like\b",m.group(),re.I)
                         # A fresh crop can begin a short spoken follow-up. Do
                         # not promote interior self-questions or undirected
                         # 'I wonder...' reflection inside a continuing answer.
                         or (m.start() == 0 and has_complete_direct_question(m.group())
                             and _TOPIC_FOLLOWUP.search(m.group())))))), None)


_COMPLEMENT_HEADS = frozenset((
    'algorithm architecture calibration camera channel coefficient component constraint data decoder '
    'detector encoder equation experiment field function geometry image information input instrument '
    'layer loss mapping material matrix measurement method model motion network operator output '
    'parameter pixel problem reconstruction reflectance representation result sample scene sensor '
    'sequence signal space spectrum system target tensor trajectory uncertainty vector video weight'
).split())
_COMPLEMENT_ADJECTIVES = frozenset((
    'calibrated measured learned known unknown observed reconstructed estimated trained compressed encoded '
    'forward inverse spectral polarimetric optical computational linear nonlinear temporal spatial '
    'physical neural reference nominal actual synthetic real original complete full input output '
    'motion synthesis sensing imaging rendering sampling polarization'
).split())


def _short_nominal_complement(sentence: str) -> bool:
    """A bounded NP grammar, not a general grammatical/semantic classifier.

    Only known nominal heads and simple pre-nominal modifiers are accepted.
    Conditions, predicates, unknown heads and embedded clauses remain in full
    context rather than being guessed into the compact question.
    """
    value = str(sentence).strip()
    if not value.endswith(('.', '?')):
        return False
    words = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)*", value[:-1])
    if not 2 <= len(words) <= 6 or not re.fullmatch(r'[A-Za-z -]+', value[:-1]):
        return False
    words = [word.casefold() for word in words]
    if words[0] not in {'the', 'a', 'an', 'this', 'that', 'these', 'those', 'our', 'your', 'its', 'their'}:
        return False
    head = (words[-1] if words[-1] in _COMPLEMENT_HEADS else
            {'matrices':'matrix', 'spectra':'spectrum'}.get(words[-1], words[-1].removesuffix('s')))
    if head not in _COMPLEMENT_HEADS:
        return False
    modifiers = words[1:-1]
    for index, word in enumerate(modifiers):
        if word not in _COMPLEMENT_ADJECTIVES and word not in _COMPLEMENT_HEADS:
            return False
        # 'The model measured ...' can be a new finite assertion. An approved
        # participle is accepted only immediately after the determiner.
        if word.endswith('ed') and index != 0:
            return False
    return True


def _focus_source_sentences(text: str):
    """Keep exact source spans when a missed stop joins an explicit follow-up.

    This is only a display/checklist aid; it never establishes a turn endpoint.
    A quoted self-question or an arbitrary relative ``what/which`` clause is
    not enough. Require the spoken OR plus an addressed auxiliary ask.
    """
    cue = re.compile(
        r"\bor\s+(?:(?:you\s+know|I\s+mean)[, :]+)?(?="
        r"(?:(?:what|which)\b[^.!?]{0,100}?\b|(?:how|why|where|when)\s+)?"
        r"(?:do|did|does|can|could|would|will|should|have|has|are|is)\s+you\b)", re.I)
    for match in _speech_sentences(str(text)):
        sentence = match.group().strip()
        split = None
        if not starts_direct_question(sentence):
            quotes = tuple(_QUOTED_SPEECH.finditer(sentence))
            split = next((found.start() for found in cue.finditer(sentence)
                          if found.start() and not any(
                              quote.start() <= found.start() < quote.end() for quote in quotes)), None)
        if split is None:
            yield sentence
        else:
            yield sentence[:split].strip()
            yield sentence[split:].strip()


def _alternative_fragment(sentence: str) -> bool:
    """A source-only NP/gerund option, not a new finite assertion.

    Unknown alternatives stay in complete context. Do not promote every OR
    comment to an ask merely because it happens to follow one.
    """
    value = re.sub(r"^or\b[, :]*\s*", "", sentence, flags=re.I)
    if value == sentence:
        return False
    if _short_nominal_complement(value):
        return True
    return bool(re.match(r"^[A-Za-z]+ing\b", value) and
                not re.search(r"\b(?:am|is|are|was|were|do|does|did|can|could|would|will|"
                              r"should|shall|may|might|must|has|have|had)\b", value, re.I))


def _explicit_question_components(text: str) -> tuple[list[str], list[str]]:
    ask, background = [], []
    adjacent_ask = False
    source = str(text)
    suppressed = _non_request_sentence_starts(source)
    quotes = tuple(_QUOTED_SPEECH.finditer(source))
    cursor = 0
    for sentence in _focus_source_sentences(text):
        position = source.find(sentence, cursor)
        cursor = max(cursor, position + len(sentence))
        excluded = position in suppressed or any(q.start() <= position < q.end() for q in quotes)
        # A confirmed crop can punctuate an alternative as a new sentence:
        # "Which direction: improving the pipeline? Or fixing it?"  Routing
        # retains the complete source already; its compact focus must retain
        # BOTH alternatives too. Strip only this discourse prefix for cue
        # matching, never from the text shown/supplied to the model. This does
        # not affect the conservative endpoint / speaker-boundary detector.
        followup = re.sub(r"^or\b[, :]*\s*(?:(?:you\s+know|I\s+mean)\b[, :]*\s*)?",
                          "", sentence, flags=re.I)
        if excluded or is_question_invitation(sentence):
            background.append(sentence)
            adjacent_ask = False
        elif (starts_direct_question(sentence) or
              (followup != sentence and starts_direct_question(followup))) and _request_has_content(sentence) and not QuestionAssembler.incomplete_text(sentence):
            ask.append(sentence)
            adjacent_ask = True
        elif adjacent_ask and (_short_nominal_complement(sentence) or _alternative_fragment(sentence)):
            # A punctuation model can render a spoken object as 'The video.'
            # or an either/or alternative as its own fragment. Keep the exact
            # nearby source with its ask rather than inventing a second ask.
            ask[-1] += ' ' + sentence
        else:
            background.append(sentence)
            adjacent_ask = False
    return ask, background


def explicit_question_focus(text: str) -> tuple[str, str]:
    """Extract existing ask spans for a compact preview; never rewrite them."""
    ask, background = _explicit_question_components(text)
    return ' '.join(ask), ' '.join(background)


def explicit_question_parts(text: str) -> tuple[str, ...]:
    """The same source spans used by focus, including indirect spoken requests."""
    return tuple(_explicit_question_components(text)[0])


def join_fragments(left: str, right: str) -> str:
    left, right = left.strip(), right.strip()
    if not left:
        return right
    if not right:
        return left
    a, b = left.split(), right.split()
    for n in range(min(len(a), len(b), 256), 1, -1):
        if [w.lower().strip('.,?!') for w in a[-n:]] == [w.lower().strip('.,?!') for w in b[:n]]:
            return " ".join(a + b[n:])
    if left.casefold() == right.casefold():
        return left
    return left + " " + right


class CaptionTurn:
    """One visible turn can contain several ASR windows and provisional edits.

    Only overlap is removed; no LLM rewrites, grammar guesses or invented words.
    A controller seals the card after real inactivity, not each audio hard cut.
    """
    def __init__(self):
        self.identity = ""
        self.committed = ""

    def update(self, text: str, final: bool = False, *, deduplicate: bool = True) -> tuple[str, str]:
        if not self.identity:
            self.identity = str(uuid.uuid4())
        combined = (join_fragments(self.committed, text) if deduplicate else
                    " ".join(part for part in (self.committed.strip(), text.strip()) if part))
        if final:
            self.committed = combined
        return self.identity, combined

    def seal(self):
        self.identity = ""
        self.committed = ""


class QuestionAssembler:
    def __init__(self, silence_seconds: float = 1.0, continuation_seconds: float = 12.0):
        self.silence_seconds = silence_seconds
        self.continuation_seconds = continuation_seconds
        self.text = ""
        self.generation = 0
        self.version = 0
        self.last_voice = 0.0
        self.last_text = 0.0
        self._answered_version = 0
        self._resume_gap = 0.0

    def voice(self, at: float | None = None):
        now = time.monotonic() if at is None else at
        gap = now - max(self.last_voice, self.last_text)
        if gap > .4:
            self._resume_gap = gap
        self.last_voice = now

    def add(self, text: str, at: float | None = None, *, deduplicate: bool = True) -> str:
        now = time.monotonic() if at is None else at
        # Submitting a suggestion must not destroy its question. A short pause
        # followed by a qualification is the SAME turn, including while a model
        # is answering. Start a new turn after a long gap or explicit reset.
        gap = max(self._resume_gap, now - max(self.last_voice, self.last_text))
        if self._answered_version and self.text and gap >= self.continuation_seconds:
            last_voice = self.last_voice
            self.reset()
            self.last_voice = last_voice
        self._resume_gap = 0.0
        combined = (join_fragments(self.text, text) if deduplicate else
                    " ".join(part for part in (self.text.strip(), text.strip()) if part))[-9000:]
        self.last_text = now
        if combined != self.text:
            self.text = combined
            self.version += 1
        return self.text

    def ready(self, now: float | None = None, pending: int = 0) -> bool:
        now = time.monotonic() if now is None else now
        return bool(self.text and not self.incomplete and self.version != self._answered_version and not pending
                    and now - self.last_voice >= self.silence_seconds
                    and now - self.last_text >= 0.18)

    @staticmethod
    def incomplete_text(text: str) -> bool:
        """Conservative syntactic tails, never a claim of semantic completeness.

        ASR can append any punctuation at a hard cut; a guessed question mark
        must not make an unfinished modal/conjunction look answerable.
        """
        spans = _speech_sentences(str(text))
        tail = spans[-1].group().strip() if spans else ''
        introduction = _QUESTION_INTRO.match(tail)
        if introduction and re.fullmatch(
                r"[\s,.?!:;…]*(?:(?:for\s+you|to\s+ask|if\s+I\s+may|please)[\s,.?!:;…]*)*",
                tail[introduction.end():], re.I):
            # A guessed question mark on an announcement is not the actual
            # question. Keep its source and wait for the substantive request.
            return True
        if _REQUEST_FRAME.search(tail) and not _request_has_content(tail):
            return True
        # These are source-integrity guards, not repairs: do not invent a
        # subject after a modal or an object after a cropped alternative.
        if re.search(r"(?:^|[.!?]\s*)(?:how|why|when|where)\s+"
                r"(?:can|could|would|should|will|must|might)\s+"
                r"(?:beyond|into|between|without|through|from|of|for|to)\b", str(text), re.I):
            return True
        return bool(re.search(
            r"\b(?:and|or|because|if|whether|with|without|of|to|the|a|an|for|which|that|on|about|regarding|"
            r"in|at|from|into|between|by|do|does|did|can|would|could|should|will|shall|might|must|"
            r"my|your|our|their|its|not|than|versus)[,.?!:; ]*$|"
            r"(?:[=+×÷/]|(?<!\w)-(?!\w))\s*[.?!]*$|"
            r"(?:^|[.!?]\s*)(?:how|why|what|where|when|which)\s+"
            r"(?:would|could|can|should|will|do|did|does)\s+(?:you|we|they|it)[,.?! ]*$|"
            r"(?:^|[.!?]\s*)(?:what|which|who|where|how|why|when)\s+"
            r"(?:is|are|was|were|has|have)[,.?! ]*$|"
            r"(?:^|[.!?]\s*)(?:can|could|would|will|do|did|have)\s+you[,.?! ]*$|"
            r"\bof\s+all[,.?!:; ]*$|"
            r"(?:^|[.!?]\s*)(?:or|and)\s+(?:convey|recover|infer|estimate|measure|acquire)[,.?! ]*$",
            text, re.I))

    @property
    def incomplete(self) -> bool:
        # A hard audio cut after a modal/possessive is not a completed ask,
        # even if Whisper guessed a full stop at that boundary.
        return self.incomplete_text(self.text)

    def consume(self, *, retain: bool = False) -> tuple[str, int]:
        result = (self.text, self.version)
        self._answered_version = self.version
        if not retain:
            self.text = ""
        return result

    def reset(self):
        self.generation += 1
        self.text = ""
        self.last_text = 0.0
        self.last_voice = 0.0
        self._resume_gap = 0.0
        self._answered_version = self.version

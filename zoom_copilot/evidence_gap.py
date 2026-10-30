"""Conservative routing for requests to quote a candidate's completed work.

This is a local evidence-availability check, not a semantic verifier or a claim
that the candidate never did something. Only the final request's personal
entries and attributed fact cards count; references, model knowledge and future
plans cannot establish a personal experiment or role. No network calls occur.
"""
from __future__ import annotations

import re

from .candidate_profile import candidate_fact_id
from .knowledge import tokenize


EVIDENCE_GAP_EN = (
    "The personal evidence retrieved for this question does not establish the completed result "
    "or individual role being asked about. This is a limitation of the records available for "
    "this question, not a conclusion about my entire experience. I would need a supporting "
    "experiment record or publication before quoting a personal metric or role."
)
EVIDENCE_GAP_ZH = (
    "本题检索到的个人资料，尚不足以确认所问的已完成结果或本人具体承担的角色。"
    "这只说明本题当前可用记录的证据有限，并不是对我全部经历的判断。"
    "在引用个人实测指标或说明本人角色之前，我需要有相应的实验记录或论文作为依据。"
)
EVIDENCE_GAP_WARNING = (
    "本题已检索的个人资料不足以支持所问已完成成果或本人角色，已显示程序生成的受约束提示；"
    "这不表示您从未做过，也不是模型已核实的结论。未增加模型调用。"
)
EVIDENCE_GAP_PARTIAL_WARNING = (
    "完整性限制：本题还包含未来方案或贡献的子问。当前受约束提示只说明既往个人证据不足，"
    "尚未回答未来子问；可将未来子问单独提交。未从未经核验的原回答中拼接建议。"
)

# Linguistic intent words only: no research domains, model names or case IDs.
_ACTIONS = (r"do|develop|design|build|implement|contribute|lead|perform|conduct|run|"
            r"complete|achieve|measure|obtain|validate|demonstrate|publish|test|acquire|"
            r"collect|evaluate|deliver|produce|establish|observe|record")
_PAST = (r"done|developed|designed|built|implemented|contributed|led|performed|conducted|"
         r"run|completed|achieved|measured|obtained|validated|demonstrated|published|tested|"
         r"acquired|collected|evaluated|delivered|produced|established|observed|recorded")
_ADVERBS = r"(?:(?:personally|already|actually|previously|ever|yourself|successfully)\s+){0,3}"
_DIRECT = re.compile(
    rf"\b(?:did\s+you\s+{_ADVERBS}(?:{_ACTIONS})|have\s+you\s+{_ADVERBS}(?:{_PAST})|"
    rf"(?:what|which|how|describe|tell\s+(?:me|us))\b[^,;?!.]{{0,130}}\byou\s+(?:have\s+)?{_ADVERBS}(?:{_PAST}))\b", re.I)
_ASK = re.compile(r"\b(?:what|which|how|did you|have you|were you|was your|tell|describe|report|quote|give|provide|state|suggest|recommend|outline|propose|discuss)\b|"
                  r"什么|哪些|多少|是否|请.{0,4}(?:介绍|说明|给出)|你.{0,10}(?:完成|负责).{0,10}[吗呢]", re.I)
_ROLE = re.compile(
    r"\b(?:your\s+(?:(?:own|actual|personal|specific)\s+){0,2}(?:role|responsibilit\w*|contribution)|"
    r"were\s+you\s+(?:(?:personally|actually)\s+)?responsible)\b", re.I)
_PAST_CUE = re.compile(r"\b(?:was|were|did|had|completed|published|previous|past|last|already|personally|measured|recorded)\b|"
                       r"已经|曾经|过去|完成了|做过|实测|负责了", re.I)
_PERSONAL_RESULT = re.compile(
    r"\b(?:your|my)\b.{0,100}\b(?:experiments?|trials?|benchmarks?|prototypes?|stud(?:y|ies)|tests?|results?|metrics?|measurements?|"
    r"accuracy|error|improvement|performance|papers?|publications?)\b|"
    r"\b(?:experiments?|trials?|results?|metrics?|measurements?|improvement)\b.{0,100}\b(?:your|my)\b", re.I)
_RESULT_NOUN = re.compile(r"\b(?:values?|results?|metrics?|measurements?|accuracy|error|improvement|performance|papers?|publications?)\b", re.I)
# Value requests accept open vocabulary (including unseen metric acronyms).
# A completed personal record supplies the scope, not a metric/domain whitelist.
_VALUE_REQUEST = re.compile(
    r"^(?:what|which)\s+(?:is|are|was|were)\b|"
    r"^how\s+(?:much|many|large|small|high|low|fast|slow|accurate|precise)\b|"
    r"^(?:give|provide|state|report|quote)\b", re.I)
_CONCEPT_REQUEST = re.compile(
    r"^(?:what\s+(?:does|do)|how\s+(?:is|are|does|do)|why)\b|"
    r"\b(?:meaning|definition|mechanism|principle|reason|cause|objective|purpose|rationale|interpretation|difference)\b", re.I)
_FUTURE = re.compile(r"\b(?:would|will|should|could|plan(?:ning)?|propos\w*|future|next|hypothetic\w*|suppose)\b|"
                     r"将来|未来|计划|打算|假设|如果|准备|拟议", re.I)
_ZH_COMPLETED = re.compile(
    r"(?:你|您|我)(?:本人|自己)?(?:已经|曾经|实际|亲自|过去)?(?:完成|做过|测得|测量了|取得|负责|开发了|实现了)|"
    r"(?:你|您|我)(?:本人|自己)?的.{0,35}(?:已完成|实测|过去|已发表)")
_SPLIT = re.compile(
    r"(?<=[?!;。！？；])\s*|(?<=\.)\s+(?=[A-Z])|"
    r"(?:,\s*|\s+(?:and|but|then|also)\s+)"
    r"(?=(?:(?:then|also|next|after\s+that)\s+)?(?:please\s+)?"
    r"(?:what|which|how|why|would|will|could|can|suggest|recommend|outline|describe|propose|discuss|explain|clarify|tell|walk|talk)\b)", re.I)

# Remove request/record boilerplate from topical overlap. The remaining words
# are open vocabulary; this is deliberately not a list of supported disciplines.
_GENERIC = set("""a an the what which how when where who whom why your yours my mine own
personal personally actual actually already previous previously ever exact exactly
specific specifically individual completed complete completion past current future
did done do does have has had was were is are be been being i me you yourself we our
they their his her candidate supplied document documented record records retrieved
evidence established establish confirm confirmed result results metric metrics
measurement measurements measured measure measuring experiment experiments
experimental trial trials benchmark benchmarks prototype prototypes study studies work works research doctoral phd master
masters contribution contributions role roles responsibility responsibilities
responsible part parts amount number value values paper papers publication
publications published report reports reported quoting quote achieve achieved gain
gains improvement improvements improve performance accuracy error errors mean
percentage percent quantitative numerical tested test testing validation validated
validate do doing develop developed developing design designed designing build
built building implement implemented implementing lead led perform performed
conduct conducted run acquired acquire acquisition collect collected dataset data
method methods system systems approach model models network networks project
projects task tasks team lab laboratory first main key one two three describe tell
explain please about for of with from in on at to and or but under across against
that this these those then than also only same corresponding source sources
according before after final finished successful successfully achieved establish
demonstrate demonstrated obtain obtained support supporting stated states statement
user explicit confirmed text supplied source sources cv script speaking personal
last give provide state large small high low fast slow precise
""".split())
_ZH_GENERIC = set(tokenize("你本人的自己已经曾经实际亲自过去完成做过实测测量取得负责开发实现"
                          "什么哪些多少是否请介绍说明给出结果指标实验记录论文贡献角色研究资料"))


def personal_record_requests(question: str) -> list[str]:
    """Find explicit completed-personal asks, not premises or general science.

    A mixed past/future question retains its past clause. A technical question
    containing a statement such as 'your system measured X' is not enough.
    """
    requests = []
    for clause in _SPLIT.split(str(question or "")):
        if not _ASK.search(clause):
            continue
        direct = bool(_DIRECT.search(clause) or _ZH_COMPLETED.search(clause))
        ask = clause[_ASK.search(clause).start():]
        ask_head = re.split(r"\b(?:in|on|for|from|about|with)\b", ask, maxsplit=1, flags=re.I)[0]
        value_request = ((_RESULT_NOUN.search(ask_head) or _VALUE_REQUEST.search(ask_head))
                         and not _CONCEPT_REQUEST.search(ask_head))
        scoped = bool(_PAST_CUE.search(clause) and
                      (_ROLE.search(clause) or (_PERSONAL_RESULT.search(clause) and value_request)))
        if not (direct or scoped):
            continue
        # 'What would you have done?' is hypothetical, whereas 'What did you
        # measure, and how would you extend it?' keeps the completed ask.
        hypothetical = re.search(r"\b(?:would|could|should)\s+you\s+(?:have\s+)?(?:" + _PAST + r")\b|"
                                 r"\b(?:suppose|if)\s+you\b|假设|如果", clause, re.I)
        if hypothetical and not re.search(r"\b(?:did|have)\s+you\b|已经|曾经|实测", clause, re.I):
            continue
        if not direct and re.search(r"\b(?:would|will|next|future|plan(?:ned)?|propos\w*)\b|将来|未来|计划|拟议", clause, re.I):
            continue
        requests.append(clause.strip())
    return requests


def has_unanswered_future_subquestion(question: str) -> bool:
    """Flag mixed-question incompleteness rather than reuse an unchecked draft."""
    future_ask = re.compile(r"\b(?:would|will|should)\s+(?:you|your|we|our)\b|"
                            r"\b(?:future|next|plan\w*|propos\w*)\b|"
                            r"\b(?:if|when)\s+you\s+(?:join|joined)\b|未来|计划|打算", re.I)
    return any(_ASK.search(clause) and future_ask.search(clause)
               and not personal_record_requests(clause)
               for clause in _SPLIT.split(str(question or "")))


def split_answer_subquestions(question: str) -> list[str]:
    """Retain context while separating explicit requests for coverage/routing.

    This is a local checklist, not a semantic rewrite. The complete original
    question must also remain in the model request. Non-question context stays
    attached to its neighbour instead of being silently discarded.
    """
    parts, background = [], []
    for clause in _SPLIT.split(str(question or "")):
        clause = clause.strip(' ,\r\n')
        if not clause:
            continue
        asked = bool(_ASK.search(clause) or re.match(
            r"\b(?:why|when|where|who|is|are|does|do|can|could|would|will|should)\b|为什么|怎样|如何", clause, re.I))
        if asked:
            if background:
                clause = '. '.join([*background, clause])
                background.clear()
            parts.append(clause)
        elif re.match(r'^(?:if|assuming|suppose|given|regarding|under|for)\b', clause, re.I):
            background.append(clause)
        elif parts:
            parts[-1] += '. ' + clause
        else:
            background.append(clause)
    if background:
        parts.append('. '.join(background))
    return parts or ([question.strip()] if question.strip() else [])


def mixed_evidence_answer_plan(question: str, final_request: dict) -> list[dict]:
    """A separate output contract for missing-history + answerable-future asks.

    The model never drafts the unsupported personal-history section. Future
    sections are generated in their own fields, not extracted from a risky
    whole-answer draft. Unclassified sections remain explicitly incomplete.
    """
    if not personal_evidence_insufficient(question, final_request):
        return []
    plan = []
    for index, part in enumerate(split_answer_subquestions(question), 1):
        if personal_evidence_insufficient(part, final_request):
            mode = 'local_evidence_gap'
        elif has_unanswered_future_subquestion(part):
            mode = 'future_proposal'
        else:
            mode = 'not_generated_in_guarded_reply'
        plan.append({'id': f'Q{index}', 'question_en': part, 'mode': mode})
    return plan if any(row['mode'] == 'future_proposal' for row in plan) else []


def future_section_violation(english: str, chinese: str) -> str:
    """Narrow deterministic checks; these do not certify scientific correctness.

    A future-only section must be phrased as a proposal and may not smuggle in
    candidate past achievements/denials or credentials. Both languages are
    checked before either becomes visible. General factual claims still need
    source review; lexical checks are not a substitute for that review.
    """
    if not re.search(r"\b(?:would|could|should|will|propose|plan|recommend|suggest|first\s+step)\b", english, re.I):
        return 'future_status_missing'
    if not re.search(r'会|将|可|应|拟|计划|建议|首先|未来|下一步|打算', chinese):
        return 'future_status_missing_in_translation'
    personal_past = re.compile(
        r"\b(?:I|we)\s+(?:(?:have|had|have\s+not|had\s+not)\s+)?"
        r"(?:(?:not|never|already|previously|personally|successfully|also)\s+){0,3}"
        r"(?:" + _PAST + r"|worked|used|was|were|invented|proved|established)\b|"
        r"\b(?:my|our)\s+(?:[a-z-]+\s+){0,3}(?:experience|background|expertise|publications?|achievements?|"
        r"previous\s+work|prior\s+work|published\s+work|completed\s+work|demonstrated\s+results?)\b|"
        r"\b(?:as|being)\s+(?:a|an)\s+(?:expert|inventor|author|experienced\s+researcher)\b", re.I)
    if personal_past.search(english):
        return 'personal_history_in_future_section'
    # Completed claims also use did + bare verb or a possessive artifact as
    # subject. A prospective 'our prototype would achieve' is deliberately not
    # matched. This remains a narrow language check, not semantic verification.
    if re.search(r"\b(?:I|we)\s+did\s+(?:not\s+)?(?:" + _ACTIONS + r")\b|"
                 r"\b(?:my|our)\s+(?:[a-z-]+\s+){0,4}(?:" + _PAST + r"|was|were)\b",
                 english, re.I):
        return 'personal_history_in_future_section'
    if re.search(r'(?:我|我们)(?:(?![会将拟可应]).){0,14}(?:已经|曾经|从未|过去|之前|完成了|做过|开发了|取得了|负责过|发表过)|'
                 r'(?:我|我们|我们的.{0,6})(?:曾|已)?(?:测得|取得了|达到了|证明了|完成了)|'
                 r'(?:我的|我们的).{0,12}(?:经验|经历|成果|专长|论文|既往工作)|作为.{0,6}(?:专家|发明者|作者)', chinese):
        return 'personal_history_in_future_translation'
    return ''


def _subject_terms(text: str) -> set[str]:
    terms = set(tokenize(text.replace("_", " "))) - _GENERIC - _ZH_GENERIC
    # Numbers alone do not establish ownership or a topical relationship.
    return {term[:-1] if len(term) > 4 and term.endswith('s') and not term.endswith(('ss', 'us')) else term
            for term in terms if not term.replace('.', '').isdigit() and len(term) > 1}


def _personal_records(data: dict) -> list[str]:
    records = []
    for card in data.get("research_fact_cards_untrusted", []):
        if (card.get("provenance") == "document_derived"
                and card.get("kind") in {"personal_contribution", "personal_experience"}
                and card.get("status") in {"document_reported", "published_in_supplied_record", "completed"}):
            records.append(str(card.get("topic", "")) + " " + str(card.get("claim_en", "")))
    personal = data.get("candidate_confirmed_facts", {})
    if personal.get("origin") != "explicit_user_candidate_profile":
        return records
    for entry in personal.get("entries", []):
        text = str(entry.get("text", ""))
        if entry.get("id") != candidate_fact_id(text):
            continue
        if entry.get("kind") not in {"contribution", "master", "fact"}:
            continue
        # Source labels are not claims, and a future clause cannot prove a past
        # role. Split mixed entries so an explicit completed clause remains usable.
        claim = re.split(r"\bSources?:", text, maxsplit=1, flags=re.I)[0]
        for clause in _SPLIT.split(claim):
            if _FUTURE.search(clause) or re.search(r"\b(?:not|never)\b.{0,18}\b(?:" + _ACTIONS + "|" + _PAST + r")\b", clause, re.I):
                continue
            if entry.get("kind") in {"contribution", "master"} or re.search(
                    rf"\bI\s+(?:have\s+)?{_ADVERBS}(?:{_PAST})\b|我(?:已经|曾经|亲自)?(?:完成|负责|开发|实现|测得)", clause, re.I):
                records.append(clause)
    return records


def personal_evidence_insufficient(question: str, final_request: dict) -> bool:
    """True only for explicit personal-record requests without same-topic support.

    This does not certify supported answers: a same-topic record only permits
    normal generation, whose numbers, scope and citations still need checking.
    """
    requests = personal_record_requests(question)
    if not requests:
        return False
    records = [_subject_terms(text) for text in _personal_records(final_request)]
    for request in requests:
        terms = _subject_terms(request)
        # A broad 'what did you personally develop?' can use any completed
        # contribution. Specific asks need topic overlap in that completed record.
        if not any((not terms) or (terms & record) for record in records):
            return True
    return False

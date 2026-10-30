"""Bounded, explicit candidate facts for interview answers and local retrieval.

The profile is authored/confirmed by the user.  Imported research papers are
literature, not an implicit CV; neither author lists nor filenames identify the
candidate.  This module never reads a file, calls a model or invents a summary.
"""
from __future__ import annotations

import re
import hashlib
from collections import Counter
import math


PROFILE_SOURCE = "候选人背景（用户提供，非论文原文）"


def candidate_fact_id(text: str) -> str:
    """Stable across selection/reordering; different same-named entries stay distinct."""
    return "profile-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:20]


_PERSONAL = re.compile(
    r"\b(?:your|yours)\s+(?:(?:own|main|key|specific|doctoral|phd|ph\.d\.?|research)\s+){0,3}"
    r"(?:contributions?|roles?|responsibilit\w*|achievements?|experiments?|results?|"
    r"publications?|papers?|thesis|dissertation|phd|ph\.d\.?|background|experience|skills?|strengths?|weaknesses?|projects?|career|education|work|method|approach|research)\b|"
    r"\b(?:what|which)\s+(?:exactly\s+)?(?:did|have)\s+you\s+(?:personally\s+)?"
    r"(?:do|develop|design|implement|contribute|publish|achieve|validate|measure|demonstrate)\b|"
    r"\b(?:tell|walk)\s+(?:me|us)\s+(?:about|through)\s+(?:your|yourself)\b|"
    r"\b(?:introduce yourself|why should we hire you|why do you want (?:this|the) (?:job|role|position))\b|"
    r"(?:你(?:本人|自己)?的|您(?:本人|自己)?的).{0,16}(?:博士|研究|贡献|成果|工作|论文|方法|实验|经历|技能|优势|缺点|教育|项目)|"
    r"(?:你|您)(?:本人|自己)?(?:负责|完成|做了|设计|实现)", re.I)
_BRIDGE = re.compile(
    r"\b(?:your|phd|ph\.d\.?|doctoral)\b.{0,90}\b(?:fit|relat\w*|connect\w*|transfer\w*|"
    r"target role|position|job|career|postdoc|co.design)\b|"
    r"(?:经验|技能|博士|你的研究|您的研究).{0,40}(?:关联|联系|结合|岗位|职位|博后)", re.I)

# Spoken questions commonly put a number or an adjective between "your" and
# the requested achievement. Keep this bounded to one sentence rather than
# enumerating a few permitted adjectives (which missed "your three main ...").
_PERSONAL_FLEX = re.compile(
    r"\byour\b[^.!?\n]{0,90}\b(?:contributions?|roles?|responsibilit\w*|achievements?|"
    r"experiments?|results?|publications?|papers?|thesis|dissertation|phd|doctoral|"
    r"background|experience|work|method|approach|research|error|improvement)\b|"
    r"\b(?:did|have|had|were)\s+you\b[^.!?\n]{0,55}\b(?:publish\w*|recover\w*|"
    r"develop\w*|design\w*|implement\w*|build|built|do|done|contribut\w*|achiev\w*|validat\w*|measur\w*|demonstrat\w*)\b", re.I)
_GENERIC = set("""identity topic personal contribution contributions main key three two one first
second third phd doctoral master masters background experience previous own already exactly
completed future proposal postdoc published publication publications result results measured
measurement numerical observe observed gain error source sources page pages speaking script
research context local document lookup support contribute relate related transfer transferable
strengths work method approach how what which describe explain new mean means full only
part particular current question please you your my i not no including include
major important novel original innovative core primary principal significant benefit
benefits relevance relevant skills strengths weaknesses career education introduce yourself
develop developed developing design designed designing implement implemented implementing
build built do done achieve achieved validate validated demonstrate demonstrated
""".split())


def _claim_text(row: str) -> str:
    """Source names are retained in output, but are not subject-relevance votes."""
    return re.split(r"\bSources?:", row, maxsplit=1, flags=re.I)[0]


def _row_kind(row: str) -> str:
    head = row.split(':', 1)[0].casefold()
    if re.search(r"master|硕士", head): return 'master'
    if re.search(r"future|proposal|proposed|未来|拟议", head): return 'future'
    if re.search(r"transfer|迁移", head): return 'transfer'
    if re.search(r"identity|\bname\b|身份|姓名", head): return 'identity'
    if re.search(r"\btopic\b|研究方向", head): return 'topic'
    if re.search(r"contribution|贡献", head): return 'contribution'
    return 'fact'


def _selection_order(question: str, rows: list[str], kind: str) -> list[int]:
    from .knowledge import tokenize
    query = set(tokenize(question)) - _GENERIC
    terms = [set(tokenize(_claim_text(row))) - _GENERIC for row in rows]
    kinds = [_row_kind(row) for row in rows]
    frequency = Counter(term for words in terms for term in words)
    weight = lambda term: math.log1p((len(rows) + 1) / (frequency[term] + 1))
    scores = [sum(weight(term) for term in query & words) for words in terms]
    master_asked = bool(re.search(r"master|\bmsc\b|硕士", question, re.I))
    identity_asked = bool(re.search(r"\b(?:name|university|supervisor|graduate|graduation|background|introduce)\b|姓名|导师|毕业|介绍自己", question, re.I))
    future_relevant = kind == 'research_bridge' or bool(re.search(
        r'\b(?:future|propos\w*|postdoc|target role|position|career|transfer)\b|博后|未来|岗位|职位|转型', question,re.I))
    # Retain prospective caveats when the user's own future/transfer rows match
    # this question. No named research field is baked into this selection.
    future_relevant = future_relevant or any(k in ('future', 'transfer') and score > 0
        for k, score in zip(kinds, scores))
    # An overview needs the candidate's actual doctoral methods before unrelated
    # publication lists or the other lab's vocabulary. A bridge additionally
    # needs the proposed/transfer status; those lines remain verbatim.
    bridge_terms = {'role', 'position', 'fit', 'connect', 'connection', 'skills'}
    broad = kind == 'research_bridge' or (kind != 'scientific' and not (query - bridge_terms))
    if broad:
        contributions = [i for i,k in enumerate(kinds) if k == 'contribution']
        # Established core method entries precede a general experiments entry.
        core = [i for i in contributions if not re.search(r'contribution\s*[—:-]\s*experiments', rows[i], re.I)]
        order = core[:3]
        if kind == 'research_bridge':
            order += [i for i,k in enumerate(kinds) if k == 'future'][:1]
            order += [i for i,k in enumerate(kinds) if k == 'transfer']
        order += contributions + [i for i,k in enumerate(kinds) if k == 'topic']
        order += sorted(range(len(rows)), key=lambda i:(-scores[i],i))
    else:
        order = sorted(range(len(rows)), key=lambda i:(-scores[i],kinds[i]=='topic',i))
        threshold = max(scores, default=0) * .30
        order = [i for i in order if scores[i] > 0 and scores[i] >= threshold]
        if any(kinds[i] in ('contribution','fact') for i in order):
            order = [i for i in order if kinds[i] != 'topic']
        if kind == 'personal_history' and future_relevant:
            # A completed-result question in a proposed application must retain
            # the explicit future/new-area caveat even if an old experiment's
            # distance or numbers dominate lexical similarity.
            scope = [i for i,k in enumerate(kinds) if k=='future' and scores[i]>0][:1]
            scope += [i for i,k in enumerate(kinds) if k=='transfer' and scores[i]>0]
            order = order[:1] + scope + order[1:]
        if identity_asked:
            order = [i for i,k in enumerate(kinds) if k == 'identity'] + order
    result=[]
    for i in dict.fromkeys(order):
        if kinds[i] == 'master' and not master_asked: continue
        if kinds[i] == 'identity' and not identity_asked: continue
        if kinds[i] in ('future','transfer') and not future_relevant: continue
        # Do not fill unused room with unrelated facts merely because they fit.
        if not broad and scores[i] == 0 and kinds[i] != 'identity': continue
        result.append(i)
    return result


def personal_question_kind(question: str) -> str:
    if _BRIDGE.search(question):
        return "research_bridge"
    return "personal_history" if _PERSONAL.search(question) or _PERSONAL_FLEX.search(question) else "scientific"


def _rows(profile: str) -> list[str]:
    # Keep user wording and source/page annotations verbatim, apart from controls.
    clean = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", str(profile or ""))
    if len(clean) > 12000:
        raise ValueError("候选人背景超过 12000 字符，请先精简；本次未截断并发送。")
    return [row.strip() for row in clean.splitlines() if row.strip()][:80]


def candidate_context(question: str, profile: str = "", maximum: int = 2200) -> dict:
    """Select complete user entries, retaining source hints without extracting claims.

    Candidate facts are distinct from source excerpts. Their confirmation means
    that the user entered them, not that a paper author list was cross-verified.
    """
    rows = _rows(profile)
    kind = personal_question_kind(question)
    order = _selection_order(question, rows, kind)
    selected, used = [], 0
    for i in order:
        if len(rows[i]) + used + 1 > maximum:
            continue
        selected.append((i, rows[i])); used += len(rows[i]) + 1
    selected.sort()
    return {
        "question_kind": kind,
        "origin": "explicit_user_candidate_profile",
        "facts": [text for _, text in selected],
        "citation_ids": [candidate_fact_id(text) for _, text in selected],
        "source": PROFILE_SOURCE,
        "profile_available": bool(rows),
        "omitted_entries": len(rows) - len(selected),
        "personal_contribution_missing": kind == "personal_history" and not selected,
        "meaning": "User-provided entries, not verified paper excerpts. citation_ids maps to facts in the same order; source/page hints inside a fact are not independently verified citations.",
    }


def candidate_prompt_context(context: dict) -> dict:
    """Pair each complete entry with its ID for the model, without duplicating text.

    The retrieval API retains its legacy facts list. Parallel facts/IDs are an
    internal representation only; the model must never infer their association
    by position in separate arrays. Row kinds describe the user's headings, not
    an independently established experiment/publication status.
    """
    result = {key: value for key, value in context.items()
              if key not in ("facts", "citation_ids", "meaning", "personal_contribution_missing")}
    result["entries"] = [
        {"id": candidate_fact_id(text), "kind": _row_kind(text), "text": text}
        for text in context.get("facts", [])
    ]
    if not result["entries"]:
        result.pop("source", None)
        result.pop("origin", None)
        if context.get("question_kind") == "personal_history":
            result["personal_contribution_missing"] = True
        result["meaning"] = "No profile entries selected; this is not evidence of absent personal experience."
        return result
    result["meaning"] = (
        "User-provided complete entries, not verified paper excerpts. Cite each used entry's own id. "
        "Future applies only to the stated proposal; future/transfer entries do not establish "
        "that all related prior experiments or publications are absent."
    )
    return result


def retrieval_query(question: str, candidate_profile: str = "", research_profile: str = "") -> str:
    """Expand vague personal-history asks with explicitly supplied local context.

    Specific scientific questions stay unchanged: broad topic words otherwise
    displace exact equations/definitions. This text goes to the local index only.
    """
    if personal_question_kind(question) == "scientific":
        return question
    selected = candidate_context(question, candidate_profile, maximum=2600)["facts"]
    topic = re.sub(r"\s+", " ", str(research_profile or "")).strip()[:650]
    # This is a local search expansion, not a claim summary sent to a model.
    # Short claim openings add method vocabulary without allowing a full CV's
    # metadata and unrelated numbers to overwhelm the actual question.
    hints = [_claim_text(row).split('. ', 1)[0][:210] for row in selected
             if _row_kind(row) not in ('identity', 'topic')]
    if topic and not any(_row_kind(row)=='topic' for row in _rows(candidate_profile)):
        hints.append(topic.split('. ',1)[0][:180])
    extra = "\n".join(hints[:5]).strip() or topic
    return question + ("\nCandidate research context for local document lookup:\n" + extra if extra else "")


def missing_personal_warning(question: str, profile: str = "") -> str:
    if personal_question_kind(question) == "personal_history" and not str(profile or "").strip():
        return ("缺少已确认的候选人经历或贡献：参考资料不能自动证明您本人负责的职责、项目和成果。"
                "请在“候选人背景”填写事实及对应文件/页码；本次不得将他人成果直接称为您的成果。")
    return ""

"""Local, document-derived research facts with complete conditions and provenance.

This module never calls a model or treats a document author as the candidate.
Cards are usable only while their recorded document hashes and exact indexed
source excerpts still match. Selection and context fitting may omit whole cards;
they must never truncate the conditions, limitations or cited evidence of a card.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
import json
import hashlib
import math
import os
from pathlib import Path
import re
import threading

from .candidate_profile import personal_question_kind
from .knowledge import tokenize

FACT_SOURCE = "结构化资料（文档提取，非用户确认）"
_GENERIC = set('''what which how why your yours my our main key three two one first second
third describe explain tell please about already achieved result results contribution
contributions research phd doctoral study studies work method methods approach model
models experiment experiments measurement measurements proposed candidate paper published
publication publications exact specific mean compare comparison difference core important
would could can have did does is are the and for from with that this based using used
validation test tests tested scientific evidence document reported supplied under review
measured measuring report reports quantitative
across against according achieve achieved before behind because compared changes change
claim completed connect correct describes discussed disentangle every exact existing
give including investigate leave limited make making number only out parts personally
please prove put same see source specific stronger supplied suppose tell value whether
while would could future proposal proposed prior new project parametersize
'''.split())
_WEAK = set('''reconstruction spectral polarization channel data experiment experimental
experience background candidate reference literature method modelbased net model system
result results training testing evaluation target targets scene scenes study work
first second third one two three four five six seven eight nine ten time fixed
full direct directly supported physically sharp unique mostly rather completed
'''.split())
_REQUIRED_STRINGS = ('id', 'topic', 'kind', 'claim_en', 'owner', 'comparator', 'status')


def _path(value: str) -> str:
    return os.path.normcase(os.path.normpath(value.replace('\\', '/')))


def _terms(text: str) -> set[str]:
    return set(tokenize(text.replace('_', ' '))) - _GENERIC


def _body(card: dict) -> str:
    return ' '.join([card['topic'], card['claim_en'], card['owner'], card['comparator'],
                     *card['conditions'], *card['limitations']])


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False)


@dataclass
class FactSelection:
    cards: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Invalid cards are never sent, but can suppress a stale overlapping profile
    # summary so that expired facts do not quietly return through another input.
    invalid_cards: list[dict] = field(default_factory=list, repr=False)
    library_present: bool = False
    # IDs/counts only. Coverage describes retrieval availability, never that a
    # card logically proves every aspect of a subquestion.
    coverage: list[dict] = field(default_factory=list)


class ResearchFactStore:
    def __init__(self, path: Path, knowledge):
        self.path, self.knowledge = Path(path), knowledge
        self.cards: list[dict] = []
        self.invalid_cards: list[dict] = []
        self.warnings: list[str] = []
        self.library_present = False
        self._file_hash_cache = {}
        self._lock = threading.RLock()

    def _file_digest(self, source: str) -> str:
        """Rehash changed files; an absent/unstable original never validates a card."""
        try:
            path = Path(source).resolve(strict=True)
            stat = path.stat()
            signature = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
            cached = self._file_hash_cache.get(str(path))
            if cached and cached[0] == signature:
                return cached[1]
            with path.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            after = path.stat()
            if signature != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
                return ''
            self._file_hash_cache[str(path)] = (signature, digest)
            return digest
        except (OSError, ValueError):
            return ''

    def refresh(self) -> list[str]:
        with self._lock:
            return self._refresh()

    def _refresh(self) -> list[str]:
        """Read an atomic library snapshot and revalidate against current KB state."""
        self.cards, self.invalid_cards, self.warnings = [], [], []
        self.library_present = self.path.is_file()
        if not self.library_present:
            return []
        try:
            if self.path.stat().st_size > 5 * 1024 * 1024:
                raise ValueError('oversize')
            data = json.loads(self.path.read_text(encoding='utf-8'))
            if not isinstance(data, dict) or data.get('schema_version') != 1:
                raise ValueError('schema')
            documents, cards = data['documents'], data['facts']
            if not isinstance(documents, list) or not isinstance(cards, list) or len(cards) > 500:
                raise ValueError('shape')
            document_hashes = {}
            for doc in documents:
                if (not isinstance(doc, dict) or not isinstance(doc.get('source'), str)
                        or not isinstance(doc.get('sha256'), str)
                        or not re.fullmatch(r'[0-9a-fA-F]{64}', doc['sha256'])):
                    raise ValueError('document')
                key = _path(doc['source'])
                if key in document_hashes:
                    raise ValueError('duplicate_document')
                document_hashes[key] = doc['sha256'].lower()
            with getattr(self.knowledge, '_lock', nullcontext()):
                records = {_path(source): dict(record) for source, record in self.knowledge.document_records.items()}
                passages = {(_path(p.source), p.location, p.passage_id): p.text for p in self.knowledge.passages}
            identities = set()
            for card in cards:
                valid_shape = (isinstance(card, dict)
                    and all(isinstance(card.get(name), str) for name in _REQUIRED_STRINGS)
                    and all(card.get(name, '').strip() for name in ('id', 'topic', 'kind', 'claim_en'))
                    and card.get('provenance') == 'document_derived'
                    and all(isinstance(card.get(name), list)
                            and all(isinstance(value, str) for value in card[name])
                            for name in ('conditions', 'limitations'))
                    and isinstance(card.get('sources'), list) and bool(card['sources']))
                if not valid_shape or card['id'] in identities or len(_json(card)) > 24000:
                    raise ValueError('card_schema')
                identities.add(card['id'])
                usable = True
                for source in card['sources']:
                    if (not isinstance(source, dict) or not all(isinstance(source.get(key), str)
                            and source[key].strip() for key in ('passage_id', 'source', 'location', 'excerpt'))):
                        raise ValueError('source_schema')
                    key = _path(source['source'])
                    expected_hash = document_hashes.get(key)
                    actual_hash = str(records.get(key, {}).get('sha256', '')).lower()
                    original = passages.get((key, source['location'], source['passage_id']))
                    if (not expected_hash or expected_hash != actual_hash
                            or expected_hash != self._file_digest(source['source']) or original is None
                            or source['excerpt'] not in original):
                        usable = False
                # Retain only declared schema fields; embedded extraneous keys
                # are neither instructions nor hidden material sent to a provider.
                clean = {key: card[key] for key in (*_REQUIRED_STRINGS, 'conditions', 'limitations', 'provenance')}
                clean['sources'] = [{key: source[key] for key in ('passage_id', 'source', 'location', 'excerpt')}
                                    for source in card['sources']]
                (self.cards if usable else self.invalid_cards).append(clean)
            if self.invalid_cards:
                self.warnings.append(f'{len(self.invalid_cards)} 张资料事实卡因文档哈希或原文依据变化已停用，请重新核验后更新事实库。')
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            self.cards, self.invalid_cards = [], []
            self.warnings = ['结构化科研事实库读取或校验失败；本次没有使用其中的事实卡，请检查 research_facts.json。']
        return list(self.warnings)

    def select(self, question: str, maximum_chars: int = 9000) -> FactSelection:
        with self._lock:
            return self._select(question, maximum_chars)

    def _select(self, question: str, maximum_chars: int) -> FactSelection:
        self.refresh()
        selection = FactSelection(warnings=list(self.warnings), invalid_cards=list(self.invalid_cards),
                                  library_present=self.library_present)
        from .evidence_gap import split_answer_subquestions
        parts = split_answer_subquestions(question)
        rankings = [self._rank_cards(part) for part in parts]
        whole = self._rank_cards(question)
        # First reserve one complete relevant card per explicit subquestion.
        # Then round-robin additional evidence so a verbose first topic cannot
        # consume all slots. The budget, not a fixed four-card cap, is binding.
        chosen, used = [], 2
        def take(index):
            nonlocal used
            if index in chosen:
                return True
            size = len(_json(self.cards[index])) + 1
            if used + size > maximum_chars:
                return False
            chosen.append(index)
            used += size
            return True
        for ranked in rankings:
            for index in ranked:
                if take(index):
                    break
        # Explicit attribution is useful for personal questions, but never a
        # substitute for the requested experimental comparison or condition.
        if personal_question_kind(question) != 'scientific':
            for topic in dict.fromkeys(self.cards[i]['topic'] for i in chosen):
                owners = [i for i in whole if self.cards[i]['topic'] == topic
                          and self.cards[i]['kind'] in ('personal_contribution', 'personal_experience')]
                for index in owners[:1]:
                    take(index)
        for level in range(max((len(r) for r in rankings), default=0)):
            for ranked in rankings:
                if level < len(ranked):
                    take(ranked[level])
        # Whole-question ranking retains cross-clause context and attribution
        # when a short follow-up uses pronouns rather than repeating the topic.
        for index in whole:
            take(index)
        selection.cards = [self.cards[i] for i in chosen]
        for index, ranked in enumerate(rankings, 1):
            relevant = [self.cards[i]['id'] for i in ranked]
            selected = [self.cards[i]['id'] for i in ranked if i in chosen]
            selection.coverage.append({'question_id': f'Q{index}',
                'candidate_card_ids': relevant, 'selected_card_ids': selected,
                'status': 'selected' if selected else 'budget_omitted' if relevant else 'no_matching_card'})
        missing = [row['question_id'] for row in selection.coverage if row['status'] == 'budget_omitted']
        if missing:
            selection.warnings.append('子问 ' + '、'.join(missing) +
                ' 的完整资料事实卡超出资料预算，尚无已发送事实卡覆盖；没有截断条件或出处。')
        if not selection.cards and whole and not missing:
            selection.warnings.append('相关资料事实卡包含完整条件与出处，超出本题资料预算；本次未截断或发送这些卡。')
        return selection

    def _rank_cards(self, question: str) -> list[int]:
        """Rank complete cards for one request without reading files again."""
        query = _terms(question)
        broad = personal_question_kind(question) != 'scientific' and not query
        # Topic/claim/conditions determine relevance, never filenames or the
        # mere fact that a card lives in the candidate's local directory.
        terms = [_terms(_body(card)) for card in self.cards]
        core_terms = [_terms(' '.join((card['topic'], card['owner'], card['claim_en'])))
                      for card in self.cards]
        anchors = query - _WEAK
        asks_master = bool(re.search(r'\b(?:masters?|msc|m\.sc)\b|硕士', question, re.I))
        counts = {term: sum(term in words for words in terms) for term in query}
        scored = []
        for i, card in enumerate(self.cards):
            if re.search(r'\bmaster', card['topic'].replace('_', ' '), re.I) and not asks_master:
                continue
            overlap = query & terms[i]
            core_overlap = anchors & core_terms[i]
            topic_overlap = anchors & _terms(card['topic'])
            numeric_overlap = {term for term in core_overlap if re.search(r'\d', term)}
            # Matching a caveat, metadata filler, or a single broad imaging word
            # cannot make an unrelated topic relevant. Main claim/owner/topic
            # anchors gate recall; conditions refine ranking only after that.
            if not broad and not (topic_overlap or numeric_overlap or len(core_overlap) >= 2
                                  or (len(anchors) == 1 and core_overlap)):
                continue
            score = sum(math.log1p((len(self.cards) + 1) / (counts[term] + 1))
                        * (2.0 if term in core_overlap else .25 if term in _WEAK else .65)
                        for term in overlap)
            score += 2 * len(topic_overlap)
            if broad:
                score = 12 if card['kind'] == 'personal_contribution' else 0
            if score > 0:
                scored.append((score, i))
        if not scored:
            return []
        highest = max(score for score, _ in scored)
        ranked = [i for score, i in sorted(scored, key=lambda pair: (-pair[0], pair[1]))
                  if score >= highest * .35 or any(re.search(r'\d', term)
                    for term in anchors & core_terms[i])]
        if broad:
            # Overview: prefer one complete contribution per topic before extra
            # metrics or multiple descriptions of the same contribution.
            seen, diverse, extra = set(), [], []
            for index in ranked:
                topic = self.cards[index]['topic']
                if topic in seen:
                    extra.append(index)
                else:
                    seen.add(topic); diverse.append(index)
            ranked = diverse[:3]  # A short overview does not fill spare room with unrelated history/metrics.
        return ranked


def suppress_overlapping_profile(personal: dict, selection: FactSelection) -> tuple[dict, list[str]]:
    """Prefer sourced cards over overlapping summaries, without rewriting user facts.

    This flags overlap, not a proven semantic contradiction. Identity and topic
    entries remain; other user entries remain unless BOTH named source and
    subject overlap an available (or invalidated) card.
    """
    if not selection.cards and not selection.invalid_cards:
        return personal, []
    kept, removed, suppressed_ids = [], 0, []
    for entry in personal.get('entries', []):
        if entry.get('kind') in ('identity', 'topic'):
            kept.append(entry); continue
        text = entry['text']
        claim = re.split(r'\bSources?:', text, maxsplit=1, flags=re.I)[0]
        words = _terms(claim)
        overlap = False
        for card in selection.cards + selection.invalid_cards:
            named_source = any(Path(source['source'].replace('\\', '/')).name.casefold() in text.casefold()
                               for source in card['sources'])
            common = words & _terms(_body(card))
            subject_overlap = (len(common) >= 3 and len(common) / max(1, min(len(words), len(_terms(_body(card))))) >= .20)
            if named_source and subject_overlap:
                overlap = True; break
        if overlap:
            removed += 1
            suppressed_ids.append(entry['id'])
        else:
            kept.append(entry)
    if not removed:
        return personal, []
    updated = {**personal, 'entries': kept,
               'omitted_entries': personal.get('omitted_entries', 0) + removed,
               'suppressed_entry_ids': suppressed_ids,
               'suppressed_entries_fully_covered': False}
    warnings = [f'{removed} 条个人档案摘要与本题文档事实主题重叠或来源已变化，已暂不发送该摘要，避免双份表述冲突；'
                '这不代表事实卡已完整覆盖原条目中的角色或经历。身份及不重叠自填内容保留。']
    return updated, warnings


def fact_evidence(cards: list[dict]):
    """Represent only sent cards/excerpts in the existing evidence UI contract."""
    from .knowledge import Passage
    result, excerpts = [], {}
    for card in cards:
        result.append(Passage(FACT_SOURCE, card['topic'] + ' · ' + card['kind'], _json(card), card['id']))
        for source in card['sources']:
            identity = source['passage_id']
            key = (identity, source['source'], source['location'])
            row = excerpts.setdefault(key, [])
            if source['excerpt'] not in row:
                row.append(source['excerpt'])
    for (identity, source, location), snippets in excerpts.items():
        result.append(Passage(source, location, '\n[...]\n'.join(snippets), identity))
    return result


def rebind_fact_library(path: Path, knowledge) -> dict:
    """Atomically rebind a portable library without weakening content checks.

    Original full-chunk IDs disambiguate overlapping extracts after the source
    path changed. Matching only a familiar filename or a short excerpt is never
    sufficient. Any mismatch leaves the original file untouched.
    """
    from .knowledge import _passage
    path = Path(path)
    report = {'ok': False, 'valid_cards': 0, 'documents': 0, 'network_requests': 0}
    temporary = path.with_name(path.name + '.rebind.tmp')
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('schema_version') != 1 or not isinstance(data.get('documents'), list):
            raise ValueError('invalid_schema')
        mapping = {}
        for doc in data['documents']:
            old_source, digest = doc['source'], doc['sha256']
            matches = [source for source, record in knowledge.document_records.items()
                if record.get('sha256') == digest
                and Path(source.replace('\\', '/')).name.casefold() == Path(old_source.replace('\\', '/')).name.casefold()]
            exact = [source for source in matches if _path(source) == _path(old_source)]
            matches = exact or matches
            if len(matches) != 1:
                raise ValueError('document_relocation_not_unique')
            mapping[_path(old_source)] = matches[0]
            doc.setdefault('original_source', old_source)
            doc['source'] = matches[0]
        for card in data['facts']:
            for source in card['sources']:
                old_source = source['source']
                destination = mapping[_path(old_source)]
                original_source = source.get('original_source', old_source)
                original_id = source.get('original_passage_id', source['passage_id'])
                candidates = [p for p in knowledge.passages
                    if _path(p.source) == _path(destination) and p.location == source['location']
                    and source['excerpt'] in p.text
                    and _passage(original_source, p.location, p.text).passage_id == original_id]
                if len(candidates) != 1:
                    raise ValueError('original_chunk_relocation_not_unique')
                source.update(source=destination, passage_id=candidates[0].passage_id,
                              original_source=original_source, original_passage_id=original_id)
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        store = ResearchFactStore(temporary, knowledge)
        warnings = store.refresh()
        if warnings or store.invalid_cards or len(store.cards) != len(data['facts']):
            raise ValueError('relocated_library_validation_failed')
        temporary.replace(path)
        report.update(ok=True, valid_cards=len(store.cards), documents=len(data['documents']))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        allowed = {'invalid_schema', 'document_relocation_not_unique',
                   'original_chunk_relocation_not_unique', 'relocated_library_validation_failed'}
        reason = str(exc)
        report['error'] = reason if reason in allowed else 'fact_library_rebind_failed'
    finally:
        if temporary.is_file():
            temporary.unlink()
    return report

"""Fast local display restoration for English ASR, with the original preserved.

The small optional punctuation model changes punctuation/case, never the lexical
content.  Deterministic cleanup removes only explicit fillers and duplicate
starts.  This is *not* a scientific fact corrector or a claim that ASR is exact.
No model download, network request or generative LLM call occurs during restore.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
import threading
import time
from typing import Callable
from urllib.request import urlopen


MODEL_DIRECTORY = "edge-punct-en"
MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "punctuation-models/sherpa-onnx-online-punct-en-2024-08-06.tar.bz2")
MODEL_ARCHIVE_SHA256 = "9f5e5a72c7d2829635bd074fce92b6bbd5b78da8a52e7ad8ed1be933f366b99d"

# Apostrophes, hyphens and decimal points are lexical. In particular, 0.95,
# wavelength-dependent and S0 must survive the punctuation stage verbatim.
_TOKEN = re.compile(r"\w+(?:[’']\w+)*(?:[.\-]\w+)*%?", re.UNICODE)
_FILLER = re.compile(r"\b(?:u+h+|u+m+|e+r+m+)\b", re.I)
_PROTECTED_WORDS = {"not", "no", "never", "neither", "nor", "without", "cannot",
                    "uncertain", "unsure", "maybe", "perhaps", "possibly"}
_REPEAT_WORDS = {
    "i", "we", "he", "she", "they", "you", "it", "the", "a", "an", "and", "so",
    "to", "of", "is", "are", "can", "could", "will", "would", "do", "does", "what",
    "how", "which", "when", "where", "why", "this", "these", "those",
}
_INCOMPLETE_END = {"and", "or", "but", "because", "if", "with", "without", "to", "of",
                   "the", "a", "an", "than", "that", "which", "when", "where", "whose",
                   "would", "could", "should", "will", "shall", "can", "may", "might", "must",
                   "my", "your", "our", "their", "his", "her", "its"}
_QUESTION_START = re.compile(
    r"^(?:(?:so|and|then|in particular)[, ]+)?(?:what(?:'s| is| are| do| does| would| can)|"
    r"how(?:'s| is| are| do| does| would| can| could)|why\b|when\b|where\b|which\b|"
    r"could you\b|can you\b|would you\b|do you\b|does\b|is there\b|are there\b)", re.I)
_ABBREVIATIONS = {"e.g.", "i.e.", "etc.", "dr.", "prof.", "mr.", "mrs.", "ms.",
                  "vs.", "fig.", "eq.", "approx.", "al."}


def _is_explicit_question(value: str) -> bool:
    if _QUESTION_START.search(value):
        return True
    if re.search(r"(?:^|[,;])\s*(?:how|why|when|where)\s+"
                 r"(?:would|could|should|can|will|do|did|does|is|are)\s+"
                 r"(?:you|we|they|it|the)\b", value, re.I):
        return True
    # Direct question after a conditional preamble, e.g. "In particular, if
    # scattering is strong, which channels would you use?". An ordinary
    # uncertainty statement ("I do not know which channels we use") stays one.
    return bool(re.search(r"(?:^|[,;])\s*(?:which|what|how many|how much)\s+"
                          r"[^.!?]{0,100}\b(?:would|could|should|can|will|do|did)\s+you\b",
                          value, re.I))


@dataclass(frozen=True)
class RestoredUtterance:
    raw: str
    clean: str
    sentences: tuple[str, ...]
    changes: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    engine: str = "rules"
    elapsed_seconds: float = 0.0

    @property
    def display(self) -> str:
        """One sentence per line, suitable for a live caption card."""
        return "\n".join(self.sentences)


def _tokens(value: str) -> list[str]:
    return [m.group() for m in _TOKEN.finditer(value)]


def _key(value: str) -> str:
    return value.casefold().replace("’", "'")


def clean_disfluencies(value: str) -> str:
    """Conservative lexical editing; retains negation, numbers and uncertainty.

    Repeated content words can be deliberate emphasis ("very very", "not not",
    "had had"). They are preserved. Longer duplicate starts require at least
    two words, no number/negation, and no punctuation marking a real turn.
    """
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    # Whisper can split this fixed lexical compound at an acoustic cut. Keep
    # the repair specific to end-to-end; mathematical minus signs stay intact.
    value = re.sub(r'\b(end)(?:\.\s*|\s+)-\s*to\s*-\s*end\b',
                   lambda m: m[1] + '-to-end', value, flags=re.I)
    value = re.sub(r'\b(end)\s*-\s*to(?:\.\s*|\s+)-\s*end\b',
                   lambda m: m[1] + '-to-end', value, flags=re.I)
    value = _clean_nominal_restarts(value)
    def remove_filler(match):
        # In a scientific transcript "5 um" is commonly a micrometre unit,
        # while "ERM" may be an acronym. Neither is a filled pause.
        if re.search(r"\d(?:[.,]\d+)?\s*$", value[:match.start()]):
            return match.group()
        if len(match.group()) >= 3 and match.group().isupper():
            return match.group()
        return ""
    value = _FILLER.sub(remove_filler, value)
    # A cut-off syllable only, not a valid scientific hyphenated word.
    value = re.sub(r"\b([A-Za-z]{1,3})-\s+(?=\1[A-Za-z])", "", value, flags=re.I)
    for _ in range(4):
        previous = value
        words = list(_TOKEN.finditer(value))
        for length in range(7, 1, -1):
            found = False
            for i in range(len(words) - 2 * length + 1):
                first, second = words[i:i + length], words[i + length:i + 2 * length]
                a, b = [_key(x.group()) for x in first], [_key(x.group()) for x in second]
                if a != b or any(any(ch.isdigit() for ch in token) or token in _PROTECTED_WORDS
                                 or token.endswith("n't") for token in a):
                    continue
                gap = value[first[-1].end():second[0].start()]
                interior = value[first[0].start():first[-1].end()]
                if re.search(r"[.!?;:]", gap + interior):
                    continue
                value = value[:first[0].start()] + value[second[0].start():]
                found = True
                break
            if found:
                break
        # Only high-confidence function-word stutters, not arbitrary doubles.
        def dedupe(match):
            return match[1] if _key(match[1]) in _REPEAT_WORDS else match.group()
        value = re.sub(r"\b([A-Za-z]+)(?:[ ,]+\1\b)+", dedupe, value, flags=re.I)
        if value == previous:
            break
    # "I mean" may signal a correction, and must stay unless it occurs solely
    # inside an abandoned exact repeated start, e.g. "what is ... I mean what is".
    value = re.sub(
        r"\b(what(?:'s| is| are) the \w+)\s+(?:how\s+)?(?:yeah\s+)?i mean\s+(?=\1\b)",
        "", value, flags=re.I)
    # Remove parenthetical 'you know', never an actual question/condition such as
    # 'Do you know?', 'if you know', 'what you know' or 'you know the result'.
    value = re.sub(r"\band you know\s+(?=so\b)", "", value, flags=re.I)
    value = re.sub(r"\bis\s+you know\s+(?=\w+ing\b)", "is ", value, flags=re.I)
    value = re.sub(r"(?<=[,;])\s*you know\s*[,;]", " ", value, flags=re.I)
    value = re.sub(r"\s+([,.;!?])", r"\1", value)
    value = re.sub(r"([,;])(?:\s*[,;])+", r"\1", value)
    value = re.sub(r"\s+", " ", value).strip(" ,;")
    return value


def correct_obvious_grammar(value: str) -> str:
    """Narrow agreement fixes; no scientific, pronoun, tense or polarity edits.

    Broad paraphrases are deliberately left to the optional display-only worker.
    In particular, subjunctive forms such as ``if I were`` are never changed.
    """
    substitutions = [
        (r"\b(we|you|they)\s+was\b", "were"),
        (r"\b(we|you|they)\s+is\b", "are"),
        (r"\b(we|you|they)\s+has\b", "have"),
        (r"\b(I)\s+is\b", "am"),
        (r"\b(I)\s+has\b", "have"),
        (r"\b((?:the|this|our) (?:model|camera|system|method|operator|matrix)|it|he|she)\s+don't\b",
         "doesn't"),
        (r"\b((?:the|this|our) (?:model|camera|system|method|operator|matrix)|it|he|she)\s+do not\b",
         "does not"),
        (r"\b((?:these|the|our) (?:models|cameras|systems|methods|operators|matrices)|we|you|they)\s+doesn't\b",
         "don't"),
    ]
    for pattern, replacement in substitutions:
        value = re.sub(pattern, lambda match: match[1] + " " + replacement, value, flags=re.I)
    return value


def repair_fragment_boundaries(value: str) -> str:
    """Join strongly incomplete ASR cuts without rewriting scientific content.

    A pause is not a sentence boundary.  These rules repair function-word cuts
    and a few unambiguous spoken constructions; they are deliberately not a
    blanket removal of short sentences or capitalization.  In particular US,
    SO, AA and other possible abbreviations stay untouched outside a narrowly
    grammatical context. The caller retains the exact original separately.
    """
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    value = _repair_predicate_boundaries(value)
    # Edge-Punct sometimes splits a grammatical prefix into an isolated word,
    # even when Whisper supplied an intact sentence. Match ordinary casing:
    # scientific IF/HOW/IT and single-letter equations must stay protected.
    value = re.sub(r"\b(All|all),\s+(right)\b", r"\1 \2", value)
    value = re.sub(
        r"\b(What|what|How|how|Why|why|When|when|Where|where|Which|which)[,.]\s+"
        r"((?i:is|are|was|were|do|does|did|can|could|would|will|should|has|have))\b",
        lambda m: m[1] + " " + m[2].lower(), value)
    value = re.sub(
        r"\b(Can|can|Could|could|Would|would|Will|will|Should|should|Do|do|Does|does|Did|did),\s+"
        r"((?i:you|we|they|he|she|it|I))\b",
        lambda m: m[1] + " " + ("I" if m[2].casefold() == "i" else m[2].lower()), value)
    value = re.sub(
        r"\b(I|You|you|We|we|They|they|He|he|She|she|It|it)\.\s+"
        r"((?i:am|is|are|was|were|can(?:['’]t)?|cannot|could(?:n['’]t)?|"
        r"would(?:n['’]t)?|will|won['’]t|should(?:n['’]t)?|do(?:n['’]t)?|"
        r"does(?:n['’]t)?|did(?:n['’]t)?|have|has|had))\b",
        lambda m: m[1] + " " + m[2].lower(), value)
    value = re.sub(
        r"\b(If|if)\.\s+((?i:the|a|an|this|that|these|those|we|you|they|he|she|it|I|our|my|your))\b",
        lambda m: m[1] + " " + ("I" if m[2].casefold() == "i" else m[2].lower()), value)
    # "Let. US play" is the pronoun in an imperative, unlike "US researchers".
    value = re.sub(r"\blet[.!?]\s+us\s+(?=(?:play|start|begin|look|consider|take|"
                   r"try|discuss|compare|assume|suppose|say|continue|move|go)\b)",
                   "let us ", value, flags=re.I)
    # A cut inside a direct auxiliary question, including a choice joined by or.
    value = re.sub(r"\b(or|and)\s+(do|does|did|can|could|would|will|should)[.!?]\s+"
                   r"(you|we|they|he|she|it|I)\b",
                   lambda m: m[1] + " " + m[2].lower() + " "
                   + ("I" if m[3].casefold() == "i" else m[3].lower()), value, flags=re.I)
    value = re.sub(r"(^|[.!?]\s+)(do|does|did|can|could|would|will|should)[.!?]\s+"
                   r"(you|we|they|he|she|it|I)\b",
                   lambda m: m[1] + m[2] + " "
                   + ("I" if m[3].casefold() == "i" else m[3].lower()), value, flags=re.I)
    # Determiner-led continuation after a stranded preposition is sufficiently
    # specific to avoid joining ordinary complete sentences ending "looked up".
    value = re.sub(r"\b(at|of|to|with|without|from|into|between|among|than)[.!?]\s+"
                   r"(the|a|an|this|that|these|those|my|your|our|their)\b",
                   lambda m: m[1] + " " + m[2].lower(), value, flags=re.I)
    value = re.sub(r"\b(the|a|an)[.!?]\s+(?=[A-Za-z])", r"\1 ", value, flags=re.I)
    # "One model. Each, if ..." means one model each; no words are reordered.
    value = re.sub(r"\b(model|name|example|question|answer|turn|item)[.!?]\s+each\s*,\s*if\b",
                   lambda m: m[1] + " each. If", value, flags=re.I)
    value = re.sub(r"\b(model|name|example|question|answer|turn|item)[.!?]\s+each[.!?](?=\s|$)",
                   lambda m: m[1] + " each.", value, flags=re.I)
    # Discourse SO before an addressed question or an explicit "let us" is not
    # a technical acronym. Other uppercase words remain protected.
    value = re.sub(r"\bSO(?=\s*,?\s*(?:let us|do you|would you|can you)\b)", "So", value)
    value = re.sub(r"\b([Ss]o|[Oo]kay|[Ww]ell)\s+(?=(?:do you|would you|can you)\b)", r"\1, ", value)
    value = re.sub(r"\b([Oo]kay|[Ww]ell),\s+so,\s+", r"\1, so ", value)
    value = re.sub(r"\b(start|begin|go first)\s+or\s+(?=(?:do|would) you|should I)",
                   r"\1, or ", value, flags=re.I)
    value = re.sub(r"\b(start|begin|go first)[.!?]\s+or\s+"
                   r"(?=(?:(?:do|would) you (?:want|like|prefer)(?: me)? to|"
                   r"(?:should|shall|can) I) (?:start|begin|go first)\b)",
                   r"\1, or ", value, flags=re.I)
    # A learned mark after a discourse conjunction is not another sentence.
    # Match ordinary source casing only; SO may be a technical abbreviation.
    value = re.sub(r"\b(So|so|And|and|But|but)[.!?]\s+"
                   r"((?i:what|why|how|which|where|when|do you|can you|would you))\b",
                   lambda m: m[1] + ", " + m[2].lower(), value)
    value = re.sub(r'\b(And|and|But|but)\.\s+(Before|before|After|after)\s+'
                   r'(?=(?:we|you|they|he|she|it|I|the|a|an|our|this|that)\b)',
                   lambda m: m[1] + ' ' + m[2].lower() + ' ', value)
    value = re.sub(r"\b([Ff]or example)[.!?]\s+"
                   r"((?i:the|a|an|this|that|particular|we|you|it|our|my|if))\b",
                   lambda m: m[1] + ", " + m[2].lower(), value)
    return value


# These are syntactic hints, not a domain dictionary or a grammatical parser.
# An arbitrary short sentence is not evidence that a pause should be removed.
_PARTICIPLES = {
    "been", "built", "chosen", "done", "drawn", "driven", "found", "given",
    "grown", "held", "kept", "known", "learnt", "lost", "made", "put", "read",
    "run", "seen", "sent", "set", "shown", "split", "taken", "taught",
    "understood", "written",
}
_PREDICATE_MODIFIER = r"(?:not|never|already|also|still|just|almost|even|back|forward|[A-Za-z]+ly)"
_VERBAL_WORD = r"[A-Za-z]+(?:-[A-Za-z]+)*"
_ADJECTIVE = r"(?:[A-Za-z-]+(?:ical|al|ary|ive|ous|ic|less|ful|able|ible)|linear|neural|inverse)"
_AMBIGUOUS_ADJECTIVES = {
    'alternative', 'chemical', 'commercial', 'derivative', 'individual', 'inverse', 'manual',
    'material', 'narrative', 'normal', 'objective', 'official', 'potential',
    'primary', 'radical', 'rational', 'relative', 'secondary', 'signal', 'variable',
}


def _continuation_case(word: str) -> str:
    """Undo ordinary sentence-initial case without altering source symbols."""
    if (word == "I" or any(ch.isdigit() for ch in word) or word.isupper()
            or any(ch.isupper() for ch in word[1:]) or not word.isascii()):
        return word
    return word[:1].lower() + word[1:]


def _repair_predicate_boundaries(value: str) -> str:
    """Repair only locally supported predicate/noun-phrase cuts.

    This runs both before and after the optional punctuation model. Learned
    punctuation can split ``are. Completely, back-propagated``; a plural
    copula followed by an adverb and a participle is substantially stronger
    evidence than the size of either fragment. Question/exclamation boundaries
    and explicit subject-led new clauses remain separate. No words are added,
    removed, reordered or technically corrected here.
    """
    # A comma between the pronoun and its verb is not a discourse pause. Keep
    # the comma after "mean", which distinguishes a restart from "I mean it.".
    value = re.sub(r"\b(I|i),\s+(Mean|mean)\b", r"\1 mean", value)
    # Retain an audible abandoned pronoun rather than deleting it to manufacture
    # fluent speech: "I mean, it—basically, all the weights ...". A complete
    # "I mean it. Basically ..." deliberately does not match this rule.
    value = re.sub(
        r"\b(I mean),\s+(it|we|they|he|she)\.\s+"
        r"(Basically|Actually|Essentially|basically|actually|essentially),\s+"
        r"(?=(?:all|the|these|those|both|our|my|your|their)\b)",
        lambda m: m[1] + ", " + m[2] + "—" + m[3].lower() + ", ", value)

    predicate = re.compile(
        r"\b(?P<aux>am|is|are|was|were|be|been|being|have|has|had)"
        r"(?P<neg>\s+not)?\.\s+"
        r"(?P<mod>(?:" + _PREDICATE_MODIFIER + r"[, .]*\s+){0,3})"
        r"(?P<verb>" + _VERBAL_WORD + r")\b", re.I)

    def connect_predicate(match):
        verb = match['verb']
        lower = verb.casefold()
        participle = lower in _PARTICIPLES or lower.endswith('ed')
        # With no modifier, an -ing form may be the next sentence's subject
        # ("They are. Learning helps ..."). Keep that ambiguous case unchanged.
        progressive = (lower.endswith('ing') and bool(match['mod'])
                       and match['aux'].casefold() not in {'have', 'has', 'had'})
        if not (participle or progressive):
            return match.group()
        # A new nominal subject can itself look like a participle: "They are.
        # Learning is difficult." / "We have. Increased costs concern us.".
        # Leave these ambiguous boundaries alone rather than swallowing a claim.
        tail = value[match.end():]
        if _starts_finite_clause(tail, max_subject_words=2):
            return match.group()
        if _protected_case(verb) and not ('-' in verb and not verb.isupper()):
            return match.group()
        modifiers = re.sub(r"[, .]+", " ", match['mod']).strip()
        modifiers = " ".join(_continuation_case(word) for word in modifiers.split())
        parts = [match['aux'] + (match['neg'] or ''), modifiers, _continuation_case(verb)]
        return ' '.join(part for part in parts if part)

    value = predicate.sub(connect_predicate, value)
    # Remove an intrapredicate comma even when the full stop was already fixed.
    # This is deliberately limited to copula + adverb + participle; a sentence
    # starting "Clearly, we ..." still has its normal discourse comma.
    value = re.sub(
        r"\b(am|is|are|was|were|be|been|being)\s+"
        r"((?:" + _PREDICATE_MODIFIER + r"(?:,\s*|\s+)){1,3})(" + _VERBAL_WORD + r")\b",
        lambda m: (m[1] + ' ' + ' '.join(_continuation_case(word) for word in
                                        re.sub(r'[, ]+', ' ', m[2]).split()) + ' ' + m[3]
                   if ',' in m[2] and (m[3].casefold().endswith(('ed', 'ing'))
                                      or m[3].casefold() in _PARTICIPLES)
                   else m.group()), value, flags=re.I)

    # A singular indefinite determiner followed only by modifiers cannot end
    # at an adjective: "a particular mathematical. Operator". Do not apply
    # this to standalone adjectives, noun/adjective ambiguity ("the inverse"),
    # or explicit question/exclamation boundaries.
    noun_phrase = re.compile(
        r"\b(?P<prefix>(?:a|an|another|each|every)\s+"
        r"(?:(?:particular|different|single|very|more|less|such|" + _ADJECTIVE + r")\s+){0,2}"
        + _ADJECTIVE + r")\.\s+(?P<next>" + _VERBAL_WORD + r")\b", re.I)
    clause_start = {'i', 'we', 'you', 'he', 'she', 'it', 'they', 'this', 'that', 'these', 'those',
                    'what', 'how', 'why', 'when', 'where', 'which', 'but', 'and', 'or', 'so',
                    'yes', 'no', 'okay', 'ok', 'thanks', 'thank', 'is', 'are', 'can', 'could',
                    'will', 'would', 'do', 'does', 'did', 'a', 'an', 'the'}
    value = noun_phrase.sub(
        lambda m: (m.group() if m['next'].casefold() in clause_start or _protected_case(m['next'])
                   or _tokens(m['prefix'])[-1].casefold() in _AMBIGUOUS_ADJECTIVES
                   or _starts_finite_clause(value[m.end('next'):], max_subject_words=2)
                   else m['prefix'] + ' ' + _continuation_case(m['next'])), value)
    # Also repair an observed continuation after a transitive phrase such as
    # "use neural. Networks". The same conservative helper drives card sealing,
    # so keeping one card cannot still leave a false sentence inside that card.
    attributes = re.compile(r'(?P<last>\b' + _ADJECTIVE + r')\.\s+'
                            r'(?P<next>' + _VERBAL_WORD + r')\b', re.I)
    value = attributes.sub(
        lambda m: (m['last'] + ' ' + _continuation_case(m['next'])
                   if _attributive_continuation(value[:m.end('last')], value[m.start('next'):])
                   else m.group()), value)
    # An acoustic cut can split a coordinated subject before its first finite
    # verb: "all the weights in the network. And the decoder are learned".
    # Ordinary "And ..." sentences with a completed preceding predicate must
    # remain independent; the subject-only grammar check is shared with sealing.
    value = re.sub(
        r'\.\s+(And|and)\s+',
        lambda m: (' and ' if _nominal_coordination_continues(
            value[:m.start()], value[m.start(1):]) else m.group()), value)
    return value


_CONTINUATION_NOUNS = {
    'algorithm', 'architecture', 'attention', 'calibration', 'channel', 'component',
    'constraint', 'distribution', 'equation', 'estimation', 'field', 'function',
    'image', 'information', 'loss', 'mapping', 'material', 'matrix', 'measurement',
    'method', 'model', 'network', 'operator', 'optimization', 'parameter', 'problem',
    'process', 'reconstruction', 'representation', 'space', 'system', 'term',
}
_SUBJECT_NOUNS = _CONTINUATION_NOUNS | {
    'bias', 'camera', 'coefficient', 'decoder', 'detector', 'encoder', 'feature',
    'head', 'input', 'layer', 'neuron', 'output', 'pixel', 'ray', 'sample', 'sensor',
    'tensor', 'vector', 'weight',
}
_FINITE_WORD = (
    r'(?:am|is|are|was|were|be|been|being|have|has|had|do|does|did|can|could|'
    r'will|would|should|shall|must|may|might|'
    r'use[sd]?|learn[st]?|predicts?|works?|remain[sd]?|appear[sd]?|follow[sd]?|'
    r'include[sd]?|contain[sd]?|show[sn]?|yield[sd]?|need[sd]?|require[sd]?|'
    r'change[sd]?|differ[sd]?|fail[sd]?|perform[sd]?|depend[sd]?|converge[sd]?|'
    r'collapse[sd]?|concern[sd]?|mean[st]?)')
_SUBJECT_FINITE = re.compile(
    r'\b(?:' + _FINITE_WORD + r'|that|which|who|when|where|because|if)\b', re.I)


def _starts_finite_clause(value: str, *, max_subject_words: int = 0) -> bool:
    # A nearby finite verb after a nominal head makes it a new clause. Do not
    # scan through a preposition/infinitive ("known to require calibration")
    # or another sentence merely to find a verb farther away.
    subject_word = r'(?!(?:to|by|for|in|on|at|with|without|from|of|and|or)\b)[A-Za-z-]+\s+'
    pattern = (r'^\s*(?:' + subject_word + '){0,' + str(max_subject_words) + '}'
               + _FINITE_WORD + r'\b')
    return bool(re.match(pattern, value, re.I))


def _nominal_subject(value: str) -> bool:
    """Recognize a small subject-NP grammar, not arbitrary verbless English.

    Each noun phrase must end in an observed technical noun. Unknown heads,
    relative clauses, possible finite predicates and ambiguous noun modifiers
    are conservative misses. This bounds the risk of merging complete claims.
    """
    value = value.strip(' ,;:.')
    # These are removed only in the syntactic *test*. The visible/source text
    # keeps all spoken words, including an abandoned "it" after "I mean".
    value = re.sub(r'^I\s*,?\s*mean\s*,?\s+(?:it\s*,?\s+)?', '', value, flags=re.I)
    value = re.sub(r'^(?:(?:so|well|basically|actually|essentially)\s*[, ]+)+', '', value, flags=re.I)
    if _SUBJECT_FINITE.search(value) or re.search(r'[.!?;:]', value):
        return False
    groups = re.split(r'\s+(?:and|in|on|at|of|for|from|with|within|under|over|between)\s+',
                      value, flags=re.I)
    # A quantifier's "of" belongs to its initial noun phrase, not an empty PP.
    if len(groups) > 1 and groups[0].casefold() in {'all', 'some', 'most', 'many', 'both', 'each', 'any'}:
        groups[1] = groups[0] + ' ' + groups[1]
        groups = groups[1:]
    if not groups or len(groups) > 8:
        return False
    modifier_exceptions = {'synthesis', 'physics', 'stokes', 'loss', 'learning', 'imaging',
                           'rendering', 'sensing', 'scattering', 'training', 'encoding', 'decoding'}
    for group in groups:
        words = [word.casefold() for word in _tokens(group)]
        while words and words[0] in {'all', 'some', 'most', 'many', 'both', 'each', 'every', 'any',
                                     'the', 'a', 'an', 'this', 'these', 'those', 'our', 'your',
                                     'their', 'my', 'his', 'her', 'its'}:
            words.pop(0)
        if not words or len(words) > 7:
            return False
        noun = words[-1][:-1] if words[-1].endswith('s') else words[-1]
        if noun not in _SUBJECT_NOUNS:
            return False
        for word in words[:-1]:
            if word in {'the', 'a', 'an', 'our', 'your', 'their', 'my', 'his', 'her', 'its'}:
                return False
            # "the model corrupts images" is a complete clause, unlike the
            # compound noun "the motion synthesis network". Unknown -s/-ed
            # modifiers are ambiguous and therefore not automatically joined.
            if (word.endswith(('s', 'ed', 'ing')) and word not in modifier_exceptions):
                return False
    return True


def _nominal_coordination_continues(previous: str, incoming: str) -> bool:
    if previous.rstrip().endswith(('?', '!')):
        return False
    sentences = split_sentences(previous)
    if not sentences or not _nominal_subject(sentences[-1]):
        return False
    match = re.match(r'^and\s+(?P<subject>[^.!?;:]{1,180}?)\s+'
                     r'(?:am|is|are|was|were|have|has|had|can|could|will|would|should|must)\b',
                     incoming.strip(), re.I)
    return bool(match and _nominal_subject(match['subject']))


def _clean_nominal_restarts(value: str) -> str:
    """Remove only an exact nominal restart explicitly marked by an ellipsis.

    A repeated content word can be emphasis. Restrict this exception to a
    subject without a predicate, an exact repeated ending, and a following
    prepositional continuation. Numbers, negation and source acronyms never
    qualify. The caller retains the untouched raw speech separately.
    """
    for marker in reversed(list(re.finditer(r'\.{3,}|…', value))):
        before = value[:marker.start()]
        parts = split_sentences(before)
        if not parts or not _nominal_subject(parts[-1]):
            continue
        left = list(_TOKEN.finditer(before))
        after = value[marker.end():]
        right = list(_TOKEN.finditer(after))
        for size in range(min(len(left), len(right), 6), 0, -1):
            original, repeated = left[-size:], right[:size]
            keys = [_key(word.group()) for word in original]
            if keys != [_key(word.group()) for word in repeated]:
                continue
            if any(word in _PROTECTED_WORDS or word.endswith("n't") or
                   any(char.isdigit() for char in word) for word in keys):
                continue
            if any(_protected_case(word.group()) for word in original):
                continue
            if not re.match(r'\s+(?:in|on|at|of|for|from|with|within|under|over|between)\b',
                            after[repeated[-1].end():], re.I):
                continue
            kept = before[original[0].start():original[-1].end()]
            value = (value[:original[0].start()] + kept +
                     after[repeated[-1].end():])
            break
    return value


def caption_tail_incomplete(value: str) -> bool:
    """Conservative lexical tail hint; silence and speaker identity stay external.

    This deliberately does not classify every auxiliary as incomplete: "Yes,
    they are." can be a complete answer. It recognizes stranded function words,
    coordinated auxiliaries, and an attributive adjective with an observed
    determiner/transitive-verb prefix. It does not infer any missing noun.
    """
    value = str(value or '').strip().rstrip('.!?,;: ')
    words = _tokens(value)
    if not words:
        return False
    if _key(words[-1]) in _INCOMPLETE_END:
        return True
    if re.search(r'\b(?:and|or|but)\s+(?:am|is|are|was|were|be|been|being|have|has|had)'
                 r'(?:\s+not)?$', value, re.I):
        return True
    if re.search(r'\bI,?\s+mean,\s+(?:it|we|they|he|she)$', value, re.I):
        return True
    parts = split_sentences(value)
    tail = parts[-1] if parts else ''
    if (len(_tokens(tail)) >= 6 and _nominal_subject(tail)
            and re.search(r'\b(?:all|some|most|many|both|each)\s+(?:of\s+)?'
                          r'(?:the|our|these|those|your|their)\b', tail, re.I)):
        return True
    if _key(words[-1]) in _AMBIGUOUS_ADJECTIVES:
        return False
    modifier = r'(?:particular|different|single|very|more|less|such|' + _ADJECTIVE + r')'
    return bool(re.search(
        r'\b(?:a|an|another|each|every|use|uses|using|apply|applies|applying|'
        r'learn|learns|learning|fit|fits|fitting|train|trains|training|build|builds|building|'
        r'design|designs|designing|choose|chooses|choosing|select|selects|selecting)\s+'
        r'(?:' + modifier + r'\s+){0,3}' + _ADJECTIVE + r'$',
        value, re.I))


def sentence_continues(previous: str, incoming: str) -> bool:
    """Whether an observed same-speaker continuation strongly repairs a cut.

    The caller must first establish speaker/stream continuity; this helper has
    no audio or speaker evidence. It is a bounded hint for delaying a caption
    paragraph seal, not a semantic completeness detector. It never modifies
    ``previous`` or ``incoming`` and cannot invent unheard continuation words.

    Predicate evidence reuses the text repair rules. Attributive adjective +
    noun evidence uses an explicit, intentionally small technical noun list;
    unknown nouns and complete copular predicates remain conservative misses.
    """
    previous, incoming = str(previous or '').strip(), str(incoming or '').strip()
    if not previous or not incoming or previous.endswith(('?', '!')):
        return False
    left, right = _tokens(previous), _tokens(incoming)
    if not left or not right:
        return False
    # Only consider the seam. A different internal repair must not accidentally
    # make this predicate true; all lexical token positions remain unchanged.
    combined = previous + ' ' + incoming
    repaired = _repair_predicate_boundaries(combined)
    repaired_tokens = list(_TOKEN.finditer(repaired))
    if ([_key(m.group()) for m in repaired_tokens] == [_key(w) for w in left + right]
            and re.search(r'[.!?]\s*$', previous)):
        seam = repaired[repaired_tokens[len(left) - 1].end():repaired_tokens[len(left)].start()]
        if not re.search(r'[.!?]', seam):
            return True
    return (_nominal_coordination_continues(previous, incoming)
            or _attributive_continuation(previous, incoming))


def _attributive_continuation(previous: str, incoming: str) -> bool:
    left, right = _tokens(previous), _tokens(incoming)
    if not left or not right:
        return False
    last, first = left[-1], right[0].casefold()
    noun = first[:-1] if first.endswith('s') else first
    if (noun not in _CONTINUATION_NOUNS or not re.fullmatch(_ADJECTIVE, last, re.I)
            or last.casefold() in _AMBIGUOUS_ADJECTIVES):
        return False
    if len(left) < 2 or _protected_case(last):
        return False
    # Require positive evidence of an unfinished attributive construction.
    # A blacklist of copula/adverb combinations missed "purely spectral" and
    # "entirely mathematical", which can already be complete predicates.
    if not caption_tail_incomplete(previous):
        return False
    first_match = _TOKEN.search(incoming)
    if first_match and _starts_finite_clause(incoming[first_match.end():], max_subject_words=2):
        return False
    return True


def split_sentences(value: str) -> tuple[str, ...]:
    """Split restored prose without splitting decimals, equations or abbreviations."""
    result: list[str] = []
    start = 0
    for match in re.finditer(r"[.!?]+(?=\s|$)", value):
        end = match.end()
        prefix = value[start:end]
        last = prefix.split()[-1].casefold() if prefix.split() else ""
        if last in _ABBREVIATIONS:
            continue
        if re.search(r"\b[A-Z]\.$", prefix) and end < len(value):
            continue
        part = value[start:end].strip()
        if part:
            result.append(part)
        start = end
    tail = value[start:].strip()
    if tail:
        result.append(tail)
    return tuple(result)


def _protected_case(token: str) -> bool:
    letters = [c for c in token if c.isalpha()]
    return (any(c.isdigit() for c in token) or "-" in token or
            any(ord(c) > 127 for c in letters) or
            (len(letters) > 1 and all(c.isupper() for c in letters)) or
            (any(c.isupper() for c in token[1:])) or
            (len(token) == 1 and token not in {"a", "I", "i"}))


def _punctuation_projection(source: str, candidate: str) -> str | None:
    """Reject lexical edits; restore source spelling for protected tokens."""
    source_matches = list(_TOKEN.finditer(source))
    original = [m.group() for m in source_matches]
    matches = list(_TOKEN.finditer(candidate))
    if [_key(x) for x in original] != [_key(m.group()) for m in matches]:
        return None
    # Lexical matching alone does not catch a removed minus, equality sign, or
    # a comma removed from "1,000". Preserve those scientific quantities too.
    number = r"(?<!\w)[+-]?\d+(?:[.,]\d+)*(?:[eE][+-]?\d+)?%?"
    if re.findall(number, source) != re.findall(number, candidate):
        return None
    signs = r"[=<>±×÷∝≈≤≥∈]"
    if re.findall(signs, source) != re.findall(signs, candidate):
        return None
    fragments = []
    cursor = 0
    for index, (token, match) in enumerate(zip(original, matches)):
        source_gap = source[source_matches[index - 1].end():source_matches[index].start()] if index else ''
        keep_ellipsis = (index > 0 and not original[index - 1][-1:].isdigit()
                         and re.fullmatch(r'\s*(?:\.{3,}|…)\s*', source_gap))
        fragments.append('… ' if keep_ellipsis else candidate[cursor:match.start()])
        is_equation_i = token == "i" and bool(re.search(
            r"(?:=|equals|matrix|operator|vector|[A-Z])\s*$",
            source[:source_matches[index].start()], re.I)
            or re.match(r"\s*(?:=|\bequals\b)", source[source_matches[index].end():]))
        # The tiny casing model occasionally turns grammatical pronouns and
        # discourse 'So' into bogus all-capital abbreviations (ME / SO / US).
        source_grammar_case = _key(token) in {'so', 'me', 'us'} and not token.isupper()
        # Casing restoration must never invent a technical abbreviation. In
        # real Zoom captions ordinary "all/if/how/it" became ALL/IF/HOW/IT and
        # changed the subsequent translation's meaning. Keep source spelling
        # for any newly all-capital word; existing source acronyms still win.
        model_invented_acronym = (len(match.group()) > 1 and match.group().isupper()
                                  and not token.isupper())
        fragments.append(token if (_protected_case(token) or is_equation_i
                                   or source_grammar_case or model_invented_acronym) else match.group())
        cursor = match.end()
    source_tail = source[source_matches[-1].end():] if source_matches else ''
    keep_tail = (original and not original[-1][-1:].isdigit()
                 and re.fullmatch(r'\s*(?:\.{3,}|…)\s*', source_tail))
    fragments.append('…' if keep_tail else candidate[cursor:])
    return "".join(fragments)


def _keep_existing_sentence_boundaries(source: str, candidate: str) -> str:
    """Do not repunctuate short, already complete ASR sentences as fragments.

    Edge-Punct is useful on unpunctuated runs, but on existing English prose it
    sometimes adds a full stop every few words ("If I say one model. You say").
    Keep source punctuation/case for complete 2--40 word sentences. Short
    questions such as "Could you explain?" are particularly prone to invented
    commas, and need the same protection as longer sentences. Longer
    run-ons and genuinely unpunctuated spans still receive model restoration;
    clear incomplete boundaries have already been joined by the rules above.
    The rule does not infer that the ASR's words or technical claims are true.
    """
    matches = list(_TOKEN.finditer(candidate))
    replacements = []
    token_index = 0
    for sentence in split_sentences(source):
        words = list(_TOKEN.finditer(sentence))
        count = len(words)
        if not count:
            continue
        if (2 <= count <= 40 and (count >= 5 or not re.search(r"[,;:]", sentence))
                and re.search(r"[.!?]$", sentence)
                and sentence[0].isalnum() and token_index + count <= len(matches)):
            start = matches[token_index].start()
            end = (matches[token_index + count].start()
                   if token_index + count < len(matches) else len(candidate))
            replacements.append((start, end, sentence + (" " if end < len(candidate) else "")))
        token_index += count
    for start, end, replacement in reversed(replacements):
        candidate = candidate[:start] + replacement + candidate[end:]
    return candidate


def _normalize_punctuation(value: str, final: bool) -> str:
    # The punctuation model receives Whisper text that may already have commas
    # and full stops. It can append its own mark to that existing mark (",.",
    # "?.", "??"). Resolve the entire cluster before sentence splitting or a
    # question conversion, which otherwise turns "?." into another "??".
    def one_mark(match):
        cluster = match.group()
        start, end = match.span()
        # Do not reinterpret an ambiguous numeric literal such as "1,.000".
        if (start > 0 and end < len(value) and value[start - 1].isdigit()
                and value[end].isdigit()):
            return cluster
        if (re.fullmatch(r'\.{3,}', cluster)
                and not (start > 0 and value[start - 1].isdigit())):
            return '…'
        for mark in ("?", "!", ".", ";", ":", ","):
            if mark in cluster:
                return mark
        return cluster
    value = re.sub(r"[,.;:!?](?:[ \t]*[,.;:!?])+", one_mark, value)
    value = re.sub(r"\s+([,.;!?])", r"\1", value)
    value = repair_fragment_boundaries(value)
    # Small punctuation models occasionally strand the final copula ("is.").
    value = re.sub(r"([.!?])\s+(Is|Are|Was|Were)\.(?=\s|$)",
                   lambda m: " " + m[2].lower() + ".", value)
    # Never put a learned boundary in the middle of a single-letter expression.
    value = re.sub(r"(?<!\w)([A-HJ-Za-hj-z])\.\s+(?=\w)", r"\1 ", value)
    value = re.sub(r"\b(parameters|components|channels|measurements|values)\.\s+"
                   r"(?=[A-Za-z]+\d+(?:[\s,]|$))", r"\1: ", value, flags=re.I)
    value = re.sub(r",\s+(?=(?:our|my|your|their)\s+\w+\s+(?:is|are|was|were)\b)",
                   ". ", value, flags=re.I)
    # A full uncertainty clause at the end of a long list is easier to read as
    # another sentence. Keep every spoken word, including the leading 'and'.
    value = re.sub(r"(?<=\w)\s+(?=and I (?:am not sure|don't know|do not know)\b)",
                   ". ", value)
    value = re.sub(r"\bi\b", "I", value) if not re.search(r"=|\bequals\b", value) else value
    pieces = list(split_sentences(value))
    for i, piece in enumerate(pieces):
        if piece:
            # Preserve lowercase single-letter symbols, otherwise sentence case.
            first = _TOKEN.search(piece)
            if first and not (len(first.group()) == 1 and
                              (first.group() not in {"i", "a"} or
                               (first.group() == "i" and re.match(
                                   r"\s*(?:=|\bequals\b)", piece[first.end():])))):
                piece = piece[:first.start()] + piece[first.start()].upper() + piece[first.start() + 1:]
            if final and piece.endswith(".") and _is_explicit_question(piece):
                piece = piece[:-1] + "?"
            pieces[i] = piece
    value = " ".join(pieces).strip()
    if not value:
        return ""
    words = _tokens(value)
    if not final:
        # A live hypothesis stays visibly open and can be revised by later audio.
        return value.rstrip(".!?,;:")
    if not re.search(r"[.!?…][\"'’\])]*$", value) and words and not caption_tail_incomplete(value):
        last_sentence = split_sentences(value)[-1]
        value += "?" if _is_explicit_question(last_sentence) else "."
    return value


def _fallback_punctuation(value: str) -> str:
    """Conservative no-model path; never claims semantic sentence certainty."""
    # Clause starts that have an explicit subject in a long run-on. This adds
    # punctuation only, not words, and leaves scientific propositions untouched.
    value = re.sub(r"(?i)(\b(?:person|probabilities|objects|result|results|method|model))\s+"
                   r"(?=(?:so (?:if|we|what)|in particular)\b)", r"\1. ", value)
    value = re.sub(r"(?i)\b(if\s+[^.!?]{8,100}?)\s+then\b", r"\1, then", value)
    value = re.sub(r"(?i)\s+(?=but\b)", ", ", value)
    return value


class TranscriptRestorer:
    """Thread-safe CPU punctuation; loading is local and optional.

    ``model_dir`` may be the model folder itself, ``models/punctuation``, or the
    application's shared ``models`` root. Call ``load`` during capture warmup.
    Existing source text is held in ``raw`` without changes.
    """
    def __init__(self, model_dir: str | Path | None = None,
                 punctuator: Callable[[str], str] | None = None):
        root = Path(model_dir) if model_dir is not None else None
        self.model_dir = root
        self._punctuator = punctuator
        self._attempted = punctuator is not None
        self._load_warning = ""
        self._lock = threading.RLock()

    def _folder(self) -> Path | None:
        if self.model_dir is None:
            return None
        for candidate in (self.model_dir, self.model_dir / MODEL_DIRECTORY,
                          self.model_dir / "punctuation" / MODEL_DIRECTORY):
            if (candidate / "model.int8.onnx").is_file() and (candidate / "bpe.vocab").is_file():
                return candidate
        return None

    @property
    def ready(self) -> bool:
        return self._punctuator is not None or self._folder() is not None

    def load(self) -> bool:
        with self._lock:
            if self._attempted:
                return self._punctuator is not None
            self._attempted = True
            folder = self._folder()
            if folder is None:
                self._load_warning = "punctuation_model_unavailable"
                return False
            try:
                import sherpa_onnx
                model = sherpa_onnx.OnlinePunctuation(
                    sherpa_onnx.OnlinePunctuationConfig(
                        model_config=sherpa_onnx.OnlinePunctuationModelConfig(
                            cnn_bilstm=str(folder / "model.int8.onnx"),
                            bpe_vocab=str(folder / "bpe.vocab"),
                            num_threads=2, provider="cpu")))
                self._punctuator = model.add_punctuation_with_case
                return True
            except Exception:
                # No raw speech in diagnostics, and no cloud fallback.
                self._load_warning = "punctuation_model_failed"
                return False

    def prepare(self) -> bool:
        """Preload on the capture-initialization worker, before live UI updates."""
        return self.load()

    def restore(self, raw: str, final: bool = True, word_times=None) -> RestoredUtterance:
        # word_times intentionally does not force a sentence at every pause.
        start = time.perf_counter()
        raw = str(raw or "")
        clean = clean_disfluencies(raw)
        changes: list[str] = []
        warnings: list[str] = []
        if clean != re.sub(r"\s+", " ", raw).strip():
            changes.append("disfluencies")
        grammatical = correct_obvious_grammar(clean)
        if grammatical != clean:
            changes.append("grammar_agreement")
            clean = grammatical
        repaired = repair_fragment_boundaries(clean)
        if repaired != clean:
            changes.append("fragment_boundaries")
            clean = repaired
        engine = "rules"
        if clean:
            with self._lock:
                self.load()
                if self._punctuator is not None:
                    try:
                        candidate = self._punctuator(clean)
                        projected = _punctuation_projection(clean, candidate)
                        if projected is None:
                            warnings.append("punctuation_lexical_edit_rejected")
                        else:
                            clean = _keep_existing_sentence_boundaries(clean, projected)
                            engine = "edge-punct-int8-cpu"
                    except Exception:
                        warnings.append("punctuation_inference_failed")
                elif self._load_warning:
                    warnings.append(self._load_warning)
            if engine == "rules":
                clean = _fallback_punctuation(clean)
            clean = _normalize_punctuation(clean, final)
            # A 2--3 second ASR commit may end mid-question. Do not turn a
            # model-added mark after "what would" or "in your" into a finished
            # assertion/question. Explicit source punctuation remains preserved.
            original_words = _tokens(raw)
            if (final and original_words and caption_tail_incomplete(raw)
                    and not re.search(r"[.!?][\"'’\])]*$", raw.rstrip())):
                clean = clean.rstrip(".!?,;:")
            if clean != raw:
                changes.append("punctuation_and_case")
        return RestoredUtterance(raw, clean, split_sentences(clean), tuple(changes),
                                 tuple(warnings), engine, time.perf_counter() - start)


def restore_utterance(raw: str, word_times=None, final: bool = True) -> RestoredUtterance:
    """Dependency-free fallback, useful in tests and before model installation."""
    return TranscriptRestorer().restore(raw, final=final, word_times=word_times)


def install_punctuation_model(models_root: str | Path, on_status=None) -> Path:
    """Explicit installer for the Apache-2 licensed 7.5 MB int8 CPU model.

    Only the two selected files are read from the hash-verified upstream archive.
    No tar paths are extracted. An existing completed installation is reused.
    """
    status = on_status or (lambda _value: None)
    destination = Path(models_root) / "punctuation" / MODEL_DIRECTORY
    if (destination / "model.int8.onnx").is_file() and (destination / "bpe.vocab").is_file():
        status("本地英文断句模型已安装。")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    status("下载本地英文断句模型（下载包约31MB，CPU运行，无API费用）…")
    with tempfile.TemporaryDirectory(prefix="punct-install-", dir=destination.parent) as temporary:
        temporary = Path(temporary)
        archive = temporary / "model.tar.bz2"
        digest = hashlib.sha256()
        with urlopen(MODEL_URL, timeout=60) as response, archive.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != MODEL_ARCHIVE_SHA256:
            raise RuntimeError("英文断句模型校验失败，未安装。")
        staging = temporary / MODEL_DIRECTORY
        staging.mkdir()
        with tarfile.open(archive, "r:bz2") as package:
            for filename in ("model.int8.onnx", "bpe.vocab"):
                names = [m for m in package.getmembers()
                         if m.isfile() and Path(m.name).name == filename]
                if len(names) != 1:
                    raise RuntimeError("英文断句模型文件不完整，未安装。")
                source = package.extractfile(names[0])
                if source is None:
                    raise RuntimeError("英文断句模型文件不可读。")
                with source, (staging / filename).open("wb") as target:
                    shutil.copyfileobj(source, target)
        license_url = "https://raw.githubusercontent.com/frankyoujian/Edge-Punct-Casing/main/LICENSE"
        with urlopen(license_url, timeout=30) as response:
            license_text = response.read()
        if b"Apache License" not in license_text or b"Version 2.0" not in license_text:
            raise RuntimeError("英文断句模型许可信息校验失败，未安装。")
        (staging / "LICENSE").write_bytes(license_text)
        (staging / "model.json").write_text(json.dumps({
            "name": "Edge-Punct-Casing int8 English", "provider": "cpu",
            "source": MODEL_URL, "archive_sha256": MODEL_ARCHIVE_SHA256,
            "upstream": "https://github.com/frankyoujian/Edge-Punct-Casing",
            "license": "Apache-2.0", "lexical_generation": False,
        }, indent=2), encoding="utf-8")
        destination.mkdir(exist_ok=True)
        for item in staging.iterdir():
            shutil.copy2(item, destination / item.name)
    status("本地英文断句模型已就绪（CPU、int8）。")
    return destination

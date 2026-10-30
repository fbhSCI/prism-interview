"""Citation provenance is the final request, never every available local source."""
import json

import pytest

from zoom_copilot.answers import (
    AnswerService, JsonFieldStreamer, _sent_evidence, build_prompt, parse_reply,
    estimate_prompt_tokens, LOCAL_CONTEXT_MARGIN, LOCAL_MAX_OUTPUT_TOKENS,
)
from zoom_copilot.candidate_profile import PROFILE_SOURCE, candidate_context, candidate_prompt_context, candidate_fact_id
from zoom_copilot.config import Preferences
from zoom_copilot.knowledge import Passage


def reply(citations=(), **kwargs):
    return json.dumps(dict(answer_en="The gain was measured against baseline A in five scenes.",
                           answer_zh="该提升是在五个场景中相对于基线 A 测得的。",
                           question_zh="你的博士结果是什么？", citations=list(citations), **kwargs))


def test_profile_ids_are_content_based_not_selection_positions_or_titles():
    a = "Personal contribution — study: underwater recovery. Source: cv.pdf p1."
    b = "Personal contribution — study: airflow calibration. Source: cv.pdf p2."
    first = candidate_context("What are your main PhD contributions?", a + "\n" + b)
    reordered = candidate_context("What are your main PhD contributions?", b + "\n" + a)
    focused = candidate_context("What are your airflow contributions?", b + "\n" + a)
    mapping = dict(zip(first["facts"], first["citation_ids"]))
    assert mapping == dict(zip(reordered["facts"], reordered["citation_ids"]))
    assert len(set(mapping.values())) == 2
    assert focused["citation_ids"] == [mapping[b]]
    assert candidate_fact_id(a + " Not yet published.") != mapping[a]


def test_complete_identical_entries_share_one_citation_without_false_duplicate_sources():
    fact = "Personal contribution: underwater calibration. Source: cv.pdf p1."
    _, user = build_prompt("What are your contributions?", [], Preferences(candidate_profile=fact + "\n" + fact))
    sent = _sent_evidence(user, [])
    result = parse_reply(reply([candidate_fact_id(fact), candidate_fact_id(fact)]), sent)
    assert len(result.evidence) == 1
    assert result.evidence[0]["text"] == fact
    assert result.evidence[0]["source"] == PROFILE_SOURCE
    assert "cv.pdf" not in result.sources[0]  # The hint is not a verified paper source.


def test_same_filenames_in_distinct_directories_remain_distinct():
    passages = [Passage("first/study.pdf", "p1", "Underwater observation A.", "a"),
                Passage("second/study.pdf", "p1", "Underwater observation B.", "b")]
    _, user = build_prompt("Explain underwater observations.", passages, Preferences())
    result = parse_reply(reply(["a", "b"]), _sent_evidence(user, passages))
    assert [p["source"] for p in result.evidence] == [p.source for p in passages]


def test_conflicting_source_ids_are_rejected_not_silently_reassigned():
    passages = [Passage("one.pdf", "p1", "A result.", "same"),
                Passage("two.pdf", "p1", "A different result.", "same")]
    _, user = build_prompt("What is the result?", passages, Preferences())
    assert parse_reply(reply(["same"]), _sent_evidence(user, passages)).citations == []
    assert parse_reply(reply(["same"]), passages).citations == []


def test_only_actual_sent_excerpt_is_shown_and_excluded_sources_cannot_be_cited():
    passages = [Passage("a.pdf", "p1", "Spectral calibration measured in five scenes. " * 100, "sent"),
                Passage("b.pdf", "p2", "Secret additional result.", "not-sent")]
    _, user = build_prompt("Explain spectral calibration.", passages,
                           Preferences(provider="deepseek", top_k=1, brief_answer=True))
    sent = _sent_evidence(user, passages)
    result = parse_reply(reply(["sent", "not-sent", "profile-fabricated"]), sent)
    assert result.citations == ["sent"]
    assert result.evidence[0]["text"] == json.loads(user)["reference_passages_untrusted"][0]["excerpt"]
    assert len(result.evidence[0]["text"]) < len(passages[0].text)


def test_cloud_excerpt_optout_preserves_explicit_personal_context_not_paper_text():
    fact = "Personal contribution: underwater calibration. Source: cv.pdf p1."
    passage = Passage("private.pdf", "p1", "Private reference excerpt.", "private-ref")
    prefs = Preferences(provider="deepseek", cloud_consent=True, share_evidence=False, candidate_profile=fact)
    _, user = build_prompt("What are your main contributions?", [passage], prefs)
    assert json.loads(user)["reference_passages_untrusted"] == []
    assert "Private reference excerpt" not in user and "private.pdf" not in user
    sent = _sent_evidence(user, [passage])
    assert [p.passage_id for p in sent] == [candidate_fact_id(fact)]
    assert not parse_reply(reply(), sent).citations  # Available context is not an automatic citation.


def test_no_global_cloud_consent_never_sends_personal_facts(monkeypatch):
    from zoom_copilot.providers import OpenAICompatibleClient
    def fail(*args, **kwargs):
        pytest.fail("No cloud call is allowed without consent")
    monkeypatch.setattr(OpenAICompatibleClient, "complete", fail)
    service = AnswerService(Preferences(provider="deepseek", cloud_consent=False,
        candidate_profile="I developed underwater calibration.", prefer_prepared_answers=False))
    with pytest.raises(ValueError, match="云端授权"):
        service.answer("What did you develop?", [])


def test_profile_budget_omission_removes_id_as_well_as_fact():
    short = "Personal contribution: underwater calibration. Source: cv.pdf p1."
    long = "Personal contribution: underwater recovery. " + "Long qualification. " * 160
    prefs = Preferences(provider="deepseek", candidate_profile=long + "\n" + short)
    _, user = build_prompt("What are your main contributions?", [], prefs)
    data = json.loads(user)["candidate_confirmed_facts"]
    assert data["omitted_entries"] == 1
    assert [entry['text'] for entry in data['entries']] == [short]
    result = parse_reply(reply([candidate_fact_id(long), candidate_fact_id(short)]), _sent_evidence(user, []))
    assert result.citations == [candidate_fact_id(short)]


def test_local_4k_budget_drops_whole_entries_and_citation_ids_together():
    rows = [f"Personal contribution {i}: spectral calibration in airflow. " +
            "Under fixed no-flow calibration only. " * 6 + f"Source: cv.pdf p{i}." for i in range(4)]
    prefs = Preferences(provider="ollama", num_ctx=4096, candidate_profile="\n".join(rows))
    passage = Passage("study.pdf", "p4", "Stokes calibration maps three linear Stokes values only.", "ref")
    system, user = build_prompt("Explain your Stokes calibration and its limitations.", [passage], prefs)
    data = json.loads(user)["candidate_confirmed_facts"]
    assert estimate_prompt_tokens(system + user) + LOCAL_CONTEXT_MARGIN + LOCAL_MAX_OUTPUT_TOKENS <= 4096
    assert data["omitted_entries"] > 0
    assert all(entry['text'] in rows and entry['id'] == candidate_fact_id(entry['text'])
               for entry in data['entries'])
    sent = _sent_evidence(user, [passage])
    result = parse_reply(reply([candidate_fact_id(row) for row in rows]), sent)
    assert result.citations == [entry['id'] for entry in data['entries']]


def test_service_cites_profile_and_streams_no_opaque_profile_markers(monkeypatch):
    from zoom_copilot.providers import OpenAICompatibleClient
    fact = "Personal contribution: underwater recovery measured against A in five scenes. Source: cv.pdf p1."
    identity = candidate_fact_id(fact)
    events = []
    class Client:
        def __init__(self, preferences):
            self.model = self.last_model = "fake-offline-model"
        def complete(self, system, user, on_delta, cancel_event):
            assert [entry['id'] for entry in json.loads(user)["candidate_confirmed_facts"]["entries"]] == [identity]
            value = json.loads(reply([identity]))
            value["answer_en"] += f" [{identity}]"
            raw = json.dumps(value)
            for char in raw:
                on_delta(char)
            return raw
    monkeypatch.setattr("zoom_copilot.providers.OpenAICompatibleClient", Client)
    prefs = Preferences(provider="deepseek", cloud_consent=True, candidate_profile=fact,
                        prefer_prepared_answers=False)
    result = AnswerService(prefs).answer("What are your main contributions?", [],
                                        on_delta=lambda field, delta: events.append((field, delta)))
    assert result.citations == [identity] and result.evidence[0]["text"] == fact
    streamed = "".join(delta for field, delta in events if field == "answer_en")
    assert identity not in streamed and "[profile-" not in streamed
    assert streamed == result.answer_en


def test_scientific_conditions_take_priority_over_brief_length_target():
    system, _ = build_prompt("Describe your main contributions and the limits of each.", [],
                             Preferences(brief_answer=True, max_answer_words=60))
    assert "maximum 60 words" not in system
    assert "Preserve comparators, statistics, test conditions and limitations" in system
    assert "when a short preference would otherwise omit them" in system
    assert "Chinese should convey the full same content" in system
    assert "comparator, evaluation conditions and negative scope" in system
    assert "omit nonessential numbers" in system


@pytest.mark.parametrize("question", [
    "In a quick overview, what did you contribute to doctoral research?",
    "Which architecture has fewer weights, yours or the reference system?",
    "Did the later stress tests validate every channel in the calibration range?",
    "Give the title of your published semiconductor fabrication result.",
])
def test_grounding_rules_apply_to_independent_question_variants(question):
    system, _ = build_prompt(question, [], Preferences(provider="deepseek"))
    assert "give methods and their scope" in system
    assert "Include only details needed to answer the current question" in system
    assert "candidate solution's size/cost must not be assigned to its comparison baseline" in system
    assert "never transfer the proposed method's components/constraints to a baseline" in system
    assert "Evidence from one setting does not prove performance in other settings" in system
    assert "You see only the supplied entries/excerpts, not the full library" in system
    assert "the supplied material does not establish it" in system
    assert "unless an explicit supplied statement supports that absence" in system


@pytest.mark.parametrize("question", [
    "I reconstructed four analyzer intensities. Which Stokes matrix multiplies them?",
    "Does demodulation use the instrument operator or its pseudoinverse?",
])
def test_forward_and_reverse_dimension_rules_survive_alternate_phrasing(question):
    system, _ = build_prompt(question, [], Preferences())
    assert "M is n×k" not in system and "D=M^+" not in system
    assert "References support only their excerpted claims" in system


def test_prompt_entries_bind_complete_text_to_ids_without_duplicate_parallel_arrays():
    rows = ["PhD topic: optical imaging.",
            "Personal contribution: optical calibration under fixed conditions.",
            "Personal contribution: inverse reconstruction validated on held-out scenes."]
    context = candidate_context("What are your main PhD contributions?", '\n'.join(rows))
    original = json.loads(json.dumps(context))
    packed = candidate_prompt_context(context)
    assert context == original  # The local retrieval interface remains unchanged.
    assert 'facts' not in packed and 'citation_ids' not in packed
    assert [entry['text'] for entry in packed['entries']] == context['facts']
    assert all(entry['id'] == candidate_fact_id(entry['text']) for entry in packed['entries'])
    reordered = candidate_prompt_context(candidate_context(
        "What are your main PhD contributions?", '\n'.join(reversed(rows))))
    assert {entry['id']: entry['text'] for entry in packed['entries']} == {
        entry['id']: entry['text'] for entry in reordered['entries']}


def test_future_only_selection_is_not_a_completed_personal_result_assertion():
    rows = ["Future proposal, not completed work: estimate material roughness from optical measurements.",
            "Transferable strengths: optical calibration; material inference is a new area."]
    _, user = build_prompt("Which material roughness results have you achieved?", [],
        Preferences(provider='deepseek', candidate_profile='\n'.join(rows)))
    personal = json.loads(user)['candidate_confirmed_facts']
    assert {entry['kind'] for entry in personal['entries']} == {'future', 'transfer'}
    assert 'personal_contribution_missing' not in personal  # Neither proof of achievement nor absence.
    assert [entry['text'] for entry in personal['entries']] == rows
    result = parse_reply(reply([entry['id'] for entry in personal['entries']]), _sent_evidence(user, []))
    assert len(result.evidence) == 2
    assert [entry['text'] for entry in result.evidence] == rows  # No rewritten negative facts.


def test_model_can_cite_nonadjacent_contributions_without_assuming_id_order(monkeypatch):
    rows = ["PhD topic: optical imaging.",
            "Personal contribution: underwater reconstruction.",
            "Personal contribution: calibrated neural reconstruction.",
            "Personal contribution: airflow reliability under fixed calibration."]
    chosen_texts = [rows[1], rows[3]]
    class Client:
        def __init__(self, preferences):
            self.model = self.last_model = 'offline-structural-test'
        def complete(self, system, user, on_delta, cancel_event):
            entries = json.loads(user)['candidate_confirmed_facts']['entries']
            reported = [entry['id'] for entry in entries if entry['text'] in chosen_texts]
            return reply(reported)
    monkeypatch.setattr('zoom_copilot.providers.OpenAICompatibleClient', Client)
    result = AnswerService(Preferences(provider='deepseek', cloud_consent=True,
        candidate_profile='\n'.join(rows), prefer_prepared_answers=False)).answer(
            'What are your main PhD contributions?', [])
    assert [entry['text'] for entry in result.evidence] == chosen_texts
    assert result.citations == [candidate_fact_id(text) for text in chosen_texts]

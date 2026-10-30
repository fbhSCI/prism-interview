"""Bounded terminology protection and mandatory whole-sentence fallback."""
from types import SimpleNamespace
import threading

import pytest

from zoom_copilot.translation import ArgosTranslator, TranslationWorker, protect_research_terms, restore_research_terms, _correct_known_terms


def test_complete_phrases_only_longest_first_and_hyphen_variants():
    protected, terms = protect_research_terms(
        "HYPERSPECTRAL POLARIMETRIC REFLECTANCE differs from scattering-induced background interference.")
    assert len(terms) == 2
    assert terms[0].chinese == "高光谱偏振反射"
    assert terms[1].chinese == "散射引起的背景干扰"
    assert "[TERM100]" in protected and "[TERM101]" in protected
    original = "The multisensing operatorship is unrelated."
    assert protect_research_terms(original) == (original, [])


def test_repeated_terms_get_distinct_recoverable_markers():
    protected, terms = protect_research_terms("Compare cash flow with cash flow.")
    assert len(terms) == 2 and terms[0].marker != terms[1].marker
    assert restore_research_terms(protected, terms) == "Compare 现金流 with 现金流."


def test_interview_idiom_is_protected_as_a_complete_phrase():
    protected, terms = protect_research_terms("Are you a people person?")
    assert len(terms) == 1 and terms[0].chinese == "善于与人打交道的人"
    assert restore_research_terms("你是[TERM100]吗？", terms) == "你是善于与人打交道的人吗？"
    assert protect_research_terms("Many people personalize this.")[1] == []


@pytest.mark.parametrize("output", [
    "只有[TERM100]",  # Dropped second term.
    "[TERM100]和[TERM100]及[TERM101]",  # Repeated term.
    "[TERM100]与[TERM 101]",  # Mutated term; fuzzy recovery is forbidden.
    "[TERM100]与[TERM101]及[TERM999]",  # Invented term.
])
def test_any_dropped_mutated_repeated_or_unknown_marker_rejects_output(output):
    _, terms = protect_research_terms("cash flow and supply chain")
    assert restore_research_terms(output, terms) is None


def test_collision_and_limit_opt_out_without_mutating_original():
    source = "The literal [TERM100] stands next to cash flow."
    assert protect_research_terms(source) == (source, [])
    source = "cash flow " * 33
    assert protect_research_terms(source) == (source, [])


class IdentityTokenizer:
    def encode(self, text, out_type=str):
        return [text]

    def decode(self, tokens):
        return "".join(tokens)


def fake_translator(tmp_path, decode):
    translator = ArgosTranslator(tmp_path)
    translator._tokenizer = IdentityTokenizer()
    calls = []

    class Engine:
        def translate_batch(self, batches, **kwargs):
            calls.extend(batch[0] for batch in batches)
            return [SimpleNamespace(hypotheses=[[decode(batch[0])]]) for batch in batches]

    translator._translator = Engine()
    return translator, calls


def test_successful_guarded_translation_restores_all_terms_and_keeps_negation(tmp_path):
    translator, calls = fake_translator(tmp_path, lambda _: "[TERM100]并不等同[TERM101]。")
    assert translator.translate("Cash flow is not the same as supply chain.") == "现金流并不等同供应链。"
    assert len(calls) == 1 and not translator.last_warnings
    assert translator.last_metrics["terms_restored"] == 2


def test_failed_guard_retranslates_original_sentence_and_emits_warning(tmp_path):
    source = "Does ExampleSystem use novel view synthesis rather than directly estimate roughness?"

    def decode(text):
        if "[TERM" in text:
            return "ExampleSystem只剩[TERM101]"  # Lost the novel-view term.
        assert text == source
        return "ExampleSystem是否用于新颖的视图合成而不是直接估计粗糙度？"

    translator, calls = fake_translator(tmp_path, decode)
    translated = translator.translate(source)
    assert translated == "ExampleSystem是否用于新视角合成而不是直接估计粗糙度？"
    assert calls[1] == source  # Whole original, never a guessed fragment.
    assert "TERM" not in translated
    assert len(translator.last_warnings) == 1
    assert translator.last_metrics["term_fallback_sentences"] == 1


def test_fallback_is_per_coherent_unit_and_does_not_retranslate_stable_units(tmp_path):
    def decode(text):
        if text.startswith("First"):
            return "首先[TERM100]。"
        if "[TERM" in text:
            return "标记丢失。"
        return "第二句供应链。"

    translator, calls = fake_translator(tmp_path, decode)
    prefix = ('First, we explain how cash flow affects the observations during an experiment '
              'and why a separate measurement procedure is needed before any result can '
              'be interpreted reliably.')
    out = translator.translate(prefix + ' Second supply chain.')
    assert out == "首先现金流。 第二句供应链。"
    assert len(calls) == 3 and calls[-1] == "Second supply chain."


def test_postcorrection_is_source_conditioned_not_global_word_replacement():
    target = "校准的感应操作器不同于其他设备。"
    assert _correct_known_terms("The calibrated sensing operator is different.", target) == "校准的感知算子不同于其他设备。"
    assert _correct_known_terms("A human operator is working.", target) == target
    # Only the complete explicit background-interference term may correct this;
    # a generic mention of scattering must not rename genuine speckles elsewhere.
    assert _correct_known_terms("Compare scattering and speckles.", "斑点引起的背景干扰") == "斑点引起的背景干扰"


def test_worker_reports_fallback_warning_after_result():
    completed = threading.Event()
    statuses, results = [], []
    translator = SimpleNamespace(translate=lambda _: "原始句重译", last_warnings=["请核对英语原文。"])

    def status(text):
        statuses.append(text)
        completed.set()

    worker = TranslationWorker(translator, lambda *args: results.append(args), status)
    worker.start()
    worker.submit("entry1", "English source")
    assert completed.wait(3)
    worker.stop()
    worker.thread.join(3)
    assert results == [("entry1", "原始句重译")]
    assert statuses == ["翻译提示：请核对英语原文。"]

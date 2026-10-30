import threading
import time

from zoom_copilot.transcript_refine import (
    GrammarRefinement, LocalGrammarWorker, grammar_fidelity_reason, refine_grammar_local,
)
from zoom_copilot.translation_protection import (
    TranslationProtectionError, normalize_research_translation, protect_translation_source,
)
import pytest


def test_simple_agreement_refinement_is_verifiable():
    assert grammar_fidelity_reason(
        "I'm not sure, but we was using a model and it don't measure S3.",
        "I am not sure, but we were using the model and it doesn't measure S3.") == ""


@pytest.mark.parametrize(("source", "candidate"), [
    ("I use a metric.", "We use a metric."),
    ("I use a metric.", "I used a metric."),
    ("I'm not sure if we measure S3.", "We measure S3."),
    ("We do not measure S3 but we measure S2.", "We measure S3 but we do not measure S2."),
    ("The inverse rendering model may work.", "The forward rendering model will work."),
    ("A has higher reflectance than B.", "B has higher reflectance than A."),
])
def test_content_polarity_modality_tense_and_subject_edits_rejected(source, candidate):
    assert grammar_fidelity_reason(source, candidate)


def test_numbers_formula_symbols_and_variable_case_are_preserved():
    assert grammar_fidelity_reason("The value is -0.95.", "The value is 0.95.") == "number_changed"
    assert grammar_fidelity_reason("s = D i.", "s = D * i.") == "formula_changed"
    assert grammar_fidelity_reason("s equals D i.", "S equals D i.") == "technical_token_changed"


def test_formula_protection_roundtrip_keeps_spoken_contrast_exact():
    raw = "The pseudoinverse is D and s equals D i not M i. The value is 0.95."
    protected = protect_translation_source(raw)
    assert len(protected.slots) == 1
    assert protected.slots[0][1] == "s equals D i not M i"
    assert "not M i" not in protected.text
    output = "伪逆是D，而" + protected.slots[0][0] + "。这个值是0.95。"
    assert protected.restore(output) == "伪逆是D，而s equals D i not M i。这个值是0.95。"


def test_formula_protection_separates_neighboring_equations():
    raw = "S0 = I0 + I90 and S1 = I0 - I90"
    protected = protect_translation_source(raw)
    assert [literal for _, literal in protected.slots] == ["S0 = I0 + I90", "S1 = I0 - I90"]
    assert protected.restore(protected.text) == raw


def test_formula_protection_keeps_comma_and_rather_than():
    for raw in ("s = D i, not M i", "s is equal to D i rather than M i"):
        protected = protect_translation_source(raw)
        assert protected.slots[0][1] == raw
        assert protected.restore(protected.text) == raw


def test_lost_or_duplicate_formula_placeholder_rejects_translation():
    protected = protect_translation_source("s equals D i not M i")
    for output in ("s = Di - Mi", protected.text + " " + protected.text):
        with pytest.raises(TranslationProtectionError):
            protected.restore(output)


def test_normal_sentence_and_prior_placeholder_literal_are_safe():
    raw = "ZXQFORMULA123QXZ is text. The value equals a probability."
    protected = protect_translation_source(raw)
    assert protected.text == raw
    assert protected.slots == ()


def test_matrix_inverse_and_optical_polarization_normalization_is_narrow():
    source = "It is not an inverse of the instrument matrix and we measure polarization."
    assert normalize_research_translation(source, "它不是仪器矩阵的倒数，我们测量极化。") == "它不是仪器矩阵的逆，我们测量偏振。"
    assert normalize_research_translation("Compare the reciprocal and inverse of the matrix.",
                                          "比较矩阵的倒数和逆。") == "比较矩阵的倒数和逆。"
    assert normalize_research_translation("The inverse of time.", "时间的倒数。") == "时间的倒数。"


def _fake_result(raw, baseline, **kwargs):
    return GrammarRefinement(kwargs["turn_id"], kwargs["revision"], raw, baseline, baseline,
                             True, "test", 0.001, kwargs["model"])


def test_worker_default_off_does_not_call_local_model():
    called = []
    worker = LocalGrammarWorker(called.append, requester=lambda *_a, **_k: pytest.fail("called"))
    assert not worker.submit("turn1", 1, "raw", "clean")
    worker.close()
    assert called == []


def test_remote_or_unknown_ollama_model_is_not_called():
    result = refine_grammar_local("raw", "baseline", model="remote-model:cloud")
    assert not result.accepted
    assert result.reason == "unsupported_local_grammar_model"


def test_worker_coalesces_latest_revision_before_model_call():
    done = threading.Event()
    requests = []
    outputs = []
    def request(raw, baseline, **kwargs):
        requests.append(kwargs["revision"])
        return _fake_result(raw, baseline, **kwargs)
    def output(result):
        outputs.append(result)
        done.set()
    worker = LocalGrammarWorker(output, enabled=True, settle_seconds=.03, requester=request)
    try:
        assert worker.submit("t", 1, "first", "first")
        assert worker.submit("t", 2, "full", "full")
        assert done.wait(1)
        assert requests == [2]
        assert [o.clean for o in outputs] == ["full"]
    finally:
        worker.close()


def test_inflight_old_result_is_discarded_and_submit_does_not_block():
    began, release, done = threading.Event(), threading.Event(), threading.Event()
    outputs = []
    def request(raw, baseline, **kwargs):
        if kwargs["revision"] == 1:
            began.set()
            assert release.wait(1)
        return _fake_result(raw, baseline, **kwargs)
    def output(result):
        outputs.append(result)
        done.set()
    worker = LocalGrammarWorker(output, enabled=True, settle_seconds=0, requester=request)
    try:
        worker.submit("t", 1, "first", "first")
        assert began.wait(1)
        started = time.perf_counter()
        assert worker.submit("t", 2, "full", "full")
        assert time.perf_counter() - started < .1
        release.set()
        assert done.wait(1)
        assert [o.revision for o in outputs] == [2]
    finally:
        release.set()
        worker.close()


def test_close_is_nonblocking_and_inflight_result_is_invalidated():
    began, release, finished = threading.Event(), threading.Event(), threading.Event()
    outputs = []
    def request(raw, baseline, **kwargs):
        began.set()
        release.wait(1)
        finished.set()
        return _fake_result(raw, baseline, **kwargs)
    worker = LocalGrammarWorker(outputs.append, enabled=True, settle_seconds=0, requester=request)
    worker.submit("t", 1, "raw", "clean")
    assert began.wait(1)
    started = time.perf_counter()
    worker.close()
    assert time.perf_counter() - started < .1
    release.set()
    assert finished.wait(1)
    worker._thread.join(.3)
    assert outputs == []


def test_worker_failure_returns_original_display_without_losing_raw():
    done = threading.Event()
    results = []
    def broken(*_args, **_kwargs):
        raise RuntimeError("failed")
    def output(result):
        results.append(result)
        done.set()
    worker = LocalGrammarWorker(output, enabled=True, settle_seconds=0, requester=broken)
    try:
        worker.submit("turn", 1, "raw speech", "baseline.")
        assert done.wait(1)
        assert results[0].clean == "baseline."
        assert results[0].raw == "raw speech"
        assert not results[0].accepted
    finally:
        worker.close()

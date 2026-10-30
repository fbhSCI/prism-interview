from zoom_copilot.transcript_text import (
    TranscriptRestorer, clean_disfluencies, correct_obvious_grammar, restore_utterance, split_sentences,
)


def test_raw_is_exact_and_obvious_stutters_are_cleaned():
    raw = "  I I think I think we we can use the the sensing operator, um but not S3.  "
    result = restore_utterance(raw)
    assert result.raw == raw
    assert "I think we can use the sensing operator" in result.clean
    assert "not S3" in result.clean
    assert "um" not in result.clean


def test_filler_removal_does_not_delete_words_with_filler_substrings():
    value = clean_disfluencies("uh we assume a sum of signals in the aluminum medium um")
    assert value == "we assume a sum of signals in the aluminum medium"


def test_micrometre_unit_and_uppercase_scientific_acronym_are_not_fillers():
    value = clean_disfluencies("Um we use a 5 um structure and the ERM method uh")
    assert value == "we use a 5 um structure and the ERM method"


def test_repeated_negation_numbers_and_real_emphasis_are_not_deduced_as_stutter():
    value = clean_disfluencies("not not 95% 95% very very strong and had had the measurement")
    assert value == "not not 95% 95% very very strong and had had the measurement"


def test_uncertainty_remains_and_no_scientific_fact_is_corrected():
    raw = "I'm not sure but I think perhaps S3 is the intensity and the value is not 0.95"
    result = restore_utterance(raw)
    for phrase in ("I'm not sure", "I think", "perhaps", "S3 is the intensity", "not 0.95"):
        assert phrase in result.clean


def test_actual_you_know_questions_are_not_removed():
    raw = "Do you know the result? If you know the answer, tell me what you know."
    assert clean_disfluencies(raw) == raw


def test_only_filler_you_know_context_is_removed():
    raw = "and you know so this model is you know predicting a person"
    assert clean_disfluencies(raw) == "so this model is predicting a person"


def test_abandoned_exact_repeated_question_start_is_removed():
    raw = "what's the probability how yeah i mean what's the probability of him being a person"
    assert clean_disfluencies(raw) == "what's the probability of him being a person"


def test_i_mean_correction_is_retained():
    raw = "I mean S2, not S3, and I am not sure."
    assert clean_disfluencies(raw) == raw


def test_obvious_grammar_agreement_preserves_pronoun_tense_and_negation():
    raw = "I'm not sure, but we was measuring S0 and the model don't measure S3."
    result = restore_utterance(raw)
    assert "we were measuring S0" in result.clean
    assert "the model doesn't measure S3" in result.clean
    assert "I'm not sure" in result.clean
    assert "grammar_agreement" in result.changes


def test_subjunctive_and_multiplicity_are_not_fact_corrected():
    raw = "If I were sure, I would say this is a parameters problem, but I am unsure."
    assert correct_obvious_grammar(raw) == raw


def test_cut_off_syllable_is_removed_but_research_hyphen_kept():
    result = restore_utterance("we use po- polarization and wavelength-dependent attenuation")
    assert "po- " not in result.clean
    assert "polarization" in result.clean
    assert "wavelength-dependent" in result.clean


def test_sentence_split_does_not_split_decimal_abbreviation_or_initial():
    text = "Dr. Smith measured 0.95, e.g. at 532 nm. Is that sufficient? We are unsure."
    assert split_sentences(text) == (
        "Dr. Smith measured 0.95, e.g. at 532 nm.", "Is that sufficient?", "We are unsure.")


def test_learned_lexical_changes_are_rejected():
    model = lambda _: "We measure S3 with 100% accuracy."
    result = TranscriptRestorer(punctuator=model).restore("we do not measure S3 with 95% accuracy")
    assert "not measure S3" in result.clean
    assert "95%" in result.clean
    assert "100%" not in result.clean
    assert result.engine == "rules"
    assert "punctuation_lexical_edit_rejected" in result.warnings


def test_technical_token_case_is_restored():
    model = lambda _: "The GPU uses S0, S1 and S2 at 532 nm."
    result = TranscriptRestorer(punctuator=model).restore("the GPU uses S0 S1 and S2 at 532 nm")
    assert result.clean == "The GPU uses S0, S1 and S2 at 532 nm."
    assert result.engine == "edge-punct-int8-cpu"


def test_formula_letters_and_decimal_value_survive():
    model = lambda _: "The pseudoinverse is D and S equals D. I not M I. The value is 0.95."
    raw = "the pseudoinverse is D and s equals D i not M i the value is 0.95"
    result = TranscriptRestorer(punctuator=model).restore(raw)
    assert "s equals D i not M i." in result.clean
    assert "0.95" in result.clean
    assert "D. I" not in result.clean


def test_removed_minus_equality_or_number_separator_is_rejected():
    for raw, output in [("the value is -0.95", "The value is 0.95."),
                        ("i=M s", "I M s."),
                        ("the sample has 1,000 pixels", "The sample has 1 000 pixels.")]:
        result = TranscriptRestorer(punctuator=lambda _v: output).restore(raw)
        assert "punctuation_lexical_edit_rejected" in result.warnings
        assert result.engine == "rules"


def test_orphan_final_copula_is_joined():
    model = lambda _: "We can see how strong the model's prediction. Is."
    result = TranscriptRestorer(punctuator=model).restore("we can see how strong the model's prediction is")
    assert result.clean == "We can see how strong the model's prediction is."


def test_list_intro_keeps_stokes_symbols_with_sentence():
    model = lambda _: "We measure the parameters. S0 S1 and S2."
    result = TranscriptRestorer(punctuator=model).restore("we measure the parameters S0 S1 and S2")
    assert result.clean == "We measure the parameters: S0 S1 and S2."


def test_explicit_textual_question_gets_question_mark():
    result = restore_utterance("how would you distinguish material reflectance from illumination")
    assert result.clean.endswith("?")


def test_conditional_preamble_to_direct_question_gets_question_mark():
    raw = "in particular if there is strong scattering which spectral measurements would you use"
    model = lambda _: "In particular, if there is strong scattering, which spectral measurements would you use."
    result = TranscriptRestorer(punctuator=model).restore(raw)
    assert result.clean.endswith("?")


def test_complete_uncertainty_clause_and_comma_splice_are_readable_sentences():
    raw = "we do not measure S3 our measurements are S0 S1 and S2 and I am not sure"
    model = lambda _: "We do not measure S3, our measurements are S0 S1 and S2 and I am not sure."
    result = TranscriptRestorer(punctuator=model).restore(raw)
    assert result.sentences == ("We do not measure S3.", "Our measurements are S0 S1 and S2.",
                                "And I am not sure.")


def test_no_question_mark_is_invented_for_uncertain_statement():
    result = restore_utterance("I am not sure if this extends to circular polarization")
    assert result.clean.endswith(".")
    assert "?" not in result.clean


def test_provisional_remains_open_and_final_incomplete_is_not_fabricated():
    assert restore_utterance("how would you measure", final=False).clean == "How would you measure"
    assert restore_utterance("we need to compare with").clean == "We need to compare with"


def test_newlines_are_sentences_and_no_pause_boundary_is_forced():
    model = lambda _: "How would you distinguish reflectance from illumination? In particular, which channels would you use?"
    raw = "how would you distinguish reflectance from illumination in particular which channels would you use"
    result = TranscriptRestorer(punctuator=model).restore(
        raw, word_times=[{"text": "reflectance", "start": 0, "end": 0.5},
                         {"text": "from", "start": 5, "end": 5.3}])
    assert len(result.sentences) == 2
    assert "reflectance from illumination" in result.clean
    assert result.display.count("\n") == 1


def test_missing_model_prepares_without_network_and_returns_rules(tmp_path):
    restorer = TranscriptRestorer(tmp_path)
    assert not restorer.ready
    assert not restorer.prepare()
    result = restorer.restore("we do not know the answer")
    assert result.engine == "rules"
    assert "punctuation_model_unavailable" in result.warnings
    assert result.clean == "We do not know the answer."


def test_model_failure_falls_back_without_losing_speech():
    def broken(_value):
        raise RuntimeError("failure")
    result = TranscriptRestorer(punctuator=broken).restore("we do not measure S3")
    assert result.clean == "We do not measure S3."
    assert "punctuation_inference_failed" in result.warnings


def test_empty_speech_stays_empty():
    result = restore_utterance("")
    assert result.clean == result.display == result.raw == ""
    assert result.sentences == ()


def test_existing_asr_marks_do_not_combine_with_predicted_marks():
    raw = "We tell that, Okay. This is the Paris Queen. What is in your case?"
    model = lambda _: "We tell that,. Okay,. This is the Paris Queen,. What is in your case?."
    result = TranscriptRestorer(punctuator=model).restore(raw)
    assert result.clean == "We tell that. Okay. This is the Paris Queen. What is in your case?"
    assert ",." not in result.clean
    assert "??" not in result.clean
    assert result.raw == raw


def test_repeated_and_mixed_terminal_marks_are_idempotent():
    raw = "What is the probability??. We are unsure,,. The value is 0.95... Is it 95%?!"
    once = restore_utterance(raw)
    twice = restore_utterance(once.clean)
    assert once.clean == twice.clean
    assert once.clean == "What is the probability? We are unsure. The value is 0.95. Is it 95%?"
    assert "0.95" in once.clean
    assert "95%" in once.clean


def test_punctuation_cluster_normalization_does_not_reinterpret_numeric_literals():
    result = restore_utterance("The value is 1,.000 and the precision is 0.95.")
    assert "1,.000" in result.clean
    assert "0.95" in result.clean

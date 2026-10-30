"""A speech crop is not a sentence or a separate research question."""
from zoom_copilot.interview_context import analyze_question, question_parts
from zoom_copilot.question_packet import QuestionContext
from zoom_copilot.transcript_text import TranscriptRestorer, repair_fragment_boundaries, restore_utterance


GAME = (
    "SO, a lot of AA models get developed every day, and you're getting this news where "
    "Each model suddenly, which used to work really well now doesn't work. "
    "So let. US play small game. So you and me, we are gonna say one model. "
    "Each, if I say one model, you say some other model. Okay. "
    "The person who loses this, that means who doesn't say a model name at. "
    "The end of the day loses this. Okay, so do you want to start or do. "
    "You want me to start?"
)


def test_game_rules_are_background_and_turn_choice_is_one_actual_question():
    restored = restore_utterance(GAME)
    focus = analyze_question(restored.clean)
    assert restored.raw == GAME
    for phrase in ('let us play small game', 'one model each.', 'at the end of the day',
                   'do you want to start, or do you want me to start?'):
        assert phrase in restored.clean
    assert 'AA models' in restored.clean and 'AI models' not in restored.clean
    assert focus.interaction_kind == 'coordination'
    assert focus.answer_question_en == 'Okay, so do you want to start, or do you want me to start?'
    assert 'one model each.' in focus.background_en
    assert len(question_parts(restored.clean)) == 1
    assert 'let.' not in restored.clean.lower() and 'at.' not in restored.clean.lower()


def test_punctuation_model_cannot_resplit_existing_complete_game_sentences():
    # Actual failure class seen with Edge-Punct on already punctuated speech.
    source = 'If I say one model, you say some other model. Do you want to start?'
    model_output = 'If I, say one model. You say some other model. Do you want to start?'
    restored = TranscriptRestorer(punctuator=lambda _: model_output).restore(source)
    assert restored.clean == source
    assert restored.sentences == (
        'If I say one model, you say some other model.', 'Do you want to start?')


def test_unpunctuated_long_explanation_still_receives_sentence_restoration():
    source = 'we used polarization measurements we did not measure S3'
    output = 'We used polarization measurements. We did not measure S3.'
    restored = TranscriptRestorer(punctuator=lambda _: output).restore(source)
    assert restored.clean == output
    assert restored.engine == 'edge-punct-int8-cpu'


def test_packet_preserves_full_raw_setup_and_structured_focus_for_model():
    context = QuestionContext()
    context.record('whole', GAME, speaker_id=2, speaker_label='说话人 2', accepted=True)
    packet = context.build(GAME)
    assert packet.raw_question_en == GAME
    assert packet.turns[0].raw_en == GAME
    assert 'one model each.' in packet.question_en
    assert packet.interaction_kind == 'coordination'
    assert len(packet.subquestions) == 1
    for data in (packet.to_ui_data(), packet.to_prompt_data()):
        assert data['interaction_kind'] == 'coordination'
        assert 'do you want me to start?' in data['answer_question_en']
        assert 'AA models' in data['background_en']
    assert '说话人 2' in packet.display_text


def test_coordination_detector_does_not_route_technical_start_questions_locally():
    for question in (
        'Where do you start in inverse rendering?',
        'How would you start calibrating the instrument matrix?',
        'Do you want to start with spectral acquisition or material estimation?',
        'Can I start by estimating the Stokes vector?',
        'Do you want to start, or do you want me to start? How would you calibrate S0?',
    ):
        assert analyze_question(question).interaction_kind == 'research'


def test_coordination_detector_accepts_both_orders_and_preserves_choice():
    for question in (
        'Would you like to go first, or should I start?',
        'Shall I begin, or would you prefer to go first?',
        'Do you want me to start, or do you want to go first?',
    ):
        result = analyze_question(question)
        assert result.interaction_kind == 'coordination'
        assert result.answer_question_en == question


def test_false_boundaries_never_rewrite_technical_values_terms_or_negation():
    source = ('The US researchers use SO coupling and AA models at. The wavelength of 532 nm. '
              'We do not retrieve S3, and the uncertainty is -0.95, not 95%. '
              'How do you distinguish s=D i from i=M s?')
    repaired = repair_fragment_boundaries(source)
    for phrase in ('US researchers', 'SO coupling', 'AA models', '532 nm', 'not retrieve S3',
                   '-0.95', 'not 95%', 's=D i', 'i=M s'):
        assert phrase in repaired
    result = analyze_question(repaired)
    assert result.interaction_kind == 'research'
    assert result.answer_question_en == 'How do you distinguish s=D i from i=M s?'
    assert 'not 95%' in result.background_en


def test_indirect_question_without_explicit_ask_is_preserved_as_complete_focus():
    source = 'I would be interested in your research limitations and validation plans.'
    result = analyze_question(source)
    assert result.answer_question_en == source
    assert result.background_en == ''
    assert result.interaction_kind == 'research'


def test_multiple_scientific_questions_keep_setup_and_every_ask():
    source = ('The instrument matrix maps the Stokes vector to reconstructed intensities. '
              'How would you calibrate at 0.95 confidence? Which assumptions fail under scattering?')
    result = analyze_question(source)
    assert result.interaction_kind == 'research'
    assert result.background_en == source.split(' How')[0]
    assert result.answer_question_en == (
        'How would you calibrate at 0.95 confidence? Which assumptions fail under scattering?')
    packet = QuestionContext().build(source)
    assert len(packet.subquestions) == 2
    assert packet.question_en == source


def test_learned_discourse_break_does_not_become_a_standalone_question_sentence():
    source = 'So what would be my metric to see if my model is performing well'
    output = 'SO. What would be my metric to see if my model is performing well.'
    result = TranscriptRestorer(punctuator=lambda _: output).restore(source)
    assert result.clean == 'So, what would be my metric to see if my model is performing well?'
    assert len(result.sentences) == 1
    assert analyze_question(result.clean).answer_question_en == result.clean
    assert repair_fragment_boundaries('SO. What is the spin-orbit coupling?') == (
        'SO. What is the spin-orbit coupling?')


def test_hard_cut_after_auxiliary_or_possessive_stays_open_until_continuation():
    for source, output in (
        ('at object detection. So what would', 'At object detection. SO, what would?'),
        ('my model performance was satisfactory in your', 'My model performance was satisfactory in your.'),
    ):
        result = TranscriptRestorer(punctuator=lambda _, value=output: value).restore(source)
        assert not result.clean.endswith(('.', '?', '!'))
    # A speaker may use an elliptical but complete statement with explicit ASR
    # punctuation. Do not delete that original mark merely for ending in would.
    assert restore_utterance('I thought I would.').clean == 'I thought I would.'


def test_for_example_clause_keeps_its_explanation_in_same_sentence():
    source = 'For example. Particular task is object detection.'
    assert repair_fragment_boundaries(source) == 'For example, particular task is object detection.'


def test_compact_model_input_keeps_short_actual_ask_after_metadata_compression():
    import json
    source = ('This optical system measures only linear polarization. '
              'How would you validate the reconstructed Stokes components?')
    packet = QuestionContext().build(source)
    data = packet.to_prompt_data(compact=True, max_chars=1400)
    assert data['answer_question_en'] == 'How would you validate the reconstructed Stokes components?'
    assert len(json.dumps(data,ensure_ascii=False,separators=(',',':'))) <= 1400
    assert packet.question_en == source


def test_game_confirmation_tags_are_background_when_real_turn_choice_arrives():
    source = ('We work with a lot of models, right? So let us play a small game. '
              'We are gonna say one model. Each. If I say one model, you say some other model, okay? '
              'Okay? So do you want to start or do you want me to start?')
    packet = QuestionContext().build(source)
    assert packet.interaction_kind == 'coordination'
    assert packet.answer_question_en == 'So, do you want to start, or do you want me to start?'
    assert 'models, right?' in packet.background_en
    assert 'other model, okay?' in packet.background_en
    assert 'one model each. If I say' in packet.question_en
    assert len(packet.subquestions) == 1
    assert packet.raw_question_en == source


def test_nontechnical_confirmation_only_waits_for_the_actual_ask():
    for source in ('Okay?', 'We work with a lot of models, right?',
                   'If I say one model, you say some other model, okay?'):
        packet = QuestionContext().build(source)
        assert packet.interaction_kind == 'setup'
        assert packet.answer_question_en == ''
        assert packet.background_en == source
        assert packet.subquestions == ()
        assert packet.to_ui_data()['answer_question_en'] == ''
        assert packet.question_en == source


def test_scientific_confirmations_and_explicit_direct_questions_stay_answerable():
    source = 'The Stokes matrix maps i=M s, right?'
    packet = QuestionContext().build(source)
    assert packet.interaction_kind == 'research'
    assert packet.answer_question_en == source
    assert packet.subquestions == (source,)
    source = 'How do you calibrate S0, okay? Which assumptions fail under scattering?'
    packet = QuestionContext().build(source)
    assert packet.interaction_kind == 'research'
    assert len(packet.subquestions) == 2
    assert 'How do you calibrate S0, okay?' in packet.answer_question_en
    assert 'Which assumptions fail under scattering?' in packet.answer_question_en

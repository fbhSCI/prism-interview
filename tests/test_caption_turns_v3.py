from zoom_copilot.turns import CaptionTurn, QuestionAssembler, join_fragments


def test_long_overlap_is_removed_without_rewriting_content():
    first = "How would you separate illumination and material reflectance in a scattering scene using spectral and polarization observations with a calibrated imaging system"
    repeated = "illumination and material reflectance in a scattering scene using spectral and polarization observations with a calibrated imaging system and validate identifiability?"
    result = join_fragments(first, repeated)
    assert result.count("calibrated imaging system") == 1
    assert result.endswith("and validate identifiability?")


def test_caption_partial_revises_same_card_and_final_commits_only_once():
    turn = CaptionTurn()
    identity, text = turn.update("Could you explain")
    assert text == "Could you explain" and not turn.committed
    assert turn.update("Could you explain calibration uncertainty", final=True)[0] == identity
    assert turn.update("calibration uncertainty propagation?")[1] == "Could you explain calibration uncertainty propagation?"
    assert turn.update("calibration uncertainty propagation?", final=True)[1] == "Could you explain calibration uncertainty propagation?"
    turn.seal()
    assert turn.update("Why?", final=True)[0] != identity


def test_identical_final_does_not_create_new_question_version():
    question = QuestionAssembler()
    question.add("What is the calibration matrix?", at=1)
    question.add("What is the calibration matrix?", at=1.5)
    assert question.version == 1

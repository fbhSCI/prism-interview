"""Realistic source-clock endpoint regressions; no API/model/audio calls."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from test_controller_v2 import controller
from zoom_copilot.config import Preferences
from zoom_copilot.turns import QuestionAssembler


def context(c, end, lexical=None, **kwargs):
    return dict(phase_epoch=c._phase_epoch, stream_id='zoom', word_timed=True,
                start_sample=0, end_sample=end,
                text_end_sample=end if lexical is None else lexical, **kwargs)


def qa(c):
    c.change_interview_phase('qa')
    c.listening = True


def test_default_endpoint_does_not_stack_old_two_second_delay():
    q = QuestionAssembler()
    q.voice(10)
    q.add('How would you validate the reconstruction?', at=11.15)
    assert not q.ready(11.30)
    assert q.ready(11.34)
    assert not q.ready(11.34, pending=1)


@pytest.mark.parametrize('tail', ['and?', 'would!', 'not.', 'How would you?', 's ='])
def test_guessed_terminal_punctuation_cannot_complete_an_unfinished_tail(tail):
    q = QuestionAssembler()
    q.add(tail, at=1)
    assert q.incomplete
    assert not q.ready(10)


def test_empty_final_overlap_retires_draft_and_auto_answer_unblocks(controller):
    c = controller
    qa(c)
    c.on_transcript('final', 'How do you calibrate the instrument?', context(c, 16000))
    # CPU draft still carries a lexical tail not yet covered by first commit.
    c.on_transcript('partial', 'the instrument', context(c, 22000, 18000))
    assert c._partial_id is not None
    # Tail flush decodes only overlapping words. This is success, not rejection.
    c.on_transcript('final', '', context(c, 26000, empty_commit=True))
    assert c._partial_id is None
    assert c._last_final_end[('zoom', c._phase_epoch)] == 26000
    c.assembler.last_voice = c.assembler.last_text = 0
    c.auto_answer()
    assert c.busy and not c._question_uncertain


def test_empty_final_preserves_a_genuinely_newer_lexical_tail(controller):
    c = controller
    qa(c)
    c.on_transcript('final', 'How do you calibrate the instrument?', context(c, 16000))
    c.on_transcript('partial', 'and validate the result', context(c, 36000, 34000))
    before = c._live_data['en']
    c.on_transcript('final', '', context(c, 26000, empty_commit=True))
    assert c._partial_id is not None
    assert c._live_data['en'] == before
    c.assembler.last_voice = c.assembler.last_text = 0
    c.auto_answer()
    assert not c.busy


def test_silent_draft_clock_cannot_reopen_an_already_confirmed_question(controller):
    c = controller
    qa(c)
    c.on_transcript('final', 'Why is the model identifiable?', context(c, 32000))
    # Its audio clock advanced during silence; no new lexical speech exists.
    c.on_transcript('partial', 'the model identifiable', context(c, 80000, 28000))
    assert c._partial_id is None
    c.assembler.last_voice = c.assembler.last_text = 0
    c.auto_answer()
    assert c.busy


def test_pending_recognition_and_true_rejection_still_block_early_answer(controller):
    c = controller
    qa(c)
    c.on_transcript('final', 'How do you calibrate the instrument?', context(c, 32000))
    c.speech = SimpleNamespace(pending=1, stop=lambda **kw:None)
    c.assembler.last_voice = c.assembler.last_text = 0
    c.auto_answer()
    assert not c.busy
    c.speech.pending = 0
    c.on_transcript('rejected', '', context(c, 48000))
    c.auto_answer()
    assert not c.busy


def test_single_remote_keeps_professors_own_explanation_as_question_background(controller):
    c = controller
    c.pref.caption_speaker_mode = 'single_remote'
    c._listen_pref = replace(c.pref)
    qa(c)
    c.on_transcript('final', 'I think the illumination is unknown in this setup.', context(c, 16000))
    c.on_transcript('final', 'How would you separate material reflectance?', context(c, 32000))
    assert c.assembler.text.startswith('I think the illumination')
    assert c.assembler.text.endswith('material reflectance?')
    assert len(c.question_context.build(c.assembler.text).turns) >= 1


def test_readable_translation_units_do_not_split_or_drop_the_question(controller):
    c = controller
    c.pref.caption_speaker_mode = 'single_remote'
    c._listen_pref = replace(c.pref)
    c.listening = True
    first = ('Consider a computational imaging system with spectral polarization measurements '
             'collected in a strongly scattering environment. The illumination and material '
             'reflectance are both unknown in this experiment.')
    second = 'How would you formulate the forward model and evaluate identifiability?'
    c.on_transcript('final', first, context(c, 32000))
    c.on_transcript('final', second, context(c, 64000))
    assert len(c._transcripts) == 2
    assert c._transcripts[0]['raw_en'] == first
    assert c._transcripts[1]['raw_en'] == second
    assert c.assembler.text == first + ' ' + second


def test_content_free_stage_timings_distinguish_retrieval_and_provider(controller):
    c = controller
    c.request_answer('How would you calibrate the imaging system?')
    c.test_run_worker()
    timing = c.last_answer_timing
    assert timing['answer_queue_ms'] >= 0
    assert timing['answer_retrieval_ms'] >= 0
    assert timing['answer_provider_ttft_ms'] >= 0
    assert timing['answer_total_ms'] >= 0
    assert 'calibrate' not in str(timing)


def test_preferences_persist_endpoint_following_and_personal_profile(tmp_path, monkeypatch):
    monkeypatch.setenv('PRISM_INTERVIEW_HOME', str(tmp_path))
    p = Preferences(candidate_profile='My verified PhD topic.', follow_latest=False,
                    auto_endpoint_seconds=.8, caption_speaker_mode='acoustic')
    p.save()
    loaded = Preferences.load()
    assert loaded.candidate_profile == p.candidate_profile
    assert loaded.follow_latest is False
    assert loaded.auto_endpoint_seconds == .8
    assert loaded.caption_speaker_mode == 'acoustic'


@pytest.mark.parametrize('courtesy', ['Thanks.', 'Thank you.', 'Thank you very much!',
                                     'Thank you for listening.', 'you for listening.'])
def test_pure_closing_preserves_caption_without_replacing_inflight_answer(controller, courtesy):
    c = controller
    qa(c)
    translated = []
    c.translation = SimpleNamespace(submit=lambda key, text:translated.append((key,text)), stop=lambda:None)
    c.on_transcript('final', 'How do you calibrate the instrument?', context(c, 16000))
    c.assembler.last_voice = c.assembler.last_text = 0
    c.auto_answer()
    assert c.busy
    active = c._active
    question = c.assembler.text
    version = c.assembler.version
    c.on_transcript('final', courtesy, context(c, 32000))
    c.auto_answer()
    assert c.assembler.text == question and c.assembler.version == version
    assert c._question_original == question
    assert c._active == active and not c._cancel.is_set() and not c._requests
    assert c._transcripts[-1]['raw_en'] == courtesy
    assert translated and translated[-1][0][0] == c._transcripts[-1]['id']
    c.test_run_worker()
    c.auto_answer()
    assert len(c.service.calls) == 1 and not c.busy
    assert c.window.answer_english.text() == 'Suggested English answer.'


def test_completed_answer_survives_courtesy_but_mixed_scientific_followup_is_kept(controller):
    c = controller
    qa(c)
    c.on_transcript('final', 'How do you calibrate the instrument?', context(c, 16000))
    c.assembler.last_voice = c.assembler.last_text = 0
    c.auto_answer()
    c.test_run_worker()
    old_version = c.assembler.version
    c.on_transcript('final', 'Thank you.', context(c, 32000))
    c.auto_answer()
    assert c.assembler.version == old_version and len(c.service.calls) == 1 and not c.busy
    c.on_transcript('final', 'Thank you. How would you evaluate uncertainty?', context(c, 48000))
    assert c.assembler.version > old_version
    assert 'How would you evaluate uncertainty?' in c.assembler.text
    assert 'Thank you. How would you evaluate uncertainty?' in c._transcripts[-1]['raw_en']


@pytest.mark.parametrize('text', ['Thank you, but can you explain the calibration?',
                                'Thank you for defining the inverse problem.',
                                'Our material model estimates reflectance. Thank you.',
                                'How would you validate the model?'])
def test_courtesy_allowlist_does_not_silently_remove_mixed_or_scientific_speech(text):
    from zoom_copilot.app import CopilotController
    assert not CopilotController._closing_courtesy_only(text)

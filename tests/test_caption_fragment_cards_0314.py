"""Visible continuation cards and recognition integrity at the controller boundary.

The screenshot words are confirmed-text inputs, not recovered Zoom audio.
"""
from dataclasses import replace
import time

import pytest

from test_controller_v2 import controller
from test_caption_continuation_031 import setup, say


LONG = ('We calibrate the optical system before acquisition and compare all '
        'spectral channels against a reference measurement to estimate uncertainty '
        'under changing illumination and scattering conditions.')


@pytest.mark.parametrize('continuation', ['Or convey', 'And the', 'I mean'])
def test_short_hesitant_continuation_stays_in_the_same_visible_card(controller, continuation):
    c = controller
    calls = setup(c)
    say(c, LONG)
    identity = c._transcripts[0]['id']
    widget = c.window._entries[identity]
    say(c, continuation, 1)
    assert len(c._transcripts) == 1
    assert c._transcripts[0]['id'] == identity
    assert c.window._entries[identity] is widget
    assert c._transcripts[0]['raw_en'] == LONG + ' ' + continuation
    assert calls[-1][0][0] == identity
    assert continuation.casefold().strip('.') in calls[-1][1].casefold()


def test_real_speaker_change_still_starts_a_new_card_for_a_short_continuation(controller):
    c = controller
    setup(c, single=False)
    say(c, LONG, speaker='interviewer')
    say(c, 'Or convey', 1, speaker='candidate')
    assert len(c._transcripts) == 2
    assert c._transcripts[0]['speaker_id'] != c._transcripts[1]['speaker_id']


@pytest.mark.parametrize('reply', ['I agree.', 'And I agree.', 'So I understand.'])
def test_short_but_complete_response_can_start_a_new_paragraph(controller, reply):
    c = controller
    setup(c)
    say(c, LONG)
    say(c, reply, 1)
    assert len(c._transcripts) == 2


def qa(c):
    c.pref.caption_speaker_mode = 'single_remote'
    c._listen_pref = replace(c.pref)
    c.change_interview_phase('qa')
    c.listening = True


def pause(c):
    c.assembler.last_voice = c.assembler.last_text = time.monotonic() - 2
    c.auto_answer()


@pytest.mark.parametrize('text', [
    'How would you validate this material estimate? Sure.',
    'Do you have any questions? How would you validate this material estimate? Sure.',
    'I have another question. Please explain your optical calibration procedure.',
])
def test_mixed_accepted_rejected_crop_preserves_caption_but_cannot_auto_answer(controller, text):
    c = controller
    qa(c)
    c.on_transcript('final', text, {'phase_epoch': c._phase_epoch, 'recognition_gap': True})
    pause(c)
    assert c._question_uncertain
    assert not c.busy and not c._requests and c._answer_prefetch is None
    assert text in ' '.join(row['raw_en'] for row in c._transcripts)
    c.request_answer(c.assembler.text, automatic=True)
    assert not c.busy


def test_manual_review_can_explicitly_answer_after_mixed_recognition_gap(controller):
    c = controller
    qa(c)
    question = 'Please explain your optical calibration procedure.'
    c.on_transcript('final', question, {'recognition_gap': True})
    pause(c)
    assert not c.busy
    c.request_answer(question)
    assert c.busy


def test_old_phase_recognition_gap_cannot_invalidate_a_current_clean_question(controller):
    c = controller
    qa(c)
    c.on_transcript('final', 'Some old phase words.',
                    {'phase_epoch': c._phase_epoch - 1, 'recognition_gap': True})
    assert not c._question_uncertain
    c.on_transcript('final', 'How would you validate this material estimate?')
    pause(c)
    assert c.busy


def test_idle_self_question_with_gap_cannot_block_the_next_clean_request(controller):
    c = controller
    qa(c)
    text = 'How can we acquire spectral information? This question motivated my PhD research.'
    c.on_transcript('final', text, {'recognition_gap': True})
    c.assembler.last_voice = c.assembler.last_text = time.monotonic() - 20
    c.auto_answer()
    assert not c._question_uncertain and not c.assembler.text and not c.busy
    c.on_transcript('final', 'Please explain your optical calibration procedure.')
    pause(c)
    assert c.busy
    assert 'motivated' not in c._question_raw_by_request[c._active]

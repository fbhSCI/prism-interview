"""Confirmed acoustic windows are not necessarily finished English sentences."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_controller_v2 import controller
from zoom_copilot.transcript_text import TranscriptRestorer


def setup(c, single=True):
    c.pref.caption_speaker_mode = 'single_remote' if single else 'acoustic'
    c._listen_pref = replace(c.pref)
    c.listening = True
    c.restorer = TranscriptRestorer(punctuator=lambda text: text.rstrip('.') + '.')
    calls = []
    c.translation = SimpleNamespace(submit=lambda key, text: calls.append((key, text)), stop=lambda: None)
    return calls


def say(c, text, number=0, speaker='speaker_1', hard_cut=True):
    c.on_transcript('final', text, dict(phase_epoch=c._phase_epoch, stream_id='recording',
        start_sample=number*32000, end_sample=(number+1)*32000, word_timed=True,
        hard_cut=hard_cut, speaker_id=speaker, turn_id=speaker, speaker_label=speaker))


def test_user_passive_continues_same_translation_unit_and_replaces_old_chinese(controller):
    c = controller
    calls = setup(c)
    prefix = ('We train the components together, right? '
              'So this architecture combines multiple learned components. I, Mean, it. '
              'The weights in the prediction network and the preprocessing module '
              'are learned and are.')
    say(c, prefix)
    first = c._transcripts[0]['id']
    old_key = calls[-1][0]
    c.post('translation', (c._session, old_key, '此前的半句译文'))
    c.pump()
    say(c, 'Completely, back-propagated.', 1)
    assert len(c._transcripts) == 1
    row = c._transcripts[0]
    assert row['id'] == first
    assert row['raw_en'] == prefix + ' Completely, back-propagated.'
    assert 'are learned and are completely back-propagated' in calls[-1][1]
    assert not calls[-1][1].startswith('How')
    assert row['zh'] == '此前的半句译文'
    assert row['translation_pending']
    assert first in c.window._entries[first]._pending_translations
    c.post('translation', (c._session, old_key, '过时结果'))
    c.pump()
    assert row['zh'] == '此前的半句译文' and row['translation_pending']
    c.post('translation', (c._session, calls[-1][0], '所有权重通过学习获得，并进行反向传播。'))
    c.pump()
    assert not row['translation_pending']
    assert row['zh'].endswith('反向传播。')
    assert first not in c.window._entries[first]._pending_translations


@pytest.mark.parametrize('ending,continuation,phrase', [
    ('a particular mathematical.', 'Model for motion.', 'mathematical model'),
    ('use neural.', 'Networks for motion.', 'neural networks'),
])
def test_whisper_period_after_modifier_does_not_seal(controller, ending, continuation, phrase):
    c = controller
    calls = setup(c)
    prefix = 'We discuss the optical system and its reconstruction under different conditions. ' * 3 + ending
    say(c, prefix)
    assert not calls[-1][1].endswith('.')
    say(c, continuation, 1)
    assert len(c._transcripts) == 1
    assert phrase in calls[-1][1]
    assert c._transcripts[0]['raw_en'] == prefix + ' ' + continuation


def test_long_unpunctuated_clause_is_not_cut_at_320_characters(controller):
    c = controller
    calls = setup(c)
    prefix = ('For the reconstruction of scenes captured with the optical system '
              'under changing illumination and strong scattering we evaluate not '
              'only the individual observations from each detector but also their '
              'joint consistency across the different spectral and polarization '
              'channels using the calibrated measurement matrix and a regularized method')
    assert len(prefix) > 320
    say(c, prefix)
    assert len(calls) == 1  # No extra wait for the next clause or model call.
    assert c._transcripts[0]['en'] and not calls[-1][1].endswith('.')
    say(c, 'that preserves uncertainty below 0.5 units.', 1)
    assert len(c._transcripts) == 1 and len(calls) == 2
    assert 'not only' in calls[-1][1] and '0.5' in calls[-1][1]


def test_genuine_complete_sentence_can_start_next_paragraph(controller):
    c = controller
    setup(c)
    prefix = ('We calibrate the optical system before acquisition and compare all '
              'spectral channels against a reference measurement to estimate uncertainty '
              'under changing illumination and scattering conditions.')
    say(c, prefix)
    say(c, 'How would you validate the material estimate?', 1)
    assert len(c._transcripts) == 2
    assert c._transcripts[0]['raw_en'] == prefix


def test_a_different_speaker_never_continues_previous_predicate(controller):
    c = controller
    setup(c, single=False)
    say(c, 'The weights in the network are learned and are.', speaker='speaker_1')
    say(c, 'Completely different methods were used.', 1, speaker='speaker_2')
    assert len(c._transcripts) == 2
    assert c._transcripts[0]['speaker_id'] != c._transcripts[1]['speaker_id']


def test_hard_cut_keeps_genuine_question_mark(controller):
    c = controller
    calls = setup(c)
    say(c, 'How would you validate the model?')
    assert calls[-1][1].endswith('?')


def test_rejected_next_window_cannot_invalidate_pending_prefix_translation(controller):
    c = controller
    calls = setup(c)
    say(c, 'The model was learned')
    key, source = calls[-1]
    assert not source.endswith('.')
    c.on_transcript('rejected', '', dict(phase_epoch=c._phase_epoch))
    assert c._caption_sources[key[0]] == source
    assert len(calls) == 1
    c.post('translation', (c._session, key, '模型通过学习获得'))
    c.pump()
    assert c._transcripts[0]['zh'] == '模型通过学习获得'
    assert not c._transcripts[0]['translation_pending']


def test_ellipsis_restart_and_coordinated_subject_stay_in_one_translation_unit(controller):
    c = controller
    calls = setup(c)
    pieces = [
        'So this is an end-to-end learning system, right? We discussed the architecture '
        'and its reconstruction under different conditions. I mean, basically, all of the weights...',
        'weights in the motion synthesis network.',
        'and the encoder and decoder are',
        'learned and are completely back propagated.',
    ]
    for number, text in enumerate(pieces):
        say(c,text,number)
    assert len(c._transcripts) == 1
    assert c._transcripts[0]['raw_en'] == ' '.join(pieces)
    assert 'weights in the motion synthesis network and the encoder and decoder are learned' in calls[-1][1]

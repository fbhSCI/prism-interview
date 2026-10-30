"""Synthetic question whose neural/network phrase is split at a hard ASR cut."""
from dataclasses import replace
from types import SimpleNamespace

from test_controller_v2 import controller
from zoom_copilot.transcript_text import TranscriptRestorer


def test_punctuation_repair_does_not_split_neural_networks_between_translation_units(controller):
    c = controller
    c.pref.caption_speaker_mode = 'single_remote'
    c._listen_pref = replace(c.pref)
    c.listening = True
    # Reproduces the punctuation model adding a period at a hard ASR window.
    # There was no audible sentence boundary after the adjective "neural".
    c.restorer = TranscriptRestorer(punctuator=lambda text: text.rstrip('.') + '.')
    translations = []
    c.translation = SimpleNamespace(
        submit=lambda key, text: translations.append((key, text)), stop=lambda: None)
    prefix = ('Alex, I have a question about the design in your project. '
              'Your pipeline combines learned components with fixed constraints. '
              'Could you explain the parts where you use neural')
    continuation = 'networks for and then other parts where you do more classical modeling.'

    def say(text, start, end):
        c.on_transcript('final', text, dict(phase_epoch=c._phase_epoch,
            stream_id='tum-question', start_sample=start, end_sample=end,
            word_timed=True, hard_cut=True))

    say(prefix, 0, 32000)
    first_id = c._transcripts[0]['id']
    assert c._caption_sources[first_id].endswith('neural')
    first_translation_key = translations[-1][0]
    c.post('translation', (c._session, first_translation_key, '此前已显示的中文'))
    c.pump()

    say(continuation, 32000, 64000)
    assert len(c._transcripts) == 1
    assert c._transcripts[0]['id'] == first_id
    assert c._transcripts[0]['raw_en'] == prefix + ' ' + continuation
    assert 'neural networks' in translations[-1][1]
    assert translations[-1][0][0] == first_id
    # Reading text remains visible until the correctly paired update arrives.
    assert c._transcripts[0]['zh'] == '此前已显示的中文'
    c.post('translation', (c._session, translations[-1][0], '神经网络与传统建模'))
    c.pump()
    assert c._transcripts[0]['zh'] == '神经网络与传统建模'
    assert c.assembler.text == prefix + ' ' + continuation

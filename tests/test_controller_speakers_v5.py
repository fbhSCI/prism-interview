"""Voice-turn display boundaries must never truncate the actual question."""
from test_controller_v2 import controller


def context(ctrl, who, turn):
    return {'phase_epoch': ctrl._phase_epoch, 'word_timed': True,
            'speaker_id': who, 'speaker_label': f'说话人 {who}', 'turn_id': turn}


def test_acoustic_turn_separates_cards_but_preserves_question_context(controller):
    c=controller
    c.on_transcript('final', 'How do you model the scattering?', context(c,1,'turn1'))
    c.on_transcript('final', 'And which measurements constrain the material?', context(c,2,'turn2'))
    c.on_transcript('final', 'In particular, which polarization channels?', context(c,1,'turn3'))
    assert len(c._transcripts)==3
    assert [r['speaker_id'] for r in c._transcripts]==[1,2,1]
    assert c.assembler.text=='How do you model the scattering? And which measurements constrain the material? In particular, which polarization channels?'
    assert len(set(r['id'] for r in c._transcripts))==3


def test_same_voice_continuations_group_and_long_paragraphs_are_bounded(controller):
    c=controller
    meta=context(c,1,'turn1')
    first='How does the inverse problem change under scattering? '
    long='We also need calibrated spectral and polarization measurements. '*8
    c.on_transcript('final',first,meta)
    c.on_transcript('final',long,meta)
    assert len(c._transcripts)==1
    c.on_transcript('final','How would you validate these assumptions?',meta)
    assert len(c._transcripts)==2
    assert c.assembler.text.startswith(first.strip())
    assert long.strip() in c.assembler.text
    assert c._transcripts[1]['en']=='How would you validate these assumptions?'


def test_old_phase_speaker_text_is_history_only_and_translatable(controller):
    c=controller
    meta=context(c,2,'old-turn')
    c.change_interview_phase('qa')
    c.on_transcript('final','The old presentation conclusion.',meta)
    assert not c.assembler.text
    assert c._transcripts[0]['speaker_id']==2
    assert c._transcripts[0]['en']=='The old presentation conclusion.'


def test_switching_provider_never_reuses_wrong_service_key_or_persists_it(controller):
    c=controller
    c.save_settings({'provider':'deepseek','api_key':'test-deepseek-secret'})
    c.save_settings({'provider':'qwen','api_key':'test-deepseek-secret'})
    assert c.pref.api_key==''
    assert c.pref.session_api_keys.get('deepseek')=='test-deepseek-secret'
    assert 'qwen' not in c.pref.session_api_keys
    c.save_settings({'provider':'openai_compatible','api_key':'test-compatible-secret'})
    assert c.pref.session_api_keys['compatible']=='test-compatible-secret'
    from zoom_copilot.config import Preferences,app_dir
    assert not Preferences.load().api_key and not Preferences.load().session_api_keys
    saved=(app_dir()/'preferences.json').read_text(encoding='utf-8')
    assert 'test-deepseek-secret' not in saved
    assert 'test-compatible-secret' not in saved

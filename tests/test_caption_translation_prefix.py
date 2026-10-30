"""Confirmed-source translation progress without drafts, regression or cloud."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
from test_controller_v2 import controller


def setup(c, english):
    c.pref.translation_engine='qwen'
    c.pref.translation_cloud_consent=True
    c._listen_pref=replace(c.pref)
    c._caption_sources['source']=english
    c._transcripts.append({'id':'source','en':english,'zh':'','translation_pending':True})
    c.window.append_transcript('source',english,'',speaker_label='面试官',translation_pending=True)
    c._live_data={'entry_id':'source','source':english,'en':english,'zh':'',
                  'state':'已确认片段 · 正在更新中文'}


def deliver(c, source, chinese, *, identity='source', session=None):
    c.post('translation',(c._session if session is None else session,(identity,source),chinese))
    c.pump()


def test_confirmed_prefix_appears_while_suffix_pending_then_latest_completes(controller):
    c=controller;current='We estimate reflectance under varying illumination.'
    setup(c,current);c.window.show_subtitle_overlay()
    card=c.window._entries['source'];unit=card._unit_widgets['source']
    floating=c.window._overlay.caption._cards['source'];pair=floating.pairs[0]
    deliver(c,'We estimate reflectance','我们估计反射特性。')
    assert unit.chinese.text()=='我们估计反射特性。'
    assert unit.english.text()==current and not unit.pending.isHidden()
    assert c._transcripts[0]['translation_pending']
    assert 'source' in card._pending_translations
    assert pair.chinese.text()=='我们估计反射特性。'
    assert '等待中文' in floating.header.text()
    assert c._live_data['state']=='已确认片段 · 正在更新中文'
    deliver(c,current,'我们在变化照明下估计反射特性。')
    assert unit.chinese.text()=='我们在变化照明下估计反射特性。'
    assert unit.pending.isHidden() and not c._transcripts[0]['translation_pending']
    assert 'source' not in card._pending_translations
    assert c._live_data['state']=='已成句'
    assert pair is floating.pairs[0] and card._unit_widgets['source'] is unit
    assert pair.chinese.text()=='我们在变化照明下估计反射特性。'
    assert '等待中文' not in floating.header.text()


def test_out_of_order_shorter_and_duplicate_callbacks_never_regress(controller):
    c=controller;current='We estimate reflectance under varying illumination.'
    setup(c,current)
    deliver(c,'We estimate','我们估计。')
    deliver(c,'We estimate reflectance','我们估计反射特性。')
    deliver(c,'We estimate','迟到的短结果。')
    deliver(c,'We estimate reflectance','重复但不同的旧结果。')
    assert c._transcripts[0]['zh']=='我们估计反射特性。'
    assert c._transcripts[0]['translation_pending']
    assert c._caption_translation_sources[(c._session,'source')]=='We estimate reflectance'
    deliver(c,current,'完整结果。')
    deliver(c,'We estimate reflectance','较早结果。')
    assert c._transcripts[0]['zh']=='完整结果。'
    assert not c._transcripts[0]['translation_pending']


@pytest.mark.parametrize('source,current',[
    ('We can','We cannot estimate reflectance.'),
    ('We can',"We can't estimate reflectance."),
    ('The value is 1','The value is 195.'),
    ('The value is 1','The value is 1.95.'),
    ('The value is 1.','The value is 1.95.'),
    ('The value is 95','The value is 95%.'),
    ('It is wavelength','It is wavelength-dependent.'),
    ('The channel is S','The channel is S0.'),
    ('We measure','We measured reflectance.'),
    ('we estimate','We estimate reflectance.'),
])
def test_character_prefix_inside_word_number_or_changed_casing_is_not_accepted(controller,source,current):
    c=controller;setup(c,current)
    deliver(c,source,'不得显示此译文。')
    assert c._transcripts[0]['zh']==''
    assert not c._caption_translation_sources
    assert c.window._entries['source']._segments['source'][1]==''


def test_punctuation_word_boundary_is_valid_without_normalizing_source(controller):
    c=controller;setup(c,'We measure reflectance, including its uncertainty.')
    deliver(c,'We measure reflectance','我们测量反射特性。')
    assert c._transcripts[0]['zh']=='我们测量反射特性。'
    assert c._transcripts[0]['translation_pending']


def test_corrected_nonprefix_rejects_old_source_but_latest_can_replace_old_watermark(controller):
    c=controller;old='We can estimate reflectance with an optical model.'
    setup(c,old);deliver(c,old,'我们能够估计反射特性。')
    current='We cannot estimate it.'
    c._caption_sources['source']=current
    c._transcripts[0].update(en=current,translation_pending=True)
    c.window.update_message('source',current,c._transcripts[0]['zh'],translation_pending=True)
    c._live_data.update(en=current,source=current,state='已确认片段 · 正在更新中文')
    deliver(c,old,'旧肯定说法不得替换修订。')
    assert c._transcripts[0]['zh']=='我们能够估计反射特性。'
    assert c._transcripts[0]['translation_pending']
    deliver(c,current,'我们无法估计它。')
    assert c._transcripts[0]['zh']=='我们无法估计它。'
    assert not c._transcripts[0]['translation_pending']


@pytest.mark.parametrize('invalid',['session','consent','identity','unbound','empty'])
def test_session_consent_identity_and_bound_source_guards(controller,invalid):
    c=controller;setup(c,'We estimate reflectance.')
    if invalid=='session':deliver(c,'We estimate','无效结果。',session=c._session-1)
    elif invalid=='consent':
        c.pref.translation_cloud_consent=False;deliver(c,'We estimate','无效结果。')
    elif invalid=='identity':deliver(c,'We estimate','无效结果。',identity='other')
    elif invalid=='unbound':
        c.post('translation',(c._session,'source','无效结果。'));c.pump()
    else:deliver(c,'We estimate','')
    assert c._transcripts[0]['zh']=='' and not c._caption_translation_sources
    assert c._transcripts[0]['translation_pending']


def test_unconfirmed_drafts_are_never_translation_sources_or_uploads(controller):
    c=controller;setup(c,'We can')
    calls=[]
    c.translation=SimpleNamespace(submit=lambda *args:calls.append(args),
                                  submit_preview=lambda *args:calls.append(args),stop=lambda:None)
    c.pref.clean_speech=False;c.pref.live_translation=False
    c.on_transcript('partial','We cannot estimate reflectance.',{'phase_epoch':c._phase_epoch})
    assert calls==[]
    deliver(c,'We cannot estimate reflectance.','草稿不得作为确认来源。')
    assert c._transcripts[0]['zh']=='' and c._caption_sources['source']=='We can'


def test_pending_prefix_direct_ui_update_keeps_status_until_explicit_complete(controller):
    c=controller;setup(c,'We estimate reflectance under varying illumination.')
    c.window.show()
    card=c.window._entries['source'];unit=card._unit_widgets['source']
    c.window.update_translation('source','我们估计反射特性。',translation_pending=True)
    assert unit.pending.isVisible() and 'source' in card._pending_translations
    c.window.update_translation('source','我们在变化照明下估计反射特性。')
    assert not unit.pending.isVisible() and 'source' not in card._pending_translations

"""Cloud caption consent, endpoint isolation, source binding and cancellation."""
import json
import threading
from types import SimpleNamespace

import pytest

from zoom_copilot.config import Preferences
from zoom_copilot import translation_qwen as m
from zoom_copilot.translation import TranslationWorker


def preferences(**kw):
    return Preferences(**{**dict(provider='qwen', translation_cloud_consent=True,
        api_base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        qwen_paid_usage_authorized=True, api_key='test-in-memory'), **kw})


def test_continuous_consent_is_independent_and_fail_closed():
    with pytest.raises(ValueError, match='连续英文'):
        m.QwenCaptionTranslator(preferences(translation_cloud_consent=False, cloud_consent=True))
    assert m.QwenCaptionTranslator(preferences(cloud_consent=False)).ready


def test_exact_endpoint_key_never_crosses_from_answer_endpoint(monkeypatch):
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'must-not-use-unscoped-env-key')
    seen = []
    monkeypatch.setattr(m.vault, 'load_key', lambda p, b: seen.append((p,b)) or '')
    with pytest.raises(ValueError, match='尚无'):
        m.QwenCaptionTranslator(preferences(api_base_url='https://dashscope-intl.aliyuncs.com/compatible-mode/v1',
            session_api_keys={'qwen':'must-not-cross-endpoint'}))
    assert seen == [('qwen', 'https://dashscope.aliyuncs.com/compatible-mode/v1')]


def test_deepseek_session_key_not_used_for_caption(monkeypatch):
    seen=[]
    monkeypatch.setattr(m.vault,'load_key',lambda p,b: seen.append(p) or 'test-vault-qwen')
    translator=m.QwenCaptionTranslator(preferences(provider='deepseek',api_key='test-deepseek',
        session_api_keys={'deepseek':'test-deepseek','qwen':'stale-other-endpoint'}))
    assert translator._key=='test-vault-qwen' and seen==['qwen']


def test_http_payload_only_contains_confirmed_segments_and_matching_terms(monkeypatch):
    seen=[]
    def complete(req,timeout,protocol,**kw):
        seen.append((json.loads(req.data),timeout,protocol,kw))
        return json.dumps({'translations':['不需要光照采集装置。','使用95%的数据。']},ensure_ascii=False)
    monkeypatch.setattr(m,'stream_http',complete)
    translator=m.QwenCaptionTranslator(preferences(candidate_profile='PRIVATE-CV',research_profile='PRIVATE-PAPERS'))
    assert translator.translate_batch(['You would not need a light stage.','Use 95% of the data.']) == [
        '不需要光照采集装置。','使用95%的数据。']
    payload,_,protocol,kw=seen[0]
    data=json.loads(payload['messages'][1]['content'])
    assert data['segments']==['You would not need a light stage.','Use 95% of the data.']
    assert data['terminology']=={'light stage':'光照采集装置'}
    assert 'PRIVATE' not in json.dumps(payload)
    assert payload['enable_thinking'] is False and payload['stream'] is False
    assert protocol=='compatible' and kw['cancel_event'] is translator._cancel


@pytest.mark.parametrize('output',[
    {'translations':['您需要光照采集装置。']}, # missing negation
    {'translations':['不需要光照采集装置，使用数据。']}, # missing number
    {'translations':[]}, {'translations':['不需要。','多余。']},
    {'translations':[123]}, {'answer':'不需要。'},
])
def test_bad_or_misaligned_output_not_published(monkeypatch,output):
    monkeypatch.setattr(m,'stream_http',lambda *a,**kw:json.dumps(output,ensure_ascii=False))
    translator=m.QwenCaptionTranslator(preferences())
    with pytest.raises(ValueError):translator.translate('You would not need 95% of the data.')


def test_cancel_event_is_set_without_waiting_for_http():
    translator=m.QwenCaptionTranslator(preferences())
    worker=TranslationWorker(translator,lambda *a:None)
    assert worker.submit_preview('x','unconfirmed draft') is False
    worker.stop()
    assert translator._cancel.is_set()


@pytest.mark.parametrize('source,output',[
    ('Use 95% of the data.','使用195%的数据。'),
    ('Use 5 and 5 measurements.','使用5次测量。'),
    ('Use 5 measurements.','使用5次测量和7个通道。'),
    ('The result is -0.9.','结果为0.9。'),
])
def test_numeric_changes_and_count_omissions_rejected(source,output):
    with pytest.raises(ValueError,match='数值'):m._check_translation(source,output)


def test_numerals_adjacent_to_chinese_and_percent_are_preserved():
    m._check_translation('The confidence is 95%.','置信度为95%。')
    m._check_translation('Use 5 and 5 measurements.','使用5次和5次测量。')


@pytest.mark.parametrize('translated',['使用532 nm照明。','使用532纳米照明。','使用532nm照明。'])
def test_attached_physical_units_do_not_hide_numeric_values(translated):
    m._check_translation('Use 532nm illumination.',translated)
    with pytest.raises(ValueError,match='数值'):
        m._check_translation('Use 532nm illumination.',translated.replace('532','533'))


def test_coherent_batch_reuses_stable_prefix_and_only_uploads_growing_tail():
    calls=[]
    def batch(items):
        calls.append(items)
        return ['中文:'+x for x in items]
    worker=TranslationWorker(SimpleNamespace(translate_batch=batch,confirmed_only=True),lambda *a:None)
    prefix=('We evaluate the reconstruction under controlled changes in illumination, scattering, '
            'and object motion, using held out measurements and independent reference images '
            'to assess the limitations of each method.')
    first,hits,count=worker._translate_confirmed(('card','revision1'),prefix+' Second sentence.')
    second,hits2,count2=worker._translate_confirmed(('card','revision2'),prefix+' Second sentence. A new question?')
    assert calls==[[prefix,'Second sentence.'],['Second sentence. A new question?']]
    assert hits==0 and count==1 and hits2==1 and count2==1
    assert second.startswith(first)


def test_formula_batch_failure_does_not_cache_any_misaligned_piece():
    worker=TranslationWorker(SimpleNamespace(translate_batch=lambda items:['正确中文。','缺失公式。']),lambda *a:None)
    prefix=('We evaluate the reconstruction under controlled changes in illumination, scattering, '
            'and object motion, using held out measurements and independent reference images '
            'to assess the limitations of each method.')
    with pytest.raises(ValueError):
        worker._translate_confirmed('card',prefix+' s equals D i not M i.')
    assert not worker._sentence_cache


def test_cloud_batch_bound_is_full_text_no_loss():
    calls=[]
    def batch(items):calls.append(items);return ['中文'+x for x in items]
    worker=TranslationWorker(SimpleNamespace(translate_batch=batch),lambda *a:None)
    sentences=['Sentence '+str(i)+' '+('word '*95)+'.' for i in range(12)]
    source=' '.join(sentences)
    translated,_,n=worker._translate_confirmed('card',source)
    assert [x for batch in calls for x in batch]==sentences
    assert all(len(''.join(batch))<=2000 and len(batch)<=8 for batch in calls)
    assert n==len(calls) and all(x in translated for x in sentences)

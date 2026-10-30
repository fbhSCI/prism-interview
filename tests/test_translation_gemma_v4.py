import json
from contextlib import contextmanager
from types import SimpleNamespace
import pytest
from zoom_copilot.translation_gemma import GemmaTranslator, MODEL, PROMPT, translation_prompt
from zoom_copilot import translation_gemma as module
from zoom_copilot.translation import TranslationWorker
from zoom_copilot.providers import OpenAICompatibleClient
from zoom_copilot.config import Preferences


def test_gemma_only_sends_translation_to_loopback_and_rejects_truncation(tmp_path, monkeypatch):
    calls = []
    answer = {'done':True, 'done_reason':'stop', 'message':{'content':'散射会造成什么影响？'}}
    @contextmanager
    def local(request, timeout):
        import io
        calls.append(request)
        yield io.BytesIO(json.dumps(answer).encode())
    monkeypatch.setattr(module, 'urlopen', local)
    t = GemmaTranslator(tmp_path)
    assert t.translate('How does scattering affect it?') == '散射会造成什么影响？'
    req = calls[0]
    assert req.full_url == 'http://127.0.0.1:11434/api/chat'
    data = json.loads(req.data)
    assert data['model'] == MODEL
    assert data['messages'] == [{'role':'user','content':PROMPT+'How does scattering affect it?'}]
    assert 'Authorization' not in req.headers
    answer['done_reason'] = 'length'
    with pytest.raises(RuntimeError, match='未完整'):
        t.translate('Another sentence')


def test_newest_combined_caption_replaces_pending_prefix_only():
    worker = TranslationWorker(None, lambda *args: None)
    worker.submit(('a', 'prefix'), 'prefix')
    worker.submit(('b', 'unrelated'), 'unrelated')
    worker.submit(('a', 'prefix and suffix'), 'prefix and suffix')
    assert [job[1] for job in worker._jobs] == ['unrelated', 'prefix and suffix']


def test_glossary_only_supplies_terms_present_in_translation_source():
    plain = 'Could you explain your method?'
    assert translation_prompt(plain) == PROMPT + plain
    text = 'Does the sensing operator uniquely determine the spectrum?'
    prompt = translation_prompt(text)
    assert 'sensing operator = 测量算子' in prompt
    assert 'material reflectance =' not in prompt
    assert prompt.endswith(text)


def test_model_sense_hint_is_only_translation_context_and_keeps_original_terms():
    text='A lot of AA models get developed every day. Let us take turns naming models.'
    prompt=translation_prompt(text)
    assert 'computational or scientific model, use 模型' in prompt
    assert prompt.endswith(text) and 'AA models' in prompt
    assert 'computational or scientific model, use 模型' not in translation_prompt('Can you hear me?')


def test_deepseek_flash_requires_free_confirmation_and_disables_thinking(monkeypatch):
    p = Preferences(provider='deepseek', cloud_consent=True, answer_model='deepseek-flash', api_key='test-only')
    with pytest.raises(ValueError, match='免费额度'):
        OpenAICompatibleClient(p)
    p.free_quota_confirmed = True
    calls=[]
    monkeypatch.setattr('zoom_copilot.providers.stream_http', lambda request, *args, **kwargs: calls.append(request) or '{}')
    client=OpenAICompatibleClient(p)
    client.complete('system','question')
    assert calls[0].full_url == 'https://api.deepseek.com/v1/chat/completions'
    payload=json.loads(calls[0].data)
    assert payload['thinking'] == {'type':'disabled'}
    assert payload['model'] == 'deepseek-flash'

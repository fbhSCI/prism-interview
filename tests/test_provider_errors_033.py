"""Report actionable billing status without exposing provider error content."""
import io
import json
from urllib.error import HTTPError
from urllib.request import Request
import pytest
from zoom_copilot.providers import stream_http


def test_arrearage_400_is_a_billing_problem_and_never_echoes_provider_body():
    body=json.dumps({'error':{'code':'Arrearage','message':'test-secret and private paper excerpt'}}).encode()
    calls=[]
    def opener(request,timeout):
        calls.append(request)
        raise HTTPError(request.full_url,400,'Bad Request',{},io.BytesIO(body))
    with pytest.raises(RuntimeError,match='计费不可用') as caught:
        stream_http(Request('https://example.com/v1/chat/completions'),1,'compatible',lambda d:None,opener=opener)
    assert 'Arrearage' in str(caught.value)
    assert 'test-secret' not in str(caught.value) and 'paper' not in str(caught.value)
    assert len(calls)==1


def test_unknown_error_body_is_not_exposed():
    def opener(request,timeout):
        raise HTTPError(request.full_url,400,'Bad Request',{},io.BytesIO(b'{"error":{"message":"private-content","code":"sensitive-key"}}'))
    with pytest.raises(RuntimeError,match='HTTP 400') as caught:
        stream_http(Request('https://example.com/v1/chat/completions'),1,'compatible',lambda d:None,opener=opener)
    assert 'private-content' not in str(caught.value) and 'sensitive-key' not in str(caught.value)


def test_sse_billing_error_does_not_get_reported_as_an_answer():
    wire=b'data: {"error":{"code":"Arrearage","message":"private-input"}}\n\n'
    seen=[]
    with pytest.raises(RuntimeError,match='Arrearage'):
        stream_http(Request('https://example.com/v1/chat/completions'),1,'compatible',seen.append,
                    opener=lambda *a,**kw:io.BytesIO(wire))
    assert seen==[]

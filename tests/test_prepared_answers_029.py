import hashlib,json
from zoom_copilot.prepared_answers import prepared_reply,question_match
from zoom_copilot.config import Preferences
from zoom_copilot.answers import AnswerService


QUESTION='How would you evaluate material identifiability under controlled illumination?'


def setup_cache(tmp_path,monkeypatch):
    monkeypatch.setenv('PRISM_INTERVIEW_HOME',str(tmp_path))
    source=tmp_path/'script.docx';source.write_bytes(b'controlled user script fixture')
    p=Preferences(candidate_profile='Confirmed personal profile',provider='codex',cloud_consent=False)
    row={'id':'Q1','question_en':QUESTION,'question_zh':'你会如何评估材料可辨识性？',
         'answer_en':'I would compare uncertainty under controlled illumination.',
         'answer_zh':'我会在受控照明下比较不确定性。','source':str(source),'location':'Q1',
         'raw_text':'Original user-script answer, retained for inspection.',
         'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    data={'version':1,'profile_sha256':hashlib.sha256(p.candidate_profile.encode()).hexdigest(),'items':[row]}
    (tmp_path/'prepared_answers.json').write_text(json.dumps(data),encoding='utf8')
    return p,source,data


def test_prepared_answer_is_source_checked_bilingual_and_calls_no_cloud(tmp_path,monkeypatch):
    p,source,data=setup_cache(tmp_path,monkeypatch)
    service=AnswerService(p)
    monkeypatch.setattr(service,'_get_codex',lambda *_:(_ for _ in ()).throw(AssertionError('No cloud allowed')))
    events=[]
    result=service.answer(QUESTION,[],on_delta=lambda *x:events.append(x))
    assert result.provider=='prepared-script' and result.answer_zh
    assert result.evidence[0]['source']==str(source) and result.evidence[0]['location']=='Q1'
    assert result.diagnostics['cloud_calls']==0
    assert events[0]==('question_zh',result.question_zh)
    assert [json.loads(text) for kind,text in events if kind=='answer_pair']==result.pairs


def test_gui_profile_whitespace_normalization_preserves_prepared_match(tmp_path,monkeypatch):
    p,_,_=setup_cache(tmp_path,monkeypatch)
    p.candidate_profile='  '+p.candidate_profile+'\n'
    assert prepared_reply(QUESTION,p) is not None


def test_changed_profile_source_or_disabled_preference_invalidates_prepared_answer(tmp_path,monkeypatch):
    p,source,data=setup_cache(tmp_path,monkeypatch)
    assert prepared_reply(QUESTION,p)
    p.prefer_prepared_answers=False
    assert prepared_reply(QUESTION,p) is None
    p.prefer_prepared_answers=True;p.candidate_profile+=' changed'
    assert prepared_reply(QUESTION,p) is None
    p.candidate_profile='Confirmed personal profile';source.write_bytes(b'new revision')
    assert prepared_reply(QUESTION,p) is None


def test_related_multipart_negative_number_and_perspective_questions_never_reuse_wrong_answer():
    assert question_match('So, '+QUESTION,QUESTION)==1
    assert not question_match(QUESTION+' How would you validate geometry?',QUESTION)
    assert not question_match(QUESTION.replace('would you','would you not'),QUESTION)
    assert not question_match(QUESTION.replace('controlled','uncontrolled'),QUESTION)
    assert not question_match(QUESTION.replace('you','we'),QUESTION)
    assert not question_match(QUESTION+' Use five measurements.',QUESTION)


def test_malformed_optional_cache_falls_back_instead_of_breaking_answers(tmp_path,monkeypatch):
    p,source,data=setup_cache(tmp_path,monkeypatch)
    data['items'][0].pop('raw_text')
    (tmp_path/'prepared_answers.json').write_text(json.dumps(data),encoding='utf8')
    assert prepared_reply(QUESTION,p) is None
    (tmp_path/'prepared_answers.json').write_text('[]',encoding='utf8')
    assert prepared_reply(QUESTION,p) is None

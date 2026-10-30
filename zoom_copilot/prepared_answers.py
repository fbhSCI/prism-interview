"""Reuse the user's own prepared script only for narrowly matching questions.

No inference, credentials or network access. Modified source/profile invalidates
the prepared translation; a merely related question always uses normal answers.
"""
from __future__ import annotations
from pathlib import Path
import difflib,hashlib,json,re,time


def _tokens(text):
    text=re.sub(r"^(?:(?:so|well|okay|ok|please)\b[, .:]*)+",'',text.strip(),flags=re.I)
    return re.findall(r"[\w]+(?:['’-][\w]+)*|[=<>±]",text.casefold())


def _anchors(tokens):
    return ([t for t in tokens if re.search(r'\d',t)],
            [t for t in tokens if t in {'not','no','never','without'} or t.endswith("n't")],
            [t for t in tokens if t in {'i','my','mine','we','our','ours','you','your','yours','their','they'}])


def question_match(question,prepared_question):
    left,right=_tokens(question),_tokens(prepared_question)
    if min(len(left),len(right))<5 or _anchors(left)!=_anchors(right):return 0.0
    ratio=difflib.SequenceMatcher(a=left,b=right,autojunk=False).ratio()
    coverage=len(set(left)&set(right))/max(1,len(set(left)|set(right)))
    return ratio if ratio>=.92 and coverage>=.85 else 0.0


def prepared_reply(question,preferences,*,path=None,request_id=''):
    if not getattr(preferences,'prefer_prepared_answers',True):return None
    from .config import app_dir
    started=time.perf_counter()
    source=Path(path) if path else app_dir()/'prepared_answers.json'
    try:
        if source.stat().st_size>1024*1024:return None
        data=json.loads(source.read_text(encoding='utf8'))
        if not isinstance(data,dict):return None
        # UI settings normalize surrounding whitespace; that must not invalidate
        # the same factual profile on the first save/start-listening action.
        profile=getattr(preferences,'candidate_profile','').strip()
        if data.get('version')!=1 or data.get('profile_sha256')!=hashlib.sha256(profile.encode()).hexdigest():return None
        items=data.get('items')
        if not isinstance(items,list):return None
        ranked=sorted(((question_match(question,str(row.get('question_en',''))),row)
                       for row in items[:32] if isinstance(row,dict)),key=lambda pair:pair[0],reverse=True)
        if not ranked or not ranked[0][0]:return None
        if len(ranked)>1 and ranked[0][0]-ranked[1][0]<.04:return None
        confidence,row=ranked[0]
        original=Path(row['source'])
        if not original.is_file() or hashlib.sha256(original.read_bytes()).hexdigest()!=row.get('source_sha256'):return None
        if any(not isinstance(row.get(key),str) or not row[key].strip() for key in ('answer_en','answer_zh','question_zh','location','id','raw_text')):return None
        if len(row['answer_en'])>5000 or len(row['answer_zh'])>6000:return None
    except (OSError,ValueError,TypeError,KeyError):return None
    from .answers import BilingualAnswer
    identity=row['id']
    return BilingualAnswer(row['question_zh'],row['answer_en'],row['answer_zh'],
        [f'{original.name} · {row["location"]}'],citations=[identity],
        evidence=[{'id':identity,'source':str(original),'location':row['location'],'text':row['raw_text']}],
        request_id=request_id,provider='prepared-script',model='讲稿预备回答',
        warnings=['高匹配复用讲稿原文；中文为预先准备的本地译文，请按当前问题核对。'],
        diagnostics={'prepared_script':True,'match_score':round(confidence,4),'cloud_calls':0,
                     'local_lookup_seconds':round(time.perf_counter()-started,4)})

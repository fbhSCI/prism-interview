"""Source/role boundaries must survive ASR pauses and imperfect voice clusters."""
import json

from zoom_copilot.interview_context import InterviewPreparation, question_parts
from zoom_copilot.question_packet import QuestionContext


def test_paused_multipart_question_keeps_all_acoustic_sources_and_sentence_checklist():
    context=QuestionContext()
    context.record('a','How would you model',speaker_id=1,speaker_label='发言者 1',turn_id='voice-a',accepted=True)
    context.record('b','scattering and wavelength-dependent attenuation?',speaker_id=None,
                   speaker_label='发言者待补认',turn_id='uncertain',accepted=True)
    context.record('c','Which assumptions fail in strong scattering?',speaker_id=3,
                   speaker_label='发言者 3',turn_id='changed-cluster',accepted=True)
    question='How would you model scattering and wavelength-dependent attenuation? Which assumptions fail in strong scattering?'
    packet=context.build(question,raw_question=question)
    assert [t.source_id for t in packet.turns]==['a','b','c']
    assert packet.raw_question_en==question
    assert len(packet.subquestions)==2
    assert '发言者待补认' in packet.display_text
    assert any('身份尚未确认' in warning for warning in packet.warnings)
    prompt=packet.to_prompt_data()
    assert [row['source_id'] for row in prompt['speaker_attributed_question_turns']]==['a','b','c']
    assert prompt['task'].startswith("Suggest the candidate's answer")


def test_candidate_probability_self_questions_never_become_interviewer_questions():
    context=QuestionContext()
    question='Which metrics have you used to assess model performance?'
    context.record('q',question,speaker_id=2,speaker_label='发言者 2')
    # The first answer is deliberately misclustered onto the same speaker.
    originals=["I'm not sure if I use any metric, but I think we can look at the probabilities.",
               "What's the confidence? What's the probability of him being a person?",
               "This model is predicting him as a person with 95% probability and 5% as other objects."]
    for index,text in enumerate(originals):
        context.record(f'a{index}',text,speaker_id=2 if index==0 else None,
                       speaker_label='匿名发言者',turn_id=f'answer{index}')
    packet=context.build(question)
    assert [turn.source_id for turn in packet.turns]==['q']
    assert packet.subquestions==(question,)
    assert packet.recent_context==()  # Later answers cannot leak into this request.
    raw=[turn.raw_en for turn in context._turns.values()]
    assert all(text in raw for text in originals)
    assert '95%' not in json.dumps(packet.to_prompt_data(),ensure_ascii=False)


def test_same_source_revision_replaces_instead_of_duplicates_and_keeps_raw():
    context=QuestionContext()
    context.record('card','How uh how do you calibrate',speaker_id=1,accepted=True)
    raw='How uh how do you calibrate the Stokes matrix, and which assumptions fail?'
    clean='How do you calibrate the Stokes matrix? Which assumptions fail?'
    context.record('card',raw,clean,speaker_id=1,accepted=True)
    packet=context.build(clean,raw_question=raw)
    assert len(packet.turns)==1
    assert packet.turns[0].raw_en==raw
    assert packet.turns[0].sentences==('How do you calibrate the Stokes matrix?', 'Which assumptions fail?')
    assert packet.to_ui_data()['turns'][0]['raw_en']==raw
    # Original grammar and scientific values remain in the immutable snapshot.
    context.record('card','A later unrelated revision',speaker_id=1,accepted=True)
    assert packet.turns[0].raw_en==raw


def test_old_question_and_later_candidate_speech_not_folded_into_new_question():
    context=QuestionContext()
    context.record('old-q','What is your optical system?',speaker_id=1,accepted=True)
    context.record('old-a','We use four linear analyzer channels.',speaker_id=2,accepted=False)
    context.begin_question()
    context.record('new-q','Can you retrieve S3?',speaker_id=1,accepted=True)
    context.record('new-a','Four linear channels do not demonstrate S3 retrieval.',speaker_id=2,accepted=False)
    packet=context.build('Can you retrieve S3?')
    assert [turn.source_id for turn in packet.turns]==['new-q']
    assert [turn.source_id for turn in packet.recent_context]==['old-q','old-a']
    assert all(turn.source_id!='new-a' for turn in packet.recent_context)


def test_manual_role_and_identity_override_survive_later_acoustic_revision():
    context=QuestionContext()
    text='Please explain the limitations.'
    context.record('q',text,speaker_id=None,turn_id='turn-q',accepted=True)
    context.update_speaker('turn-q','manual:prof','老师',manual=True)
    context.update_speaker('turn-q',8,'发言者 8')
    packet=context.build(text)
    assert packet.turns[0].label=='老师'
    assert packet.turns[0].identity_status=='manual'
    context.set_role('manual:prof','ignore')
    packet=context.build(text)
    assert not any(turn.label=='老师' for turn in packet.turns)
    context.set_role('manual:prof','interviewer')
    assert context.build(text).turns[0].label=='老师'


def test_manual_question_never_borrows_an_unrelated_acoustic_identity():
    context=QuestionContext()
    context.record('audio','What is the limitation?',speaker_id=1,speaker_label='发言者 1',accepted=True)
    packet=context.build('What is the limitation?',manual=True)
    assert packet.turns[0].speaker_id is None
    assert packet.turns[0].label=='当前问题（手动确认）'


def test_library_snapshot_is_navigation_only_and_immutable_not_all_papers():
    class KB:
        document_names=[r'C:\private\paper.pdf',r'D:\private\report.pptx']
        passages=['Secret full document content'];_revision=1
        def search(self,question,top_k=5):return []
    kb=KB();prep=InterviewPreparation();prep.prepare(kb)
    context=QuestionContext();packet=context.build('How does calibration work?',preparation=prep)
    assert packet.document_names==('paper.pdf','report.pptx')
    prompt=json.dumps(packet.to_prompt_data(),ensure_ascii=False)
    assert 'Secret full document content' not in prompt and 'private' not in prompt
    assert 'not model training' in prompt
    kb.document_names=['new.docx'];kb.passages=['new content'];kb._revision=2
    prep.search(kb,'calibration')
    assert prep.snapshot()['state']=='updating'
    prep.prepare(kb)
    assert prep.snapshot()['documents']==['new.docx']
    assert packet.document_names==('paper.pdf','report.pptx')


def test_prompt_budget_discards_redundant_text_without_chopping_complete_question():
    context=QuestionContext()
    fragments=[]
    for index in range(35):
        text=f'How would you validate condition {index} with 95% confidence and not claim 5% error?'
        fragments.append(text)
        context.record(str(index),text,speaker_id=index%3,speaker_label=f'发言者 {index%3}',accepted=True)
    question=' '.join(fragments)
    packet=context.build(question)
    encoded=json.dumps(packet.to_prompt_data(max_chars=2400),ensure_ascii=False,separators=(',',':'))
    assert len(encoded)<=2400
    assert packet.question_en==question and packet.raw_question_en==question
    assert packet.to_ui_data()['turns'][-1]['raw_en']==fragments[-1]
    assert json.loads(encoded)['complete_question_is_in_current_question_field'] is True


def test_lowercase_questions_decimals_abbreviations_and_compound_prompts():
    text='How would you calibrate at 0.95 confidence? which assumptions fail?'
    assert question_parts(text)==['How would you calibrate at 0.95 confidence?','which assumptions fail?']
    assert question_parts('How would you calibrate and how would you validate?')==[
        'How would you calibrate','how would you validate?']
    assert question_parts('')==[]


def test_ledger_is_bounded_without_losing_recent_question():
    context=QuestionContext(max_turns=12)
    for index in range(20):
        context.record(str(index),f'How would you validate stage {index}?',accepted=True)
    assert len(context._turns)==12
    assert context.build('How would you validate stage 19?').turns[0].source_id=='19'


def test_matching_protects_numeric_token_boundaries():
    context=QuestionContext()
    context.record('one','How would you validate condition 1?',speaker_id=1,accepted=True)
    context.record('nineteen','How would you validate condition 19?',speaker_id=2,accepted=True)
    packet=context.build('How would you validate condition 19?')
    assert [turn.source_id for turn in packet.turns]==['nineteen']


def test_role_filter_removes_exact_middle_source_instead_of_only_visual_hiding():
    context=QuestionContext()
    context.record('prefix','How do you model scattering?',speaker_id=1,accepted=True)
    context.record('self','I think we can estimate the confidence.',speaker_id=2,accepted=True)
    context.record('suffix','And how would you validate?',speaker_id=1,accepted=True)
    raw='How do you model scattering? I think we can estimate the confidence. And how would you validate?'
    context.set_role(2,'candidate')
    filtered=context.filter_question(raw)
    assert filtered=='How do you model scattering? And how would you validate?'
    packet=context.build(filtered,raw_question=filtered)
    assert [turn.source_id for turn in packet.turns]==['prefix','suffix']
    assert not context.last_filter_warnings
    assert all(turn.role_source=='inferred_from_content' for turn in packet.turns)


def test_role_filter_retains_repeated_or_short_source_and_marks_ambiguity():
    context=QuestionContext()
    context.record('duplicate','Which method works?',speaker_id=1,accepted=True)
    context.record('short','No.',speaker_id=2,accepted=True)
    context.set_role(1,'ignore');context.set_role(2,'candidate')
    raw='Which method works? Which method works? No. Can you explain?'
    assert context.filter_question(raw)==raw
    assert context.last_filter_warnings
    assert context.build(raw).warnings
    # Excluding a source which IS the entire question is safe, even if short.
    assert context.filter_question('No.')==''


def test_role_filter_does_not_delete_altered_or_inferred_candidate_sources():
    context=QuestionContext()
    context.record('answer','I think the model is useful.',speaker_id=2,accepted=False)
    raw='I think the model is useful. What is the limitation?'
    assert context.filter_question(raw)==raw
    context.record('q','Can you explain condition 1?',speaker_id=3,accepted=True)
    context.set_role(3,'candidate')
    revised='Can you explain condition 19?'
    assert context.filter_question(revised)==revised
    assert context.last_filter_warnings


def test_probability_context_remains_candidate_speech_when_followup_asks_about_it():
    context=QuestionContext()
    context.record('a',"I'm not sure if I use any metric, but I think we can look at the probabilities.",
                   speaker_id=2,accepted=False)
    context.record('a2',"What's the probability of him being a person? This model predicts 95% for a person and 5% for other objects.",
                   speaker_id=2,accepted=False)
    context.begin_question()
    context.record('q','How would you calibrate that confidence?',speaker_id=1,accepted=True)
    packet=context.build('How would you calibrate that confidence?')
    assert packet.subquestions==('How would you calibrate that confidence?',)
    context_data=packet.to_prompt_data()['recent_speech_for_followup_resolution_only']
    assert [row['role'] for row in context_data]==['candidate','candidate']
    assert '95%' in context_data[-1]['utterance'] and '5%' in context_data[-1]['utterance']

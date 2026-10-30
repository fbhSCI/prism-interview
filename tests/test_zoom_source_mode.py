"""An explicitly selected output source is not an acoustic speaker identity."""
from dataclasses import replace
from types import SimpleNamespace

from test_controller_v2 import controller
from zoom_copilot.config import Preferences
from zoom_copilot.question_packet import QuestionContext


def source_mode(c):
    c.pref.caption_speaker_mode = "single_remote"
    c._listen_pref = replace(c.pref)
    c.listening = True


def acoustic_context(c, number, start, end):
    return dict(phase_epoch=c._phase_epoch, word_timed=True, stream_id="zoom-loopback",
                start_sample=int(start*16000), end_sample=int(end*16000),
                speaker_id="speaker_1" if number % 2 else None,
                speaker_label="说话人 1" if number % 2 else "说话人未确认",
                speaker_status="confirmed" if number % 2 else "unresolved",
                turn_id=f"voice-turn-{number}")


def test_source_mode_is_default_and_persists_without_identity_claim(tmp_path, monkeypatch):
    monkeypatch.setenv("PRISM_INTERVIEW_HOME", str(tmp_path))
    assert Preferences().caption_speaker_mode == "single_remote"
    assert Preferences(caption_speaker_mode="unknown").caption_speaker_mode == "single_remote"
    Preferences(caption_speaker_mode="acoustic").save()
    assert Preferences.load().caption_speaker_mode == "acoustic"


def test_source_label_is_not_reported_as_failed_voice_identification():
    q=QuestionContext()
    text='How would you reconstruct the target?'
    q.record('remote',text,speaker_id=None,speaker_label='Zoom远端（单一远端模式）',
             speaker_status='source',accepted=True)
    packet=q.build(text)
    assert packet.turns[0].identity_status=='source'
    assert not any('身份尚未确认' in warning for warning in packet.warnings)
    assert packet.turns[0].speaker_id is None


def test_single_remote_skips_acoustic_model_even_when_diarization_enabled(controller):
    c = controller
    source_mode(c)
    assert c.pref.speaker_diarization
    assert c._make_diarizer(c.pref.model_cache_dir) is None
    # An option changed while recording only applies after a capture restart.
    c.pref.caption_speaker_mode = "acoustic"
    assert c._single_remote_source()


def test_zoom_scientific_fragments_translate_whole_source_and_retain_raw(controller):
    c = controller
    source_mode(c)
    calls = []
    c.translation = SimpleNamespace(submit=lambda key, text: calls.append((key,text)), stop=lambda:None)
    # Recorded round-1 text, including its ASR error, remains unchanged. The
    # source grouping must never silently fix "targeted" to an invented word.
    pieces = ["So you will explain how your imaging system", "reconstructs a", "targeted from",
              "intensity measurements.", "if the scattering level increases how would", "you separate",
              "material reflectance from", "illumination", "I do not expect real-time",
              "reconstruction and acquisition time", "below 0.5 seconds.", "seconds would be sufficient"]
    for number, text in enumerate(pieces, 1):
        c.on_transcript("final", text, acoustic_context(c,number,number*3,number*3+2))
        # This used to seal the display as soon as the 2.2s question timer was
        # ready, including in presentation, creating a card per short pause.
        c.assembler.last_text -= 3
        c.assembler.last_voice -= 3
        c.auto_answer()
    expected = " ".join(pieces)
    # Readable translation paragraphs are separate stable entry IDs within the
    # same explicitly selected source/speaker card. No raw word disappears.
    assert 1 <= len(c._transcripts) <= 3
    assert ' '.join(row['raw_en'] for row in c._transcripts) == expected
    for row in c._transcripts:
        assert c._caption_raw[row['id']] == row['raw_en']
        assert row["speaker_id"] is None and row["speaker_status"] == "source"
        assert row["speaker_label"] == "对方 · 会议声音"
        assert row["turn_id"] == f"zoom-source-{c._session}-{c._phase_epoch}"
    # The real period after "0.5 seconds." may end a paragraph; its negation
    # and numeric condition must stay together in that paragraph's final input.
    condition = next(row for row in c._transcripts if 'I do not expect' in row['raw_en'])
    final_source = c._caption_sources[condition['id']]
    assert 'I do not expect' in final_source and '0.5' in final_source
    assert ((condition['id'], final_source), final_source) in calls
    assert any('intensity measurements' in text for _, text in calls)
    row = c._transcripts[-1]
    assert calls[-1][0] == (row["id"], c._caption_sources[row["id"]])
    assert not c.service.calls and not c.busy


def test_single_source_silence_boundary_uses_audio_clock_not_worker_wait(controller, monkeypatch):
    c = controller
    source_mode(c)
    tick = [100.0]
    monkeypatch.setattr("zoom_copilot.app.time.monotonic", lambda:tick[0])
    c.on_transcript("final", "The same remote person is speaking", acoustic_context(c,1,0,2))
    tick[0] += 40  # A delayed decoder is not 40 seconds of source silence.
    c.on_transcript("final", "about the calibrated system.", acoustic_context(c,2,14,16))
    assert len(c._transcripts) == 1  # Exactly 12 source seconds still continues.
    c.on_transcript("final", "A new paragraph after a long pause.", acoustic_context(c,3,29,32))
    assert len(c._transcripts) == 2
    assert c._transcripts[0]["turn_id"] == c._transcripts[1]["turn_id"]


def test_acoustic_revisions_cannot_promote_single_source_to_confirmed_identity(controller):
    c = controller
    source_mode(c)
    meta = acoustic_context(c,1,0,2)
    meta["speaker_revisions"] = [dict(turn_id="voice-turn-1",speaker_id="speaker_8",speaker_label="Speaker 8")]
    c.on_transcript("final", "How would you separate reflectance from illumination?", meta)
    assert c._transcripts[0]["speaker_id"] is None
    assert c._transcripts[0]["speaker_status"] == "source"
    c.on_transcript("partial", "And estimate the uncertainty", acoustic_context(c,2,2,3))
    assert c._live_data["speaker_status"] == "source"
    assert c._live_data["speaker_label"] == "对方 · 会议声音"


def test_single_source_rejected_fragment_still_blocks_automatic_answer(controller):
    c = controller
    source_mode(c)
    c.pref.auto_answer = True
    c.pref.interview_phase = "qa"
    c.on_transcript("final", "How do you calibrate", acoustic_context(c,1,0,2))
    c.on_transcript("rejected", "", acoustic_context(c,2,2,4))
    c.on_transcript("final", "the optical system?", acoustic_context(c,3,4,6))
    c.assembler.last_text -= 3
    c.assembler.last_voice -= 3
    c.auto_answer()
    assert c._question_uncertain
    assert not c.busy and not c.service.calls

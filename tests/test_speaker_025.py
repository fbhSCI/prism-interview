"""Conservative deferred speaker decisions; no names inferred from text."""
import numpy as np

from zoom_copilot.speaker import OnlineSpeakerDiarizer, RATE


def annotate(diarizer, seconds, origin=0, value=.1, text="Original words.", stream="s"):
    size = int(seconds * RATE)
    return diarizer.annotate(np.full(size, value, dtype=np.float32),
        [{"text": text, "start": origin, "end": origin + size}],
        {"start_sample": origin, "commit_start_sample": origin,
         "commit_end_sample": origin + size, "stream_id": stream}, text)


def test_initial_short_unknown_is_revisited_after_same_voice_establishes_profile(tmp_path):
    diarizer = OnlineSpeakerDiarizer(tmp_path, extractor=lambda audio: [1., 0.])
    short = annotate(diarizer, .9, text="Got it.")[0]
    assert short["speaker_id"] is None
    assert short["speaker_status"] == "pending"
    assert not diarizer.profiles
    later = annotate(diarizer, 2.5, origin=2*RATE)[0]
    revision = later["speaker_revisions"][0]
    assert revision["turn_id"] == short["turn_id"]
    assert revision["speaker_id"] == later["speaker_id"] == "speaker_1"
    assert revision["speaker_status"] == "confirmed"
    assert not diarizer._pending


def test_different_or_ambiguous_acoustics_cannot_force_short_turn_into_neighbour(tmp_path):
    diarizer = OnlineSpeakerDiarizer(tmp_path,
        extractor=lambda audio: [1., 0.] if audio.mean() < .15 else [0., 1.])
    unknown = annotate(diarizer, .9, value=.2)[0]
    later = annotate(diarizer, 2.5, origin=2*RATE, value=.1)[0]
    assert unknown["speaker_status"] == "pending"
    assert later["speaker_id"] == "speaker_1"
    assert "speaker_revisions" not in later
    final = annotate(diarizer, 2.5, origin=34*RATE, value=.1)[0]
    assert final["speaker_revisions"][0]["speaker_id"] is None
    assert final["speaker_revisions"][0]["speaker_status"] == "unresolved"


def test_short_feedback_uses_its_own_audio_crop_and_can_be_resolved_later(tmp_path):
    diarizer = OnlineSpeakerDiarizer(tmp_path,
        extractor=lambda audio: [1., 0.] if audio.mean() < .15 else [0., 1.])
    audio = np.r_[np.full(3*RATE, .1), np.full(int(.7*RATE), .2)].astype(np.float32)
    result = diarizer.annotate(audio,
        [{"text": "A long explanation.", "start": 0, "end": 3*RATE},
         {"text": " Got it.", "start": 3*RATE, "end": len(audio)}],
        {"start_sample": 0, "commit_end_sample": len(audio), "stream_id": "s"},
        "A long explanation. Got it.")
    assert len(result) == 2
    assert result[0]["speaker_id"] == "speaker_1"
    assert result[1]["speaker_id"] is None
    assert result[1]["text"] == "Got it."
    later = annotate(diarizer, 2.5, origin=5*RATE, value=.2)[0]
    assert later["speaker_id"] == "speaker_2"
    assert later["speaker_revisions"][0]["turn_id"] == result[1]["turn_id"]
    assert later["speaker_revisions"][0]["speaker_id"] == "speaker_2"


def test_human_label_is_protected_without_learning_its_audio(tmp_path):
    diarizer = OnlineSpeakerDiarizer(tmp_path, extractor=lambda audio: [1., 0.])
    unknown = annotate(diarizer, .9)[0]
    diarizer.assign_turn(unknown["turn_id"], "manual:面试官")
    later = annotate(diarizer, 2.5, origin=2*RATE)[0]
    assert "speaker_revisions" not in later
    assert len(diarizer.profiles) == 1


def test_end_of_capture_finalizes_unknown_without_inventing_identity(tmp_path):
    diarizer = OnlineSpeakerDiarizer(tmp_path, extractor=lambda audio: [1., 0.])
    unknown = annotate(diarizer, .9)[0]
    revisions = diarizer.finalize_pending()
    assert len(revisions) == 1
    assert revisions[0]["turn_id"] == unknown["turn_id"]
    assert revisions[0]["speaker_id"] is None
    assert revisions[0]["speaker_status"] == "unresolved"
    assert diarizer.finalize_pending() == []

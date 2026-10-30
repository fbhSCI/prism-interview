"""Anonymous, local, conservative online speaker grouping.

TitaNet acoustic embeddings identify *similar voices*, never a person's name.
Sliding 2.4-second windows can miss very short turns and cannot separate overlap.
Ambiguous matches stay unconfirmed. Neither audio nor voiceprints are persisted.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import time
import urllib.request

import numpy as np

MODEL_NAME = "nemo_en_titanet_large.onnx"
MODEL_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/"
             "speaker-recongition-models/" + MODEL_NAME)
MODEL_SHA256 = "d51abcf31717ef28162f26acb9d44dd4127c3d44c9b8624f699f3425daca8e77"
MODEL_LICENSE = "CC-BY-4.0"
RATE = 16000


def prepare_speaker_model(model_dir, on_status=print):
    """Explicit setup operation: download a pinned public model, not speech."""
    target = Path(model_dir) / "speaker" / MODEL_NAME
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and hashlib.sha256(target.read_bytes()).hexdigest() == MODEL_SHA256:
        return target
    temporary = target.with_suffix(".download")
    on_status("下载本地说话人模型 NVIDIA TitaNet-Large（约 102 MB）…")
    try:
        urllib.request.urlretrieve(MODEL_URL, temporary)
        if hashlib.sha256(temporary.read_bytes()).hexdigest() != MODEL_SHA256:
            raise RuntimeError("说话人模型校验失败；未启用该模型。")
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


@dataclass
class _Profile:
    identifier: str
    vector: np.ndarray
    observations: int = 1


class OnlineSpeakerDiarizer:
    """Single-worker online clustering; call ``prepare`` before capture.

    ``annotate`` returns text groups and context fields for the existing ASR
    callback. Word times are source-clock samples, so overlap never re-emits text.
    The confidence field is a cosine similarity, not a probability/accuracy claim.
    """

    def __init__(self, model_dir, *, extractor=None, window_seconds=2.4,
                 hop_seconds=.8, match_threshold=.45, new_threshold=.25,
                 margin=.10, maximum_speakers=8, turn_gap_seconds=12.0):
        self.model_path = Path(model_dir) / "speaker" / MODEL_NAME
        self.extractor = extractor
        self.window = int(window_seconds * RATE)
        self.hop = int(hop_seconds * RATE)
        self.match_threshold = match_threshold
        self.new_threshold = new_threshold
        self.margin = margin
        self.maximum_speakers = maximum_speakers
        self.turn_gap = int(turn_gap_seconds * RATE)
        self.profiles: list[_Profile] = []
        self.ready = extractor is not None
        self.failure = ""
        self.last_metrics = {}
        self._turn_number = 0
        self._last_identity = object()
        self._last_end = -self.turn_gap - 1
        self._turn_id = ""
        self._stream_id = None
        # Ephemeral voice vectors, never raw audio or text. Later speech may
        # establish a profile for an initially too-short utterance.
        self._pending: list[dict] = []
        self.pending_lifetime = int(30 * RATE)
        self.maximum_pending = 64

    def prepare(self):
        if self.ready:
            return True
        if not self.model_path.is_file():
            self.failure = "未安装本地说话人模型；文字保留为说话人未确认。"
            return False
        if hashlib.sha256(self.model_path.read_bytes()).hexdigest() != MODEL_SHA256:
            self.failure = "本地说话人模型校验失败；文字保留为说话人未确认。"
            return False
        import sherpa_onnx
        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(self.model_path), num_threads=2, debug=False, provider="cpu")
        self.extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self.ready = True
        return True

    @staticmethod
    def _normalized(vector):
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(vector)
        if not vector.size or not np.all(np.isfinite(vector)) or norm < 1e-8:
            return None
        return vector / norm

    def _embedding(self, audio):
        if callable(self.extractor):
            return self._normalized(self.extractor(audio))
        stream = self.extractor.create_stream()
        stream.accept_waveform(RATE, np.asarray(audio, dtype=np.float32))
        stream.input_finished()
        if not self.extractor.is_ready(stream):
            return None
        return self._normalized(self.extractor.compute(stream))

    def _identify(self, vector, *, learn=True, enough_audio=True):
        if vector is None:
            return None, 0.0
        scores = [(float(vector @ p.vector), p) for p in self.profiles]
        scores.sort(key=lambda value: value[0], reverse=True)
        best = scores[0][0] if scores else 0.0
        runner_up = scores[1][0] if len(scores) > 1 else -1.0
        # Short responses may match a well established voice, but cannot create
        # a new identity. This avoids treating every "yes" as a new person.
        threshold = self.match_threshold if enough_audio else max(.58, self.match_threshold)
        if scores and best >= threshold and best - runner_up >= self.margin:
            profile = scores[0][1]
            if learn and enough_audio and best >= self.match_threshold + .10:
                profile.vector = self._normalized(.9 * profile.vector + .1 * vector)
                profile.observations += 1
            return profile.identifier, best
        if learn and enough_audio and (not scores or best < self.new_threshold) and len(self.profiles) < self.maximum_speakers:
            profile = _Profile(f"speaker_{len(self.profiles)+1}", vector)
            self.profiles.append(profile)
            return profile.identifier, 1.0
        return None, best

    def _context(self, identity, score, start, end, stream_id, *, status=None, force_new=False):
        # Unknown audio does not assert that two people are the same person.
        if (force_new or stream_id != self._stream_id or identity != self._last_identity
                or start - self._last_end > self.turn_gap):
            self._turn_number += 1
            self._turn_id = f"voice-turn-{self._turn_number}"
        self._stream_id = stream_id
        self._last_identity = identity
        self._last_end = max(self._last_end, end)
        status = status or ("confirmed" if identity else "unresolved")
        label = "说话人 " + identity.split("_")[-1] if identity else (
            "说话人待补认" if status == "pending" else "说话人未确认")
        return {"speaker_id": identity, "speaker_label": label,
                "speaker_status": status,
                "speaker_confidence": round(float(score), 3),
                "speaker_score_type": "cosine_similarity",
                "turn_id": self._turn_id,
                "speaker_start_sample": int(start), "speaker_end_sample": int(end)}

    def _resolve_pending(self, current_end, stream_id):
        """Revisit short unknown turns using newly established acoustic profiles."""
        revisions, retained = [], []
        for pending in self._pending:
            identity, score = self._identify(pending["vector"], learn=False, enough_audio=False)
            expired = (pending["stream_id"] != stream_id or
                current_end - pending["end"] > self.pending_lifetime)
            if identity is not None and pending["stream_id"] == stream_id:
                revisions.append({"turn_id": pending["turn_id"], "speaker_id": identity,
                    "speaker_label": "说话人 " + identity.split("_")[-1],
                    "speaker_confidence": round(score, 3), "speaker_status": "confirmed",
                    "speaker_score_type": "cosine_similarity"})
            elif expired:
                revisions.append({"turn_id": pending["turn_id"], "speaker_id": None,
                    "speaker_label": "说话人未确认", "speaker_confidence": round(score, 3),
                    "speaker_status": "unresolved", "speaker_score_type": "cosine_similarity"})
            else:
                retained.append(pending)
        self._pending = retained
        return revisions

    def assign_turn(self, turn_id, speaker_id=None):
        """Protect a human assignment from a later automatic pending revision.

        The caller must serialize this with ``annotate``. A manual label never
        becomes an acoustic training example.
        """
        self._pending = [item for item in self._pending if item["turn_id"] != str(turn_id)]

    def finalize_pending(self):
        """Drain on the speech worker after its final audio, never on the UI thread."""
        latest = max((pending["end"] for pending in self._pending), default=self._last_end)
        return self._resolve_pending(latest + self.pending_lifetime + 1, self._stream_id)

    def _short_phrase_matches(self, audio, words, origin, *, allow_trailing=True,
                              acoustic_words=()):
        """Refine a short feedback phrase without embedding its neighbour's voice.

        ASR punctuation/silence only chooses the acoustic crop; it never chooses
        the speaker. A short crop may match an established voice, never add one.
        """
        spans, start = [], 0
        for index, word in enumerate(words):
            ended = str(word["text"]).rstrip().endswith((".", "?", "!"))
            if ended or index == len(words)-1:
                if ended or allow_trailing:
                    spans.append((start, index + 1))
                start = index + 1
        matched = {}
        for first, last in spans:
            begin = max(0, int(words[first]["start"]) - origin)
            end = min(audio.size, int(words[last-1]["end"]) - origin)
            if not int(.4 * RATE) <= end - begin < int(2 * RATE):
                continue
            # A hard commit can place only the final word of an uninterrupted
            # phrase in this interval. Keep its already decoded lookback audio;
            # never cross a sentence end or an audible gap to borrow a neighbour.
            # Words select an acoustic crop only; the embedding still has to
            # match an established profile at the same conservative threshold.
            previous = [w for w in acoustic_words if int(w["end"]) <= origin + begin]
            next_start = origin + begin
            for word in reversed(previous):
                if (str(word["text"]).rstrip().endswith((".", "?", "!"))
                        or next_start - int(word["end"]) > int(.3 * RATE)
                        or origin + end - int(word["start"]) > self.window):
                    break
                begin = max(0, int(word["start"]) - origin)
                next_start = int(word["start"])
            clip = audio[begin:end]
            if not clip.size or float(np.sqrt(np.mean(clip * clip))) < .003:
                continue
            vector = self._embedding(clip)
            identity, score = self._identify(vector, learn=False, enough_audio=False)
            # Even a currently unknown short phrase retains its own embedding
            # for later resolution, instead of taking the adjacent long window.
            for index in range(first, last):
                matched[index] = (identity, score, vector)
        return matched

    def annotate(self, audio, words, metadata, transcript):
        """Group every accepted word exactly once; unknown on any uncertainty.

        ``words`` entries contain ``text``, ``start`` and ``end`` (absolute
        samples). The caller has already removed ASR overlap duplicates.
        """
        started = time.perf_counter()
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        origin = int(metadata.get("start_sample", 0))
        lower = int(metadata.get("commit_start_sample", origin))
        upper = int(metadata.get("commit_end_sample", origin + audio.size))
        stream_id = metadata.get("stream_id", "default")
        if not self.ready or not words:
            context = self._context(None, 0, lower, upper, stream_id,
                status="unavailable" if not self.ready else "unresolved", force_new=True)
            self.last_metrics = {"speaker_seconds": round(time.perf_counter()-started, 3),
                                 "speaker_windows": 0, "speaker_profiles": len(self.profiles)}
            return [{"text": transcript, **context}]

        acoustic_words = metadata.get("acoustic_words") or words
        first = max(0, min(int(w["start"]) for w in acoustic_words) - origin)
        last = min(audio.size, max(int(w["end"]) for w in acoustic_words) - origin)
        # Clip windows to the speech span instead of embedding long surrounding
        # silence. This is acoustic grouping, not names inferred from text.
        duration = last - first
        starts = list(range(first, max(first + 1, last - self.window + 1), self.hop))
        final_start = max(first, last - self.window)
        if not starts or starts[-1] != final_start:
            starts.append(final_start)
        windows = []
        for begin in starts:
            end = min(last, begin + self.window)
            clip = audio[begin:end]
            center = origin + (begin + end) / 2
            enough = clip.size >= int(2.0 * RATE)
            if clip.size < int(.65 * RATE) or float(np.sqrt(np.mean(clip * clip))) < .003:
                identity, score = None, 0.0
            else:
                vector = self._embedding(clip)
                identity, score = self._identify(vector,
                    learn=lower <= center < upper, enough_audio=enough)
            if clip.size < int(.65 * RATE) or float(np.sqrt(np.mean(clip * clip))) < .003:
                vector = None
            windows.append((center, identity, score, vector))

        groups = []
        short_matches = self._short_phrase_matches(audio, words, origin,
            allow_trailing=not bool(metadata.get("hard_cut")), acoustic_words=acoustic_words)
        for index, word in enumerate(words):
            middle = (int(word["start"]) + int(word["end"])) / 2
            _, identity, score, vector = min(windows, key=lambda item: abs(item[0] - middle))
            if index in short_matches:
                identity, score, vector = short_matches[index]
            if not groups or groups[-1]["identity"] != identity:
                groups.append({"identity": identity, "score": score, "vector": vector, "words": [word]})
            else:
                groups[-1]["words"].append(word)
                groups[-1]["score"] = min(groups[-1]["score"], score)
        result = []
        revisions = self._resolve_pending(upper, stream_id)
        for group in groups:
            group_words = group["words"]
            if group["identity"] is None:
                # Sliding windows can straddle a voice change. Once word groups
                # are available, re-embed just the unknown words instead of
                # asking a mixed-neighbour vector to represent a short reply.
                begin = max(0, int(group_words[0]["start"]) - origin)
                end = min(audio.size, int(group_words[-1]["end"]) - origin)
                clip = audio[begin:end]
                if clip.size >= int(.4 * RATE) and float(np.sqrt(np.mean(clip * clip))) >= .003:
                    vector = self._embedding(clip)
                    identity, score = self._identify(vector, learn=False, enough_audio=False)
                    if vector is not None:
                        group["vector"], group["identity"], group["score"] = vector, identity, score
            context = self._context(group["identity"], group["score"],
                int(group_words[0]["start"]), int(group_words[-1]["end"]), stream_id,
                status="pending" if group["identity"] is None and group["vector"] is not None else None,
                force_new=group["identity"] is None)
            if group["identity"] is None and group["vector"] is not None:
                self._pending.append({"turn_id": context["turn_id"], "vector": group["vector"],
                    "end": int(group_words[-1]["end"]), "stream_id": stream_id})
                if len(self._pending) > self.maximum_pending:
                    dropped = self._pending.pop(0)
                    revisions.append({"turn_id": dropped["turn_id"], "speaker_id": None,
                        "speaker_label": "说话人未确认", "speaker_confidence": 0,
                        "speaker_status": "unresolved"})
            if revisions:
                context["speaker_revisions"] = revisions
            result.append({"text": "".join(w["text"] for w in group_words).strip(), **context})
        self.last_metrics = {"speaker_seconds": round(time.perf_counter()-started, 3),
                             "speaker_windows": len(windows), "speaker_profiles": len(self.profiles)}
        return result

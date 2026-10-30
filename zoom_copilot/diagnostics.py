"""Explicit offline packaged-runtime verification on a supplied test WAV.

No microphone capture, cloud request, model download or private corpus access.
The chosen WAV's transcript is written only to the requested verification report.
"""
from __future__ import annotations

import json
from pathlib import Path
import time
import traceback


def verify_runtime(wav: str, output: str, translation_engine: str = "argos", verify_speaker: bool = False,
                   verify_live: bool = False) -> int:
    from .runtime import configure_cuda_dlls, default_model_cache
    report = {"passed": False, "kind": "explicit-offline-runtime-verification",
              "cloud_requests": 0, "capture_started": False}
    try:
        configure_cuda_dlls()
        import ctranslate2
        import numpy as np
        from faster_whisper import WhisperModel
        from .audio import StreamingVoiceDetector
        from .speech import DEFAULT_GLOSSARY
        from .translation import ArgosTranslator
        from pypdf import PdfReader, PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
        from io import BytesIO

        detector = StreamingVoiceDetector()
        report["vad_backend"] = "silero-onnx"
        report["vad_silence_rejected"] = not detector(np.zeros(16000, dtype=np.float32))
        if not report["vad_silence_rejected"]:
            raise RuntimeError("The bundled voice detector did not reject the supplied silence.")

        model_path = default_model_cache() / "asr" / "large-v3-turbo"
        if not (model_path / "model.bin").is_file():
            raise RuntimeError("Offline verification requires an existing large-v3-turbo model.")
        if not Path(wav).is_file():
            raise FileNotFoundError("The explicit test WAV was not found.")
        device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
        started = time.perf_counter()
        model = WhisperModel(str(model_path), device=device,
                             compute_type="int8_float16" if device == "cuda" else "int8",
                             local_files_only=True)
        report["device"] = device
        report["model_load_seconds"] = round(time.perf_counter() - started, 3)
        started = time.perf_counter()
        # Decode from the path, exercising the bundled PyAV DLLs as well as CUDA.
        segments, _ = model.transcribe(wav, language="en", beam_size=3,
                                      initial_prompt=DEFAULT_GLOSSARY,
                                      vad_filter=True, word_timestamps=True,
                                      condition_on_previous_text=False)
        decoded = list(segments)
        english = " ".join(segment.text.strip() for segment in decoded)
        report["asr_seconds"] = round(time.perf_counter() - started, 3)
        if not english:
            raise RuntimeError("The supplied WAV produced no recognized English.")
        started = time.perf_counter()
        if translation_engine == "gemma":
            from .translation_gemma import GemmaTranslator
            translator = GemmaTranslator(default_model_cache())
            if not translator.ready:
                raise RuntimeError("Offline verification requires an already running Ollama with TranslateGemma installed.")
        else:
            translator = ArgosTranslator()
        from .translation_protection import protect_translation_source
        protected=protect_translation_source(english)
        chinese = protected.restore(translator.translate(protected.text))
        report["translation_engine"] = translation_engine
        report["translation_seconds"] = round(time.perf_counter() - started, 3)
        if not chinese:
            raise RuntimeError("Offline translation returned empty text.")
        report["english"] = english
        report["chinese"] = chinese
        if verify_live:
            from .streaming_asr import model_directory,create_recognizer
            from .transcript_text import TranscriptRestorer
            from faster_whisper.audio import decode_audio
            directory=model_directory(default_model_cache())
            if not (directory/'model.int8.onnx').is_file():
                raise RuntimeError('Offline verification requires an existing FastConformer model.')
            recognizer=create_recognizer(directory)
            stream=recognizer.create_stream()
            samples=decode_audio(wav,sampling_rate=16000)
            samples=np.concatenate((samples,np.zeros(16000,dtype=np.float32)))
            for index in range(0,len(samples),1600):
                stream.accept_waveform(16000,samples[index:index+1600])
                while recognizer.is_ready(stream):recognizer.decode_stream(stream)
            draft=recognizer.get_result(stream)
            if not str(draft).strip():raise RuntimeError('Frozen streaming decoder returned empty text.')
            restorer=TranscriptRestorer(default_model_cache())
            if not restorer.prepare():raise RuntimeError('Frozen punctuation model not ready.')
            restored=restorer.restore(english)
            report.update(streaming_runtime_ready=True,streaming_model=directory.name,
                          punctuation_runtime_ready=True,punctuation_engine=restored.engine,
                          punctuation_seconds=round(restored.elapsed_seconds,4))
        if verify_speaker:
            from .speaker import OnlineSpeakerDiarizer
            from faster_whisper.audio import decode_audio
            diarizer = OnlineSpeakerDiarizer(default_model_cache())
            if not diarizer.prepare():
                raise RuntimeError(diarizer.failure)
            words = [{"text": word.word, "start": int(word.start * 16000), "end": int(word.end * 16000)}
                     for part in decoded for word in (part.words or [])]
            groups = diarizer.annotate(decode_audio(wav, sampling_rate=16000), words, {}, english)
            report["speaker_runtime_ready"] = True
            report["speaker_groups"] = len(groups)
            report["speaker_model"] = diarizer.model_path.name
            report["speaker_metrics"] = dict(diarizer.last_metrics)
            if not groups or "text" not in groups[0]:
                raise RuntimeError("Frozen speaker runtime returned no usable text groups.")
            if not diarizer.last_metrics.get("speaker_windows"):
                raise RuntimeError("Frozen speaker runtime did not process an acoustic embedding window.")
        # Confirm PDF import/decoding works in the frozen distribution as well.
        pdf = PdfWriter()
        page = pdf.add_blank_page(width=612, height=792)
        font = DictionaryObject({NameObject("/Type"): NameObject("/Font"),
                                 NameObject("/Subtype"): NameObject("/Type1"),
                                 NameObject("/BaseFont"): NameObject("/Helvetica")})
        page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"):
            DictionaryObject({NameObject("/F1"): pdf._add_object(font)})})
        content = DecodedStreamObject()
        content.set_data(b"BT /F1 12 Tf 72 720 Td (Prism runtime PDF verification) Tj ET")
        page[NameObject("/Contents")] = pdf._add_object(content)
        buffer = BytesIO()
        pdf.write(buffer)
        report["pdf_roundtrip"] = "Prism runtime PDF verification" in PdfReader(buffer).pages[0].extract_text()
        report["passed"] = bool(report["pdf_roundtrip"])
    except Exception:
        report["error"] = traceback.format_exc()
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if report["passed"] else 1

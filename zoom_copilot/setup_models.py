"""Download public local model weights, never interview data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil


def prepare(model_dir, asr="large-v3-turbo", translation=True, skip_asr=False,
            translation_engine="argos", on_status=print, speaker=False, streaming=False,
            punctuation=False, streaming_backend="fastconformer"):
    root=Path(model_dir).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    if not skip_asr:
        from faster_whisper.utils import _MODELS
        from huggingface_hub import snapshot_download
        repository=_MODELS.get(asr)
        if not repository:
            raise ValueError("Unsupported ASR model: "+asr)
        target=root/"asr"/asr
        needed=4*1024**3 if "large" in asr else 2*1024**3
        if not (target/"model.bin").is_file() and shutil.disk_usage(root).free<needed:
            raise RuntimeError("模型目录可用空间不足，请在其他磁盘选择至少4GB空闲目录。")
        on_status("Downloading public ASR weights: "+repository)
        snapshot_download(repository,local_dir=str(target),token=False,
            allow_patterns=["config.json","preprocessor_config.json","model.bin","tokenizer.json","vocabulary.*"],
            max_workers=3)
        if not (target/"model.bin").is_file():
            raise RuntimeError("ASR download did not produce model.bin")
    if translation:
        if translation_engine=="nllb":
            from .translation_nllb import NllbTranslator
            translator=NllbTranslator(root/"translation-nllb")
            if not translator.ready:
                translator.install(on_status=on_status)
        else:
            from .translation import ArgosTranslator
            translator=ArgosTranslator(root/"translation")
            if not translator.ready:
                translator.install_english_chinese(on_status)
        translator.translate("Could you explain your method?")
        on_status("Offline English-to-Chinese translation validated.")
    speaker_path = None
    if speaker:
        from .speaker import prepare_speaker_model, OnlineSpeakerDiarizer
        speaker_path = prepare_speaker_model(root, on_status)
        diarizer = OnlineSpeakerDiarizer(root)
        if not diarizer.prepare():
            raise RuntimeError(diarizer.failure)
        on_status("Local speaker embedding model validated.")
    streaming_path = None
    if streaming:
        from .streaming_asr import prepare_streaming_model, create_recognizer
        streaming_path = prepare_streaming_model(root, on_status, streaming_backend)
        create_recognizer(streaming_path)
        on_status("Local incremental English draft model validated.")
    punctuation_path = None
    if punctuation:
        from .transcript_text import install_punctuation_model
        punctuation_path = install_punctuation_model(root, on_status)
    return {"model_dir":str(root),"asr":None if skip_asr else asr,"translation_engine":translation_engine if translation else None,
            "speaker_model":str(speaker_path) if speaker_path else None,
            "streaming_model":str(streaming_path) if streaming_path else None,
            "punctuation_model":str(punctuation_path) if punctuation_path else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir",required=True)
    parser.add_argument("--asr",default="large-v3-turbo")
    parser.add_argument("--translation",action="store_true")
    parser.add_argument("--skip-asr",action="store_true")
    parser.add_argument("--translation-engine",choices=["argos","nllb"],default="argos")
    parser.add_argument("--speaker",action="store_true",help="Install verified local anonymous speaker grouping model")
    parser.add_argument("--streaming",action="store_true",help="Install verified local incremental English draft model")
    parser.add_argument("--streaming-backend",choices=["fastconformer","zipformer"],default="fastconformer")
    parser.add_argument("--punctuation",action="store_true",help="Install verified local English punctuation model")
    args=parser.parse_args()
    result=prepare(args.model_dir,args.asr,args.translation,args.skip_asr,args.translation_engine,speaker=args.speaker,
                   streaming=args.streaming,punctuation=args.punctuation,streaming_backend=args.streaming_backend)
    print(json.dumps(result,ensure_ascii=True))


if __name__=="__main__":
    main()

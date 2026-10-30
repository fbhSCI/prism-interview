"""Download public local model weights, never interview data."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys


def prepare(model_dir, asr="large-v3-turbo", translation=True, skip_asr=False,
            translation_engine="argos", on_status=print, speaker=False, streaming=False,
            punctuation=False, streaming_backend="fastconformer"):
    root=Path(model_dir).expanduser().resolve()
    root.mkdir(parents=True,exist_ok=True)
    if not skip_asr:
        from faster_whisper.utils import _MODELS
        repository=_MODELS.get(asr)
        if not repository:
            raise ValueError("Unsupported ASR model: "+asr)
        target=root/"asr"/asr
        needed=4*1024**3 if "large" in asr else 2*1024**3
        if not (target/"model.bin").is_file() and shutil.disk_usage(root).free<needed:
            raise RuntimeError("模型目录可用空间不足，请在其他磁盘选择至少4GB空闲目录。")
        required = ("model.bin", "config.json", "tokenizer.json")
        cached = all((target/name).is_file() and (target/name).stat().st_size > 0 for name in required)
        if cached:
            on_status("已找到本地英语识别模型：" + asr)
        else:
            from huggingface_hub import snapshot_download
            progress_options = {}
            if sys.stderr is None:
                # PyInstaller's --windowed build has no stderr stream. Disable
                # HF's nested transfer bars as well as its outer iterator; the
                # desktop dialog has its own queued status/progress display.
                # This is process-local and never changes the user's env vars.
                from huggingface_hub.utils import disable_progress_bars, tqdm as hf_tqdm
                disable_progress_bars()

                class WindowedTqdm(hf_tqdm):
                    def __init__(self, *args, **kwargs):
                        kwargs["disable"] = True
                        super().__init__(*args, **kwargs)

                progress_options["tqdm_class"] = WindowedTqdm
            on_status("正在下载公开英语识别模型："+repository)
            snapshot_download(repository,local_dir=str(target),token=False,
                allow_patterns=["config.json","preprocessor_config.json","model.bin","tokenizer.json","vocabulary.*",
                                "README.md", "LICENSE", "LICENSE.*", "COPYING*"],
                max_workers=3, **progress_options)
            if not all((target/name).is_file() and (target/name).stat().st_size > 0 for name in required):
                raise RuntimeError("ASR download did not produce the required model files")
            _preserve_whisper_notice(target, repository, on_status)
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


def _preserve_whisper_notice(target, repository, on_status):
    """Keep source attribution with a new download, without reading any auth.

    Cached usable models never require a network call merely to open/check the
    application. Model redistribution notices are reviewed by the packager.
    """
    target = Path(target)
    (target / "MODEL_SOURCE.txt").write_text(
        "Converted model: https://huggingface.co/" + repository + "\n"
        "Original Whisper model: https://github.com/openai/whisper\n"
        "Original model license: MIT; preserve the upstream notices.\n", encoding="utf-8")
    notice = target / "WHISPER_LICENSE.txt"
    if notice.is_file():
        return
    try:
        from urllib.request import urlopen
        with urlopen("https://raw.githubusercontent.com/openai/whisper/main/LICENSE", timeout=20) as response:
            text = response.read(65537).decode("utf-8")
        if len(text.encode("utf-8")) > 65536 or "MIT License" not in text or "Permission is hereby granted" not in text:
            raise ValueError("Unexpected license response")
        temporary = notice.with_suffix(".txt.part")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(notice)
    except Exception:
        on_status("识别模型已下载；上游许可证暂未下载，请保留随包的第三方说明，重新分发前核对许可证。")


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

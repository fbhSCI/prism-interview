# Installation and RTX 4060 migration / 安装与迁移

## Requirements / 环境

Windows 10/11 x64; a supported NVIDIA driver for GPU recognition; 16 GB RAM recommended. CPU recognition remains available with higher latency. Source installation supports Python 3.11/3.12 x64. Reserve at least 15 GB for environment, caches and selected models. Actual memory and disk use depends on optional models.

本公开版与个人版分开安装。不要覆盖旧目录，不要把旧数据目录整体复制进来。若要使用自己的资料，请通过软件重新导入。密钥请在新软件内重新填写。

## A. Prebuilt application / 便携程序

1. Download `PrismInterview-0.4.0-Windows-x64.zip` from the GitHub release.
2. Extract the entire folder into a new writable path, for example `D:\PrismInterview`.
3. Keep `PrismInterview.exe`, `_internal`, `runtime.json`, and the notices together. Do not run an EXE from inside a ZIP.
4. Prepare model weights as described below. The release does not bundle every optional model or provider CLI.
5. Run `PrismInterview.exe`. First launch stays idle.

## B. Source installation / 源码安装

In PowerShell, inside a fresh checkout:

```powershell
.\setup_windows.ps1 -Gpu -ModelDirectory 'D:\PrismInterviewModels' -TemporaryDirectory 'D:\PrismInterviewTemp'
.\start_windows.bat
```

If execution policy blocks the script, invoke this single script with a process-specific bypass:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\setup_windows.ps1 -Gpu -ModelDirectory 'D:\PrismInterviewModels' -TemporaryDirectory 'D:\PrismInterviewTemp'
```

For deliberate noninteractive setup:

```powershell
.\setup_windows.ps1 -Gpu -NoPrompt -DownloadAsr -DownloadTranslation -ModelDirectory 'D:\PrismInterviewModels' -TemporaryDirectory 'D:\PrismInterviewTemp'
```

Model downloads need network access and may take time. For optional Argos translation use `install_translation_windows.bat`; some optional Python dependencies need additional installation and are not inside the compact Windows app. Qwen cloud translation is a separate choice and needs consent, an endpoint and a valid API key.

### Preparing model paths for a prebuilt app

Use source setup on a supported Python environment to download models. Then either use `export_runtime.ps1` to copy the allowlisted model and CUDA files into the app's separate `models` / `gpu` folders, or explicitly configure model paths in `runtime.json`:

```json
{"data_dir":"data","model_dir":"D:\\PrismInterviewModels","cuda_dir":"D:\\PrismInterview\\gpu"}
```

The optional CUDA path must contain supported runtime libraries in the layout described by `export_runtime.ps1`; do not copy random system DLLs. The GPU-enabled release already contains its collected runtime libraries under `_internal`.

## First interview / 第一次使用

1. **Audio**: set Zoom's speaker to a specific device, select that output in Prism Interview, and verify the input meter with remote speech. Loopback hears the selected device, including other apps playing through it. It does not automatically capture your microphone.
2. **Recognition**: start with Whisper `large-v3-turbo`, GPU acceleration and the default single remote speaker mode. The faster confirmation profile is optional; measure it with your own speech.
3. **Translation**: choose a local engine with installed weights, or configure Qwen and allow confirmed segments to be sent. An unavailable engine is not a working translator.
4. **Your context**: import CV, presentation, project summaries and reference documents. Fill in the target role/background. Check extraction and document names before asking questions.
5. **Answers**: choose a provider and exact available model in the model settings. Enter credentials only there. Check the provider balance/allowance yourself. Official Codex integration requires the user's own supported CLI and login; this application does not supply a subscription or reuse another app's authentication files.
6. **Reading**: resize or float panels, enable/disable follow-latest, and place transparent captions on the desired monitor. Keep English and Chinese pairs visible.
7. **History**: click Save conversation for a manual local snapshot, or explicitly enable auto-save. History opens local records without restarting capture or contacting a model. Export a copy only when needed. Auto-save defaults off.

## Moving to another computer / 迁移到 4060 笔记本

Copy the full application folder or source archive to a **new** destination. Download/copy only model weights whose licenses permit your use. Keep notices. Do not copy `.venv` between machines; recreate it for source installs. Do not transfer another application's authentication files or Windows Credential Manager entries.

On the laptop, install/update the NVIDIA driver, verify microphone/speaker permissions, re-enter provider credentials, configure paths and re-import your own documents. Use the same output device in Zoom and Prism Interview. Run a short bilingual mock interview with a second participant. Check complete sentences, translation, automatic question capture, document grounding, overlay and save/export before a real interview.

### Codex handoff prompt / 给另一台电脑 Codex 的指令

> Work only in this new Prism Interview directory; preserve other installations. Read AGENTS.md and this guide. Inspect Python, Windows, NVIDIA driver, free disk space and model paths. Install missing public dependencies and models through setup_windows.ps1. Do not read/copy credentials from another app. Do not enable cloud use or start capture without my configuration. Help me enter provider keys in the software, import my own materials, and run a mock Zoom test. Report actual measured results and remaining limitations; do not call a loopback test a two-end interview test. Keep transcripts and documents out of diagnostics and Git.

## Build / 构建

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\build_windows.ps1 -IncludeGpuLibraries -OutputDirectory 'D:\PrismBuild'
```

Optional engines and model weights have separate licenses. Review `THIRD_PARTY_NOTICES.md`; review generated dependency notices before distributing your build.

## Troubleshooting / 排错

- **No window**: fully extract first; run `start_windows.bat` to see an error. Source installs need the project's `.venv`.
- **No remote speech**: use the same output device in Zoom and Prism, check Zoom volume and the app's input meter. Bluetooth devices can switch profiles during calls.
- **No Chinese**: check installed local weights or the selected cloud translation endpoint/consent/key. Caption translation and answer generation are independent.
- **Slow answers**: distinguish speech collection, translation, retrieval and model latency. Configure a faster available answer model, verify network and avoid an unnecessarily long question/context. Do not switch speech models expecting to fix cloud token generation.
- **Generic answers**: inspect imported document extraction, relevant evidence and candidate context. Add an explicit project contribution summary. Retrieval is not training, and no provider guarantees factual answers.


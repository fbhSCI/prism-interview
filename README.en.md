<p align="center"><img src="docs/assets/brand.svg" width="72" alt="Prism Interview"></p>
<h1 align="center">Prism Interview</h1>
<p align="center"><strong>Understand the English question. Answer with your own preparation.</strong></p>
<p align="center"><a href="https://fbhsci.github.io/prism-interview/">Product site</a> · <a href="https://github.com/fbhSCI/prism-interview/releases">Download for Windows</a> · <a href="README.md">简体中文</a> · <a href="docs/GETTING_STARTED.md">Setup guide</a></p>

Prism Interview helps Chinese-speaking candidates during English interviews. In a Zoom call, it shows the interviewer's English and Chinese translation, collects the question, retrieves relevant passages from your preparation materials, and suggests an answer with paired English and Chinese paragraphs.

Use it for technical interviews, research interviews and professional discussions—or select a headset microphone to practise your own English.

![Actual desktop interface with example content](docs/assets/desktop.png)

## What it does during an interview

- **Keep up with speech.** Paired English–Chinese captions update during the conversation, independently of answer generation. Resize or float panels, or put transparent captions on another screen.
- **Reduce manual work.** Automatic assistance collects questions and follow-ups, including requests such as “Walk me through your method.” You can still edit a question, generate manually or cancel.
- **Answer from your own context.** Import a CV, slides, papers, project notes or a research plan before the interview. Add your target role and actual contribution. Relevant passages are retrieved for each question, with sources available beside the answer.
- **Choose what to hear.** Select meeting/computer audio, microphone/headset input, or both. Source labels distinguish “对方” (remote) and “我” (you). Your microphone speech is translated without triggering automatic answers.
- **Review afterwards.** Save manually or enable local auto-save; browse history and export TXT, Markdown or JSON. Auto-save is off initially, and history does not include audio recordings.

## First use: six steps

Windows 10/11 x64, 16 GB RAM and an NVIDIA RTX GPU are recommended. CPU recognition is available with higher latency.

1. **Download and install.** Open [Releases](https://github.com/fbhSCI/prism-interview/releases), expand Assets and download `PrismInterview-0.4.1-Windows-Setup.zip`. Extract the whole folder, run **安装PrismInterview.cmd**, then open **Prism Interview 0.4.1** from the desktop. Python and administrator access are not required.
2. **Configure Qwen.** Click **配置千问 API Key** or **选择回答模型** and enter your own Qwen key. A fresh installation selects **`qwen3.8-flash`** by default. Follow the steps below.
3. **Prepare recognition.** In **模型与设置**, click **下载／检查语音模型** and wait for completion before your interview. This is first-time preparation.
4. **Select the source.** For Zoom, choose **会议／电脑声音** and the same output device as Zoom. For practice, choose **麦克风／耳麦** and your headset microphone. Choose **双方同时识别** when you want both sides captioned, with each device selected separately.
5. **Add your materials.** Import files through **面试资料库**. Fill in your field, target role and actual contribution under **模型与设置 → 面试背景**.
6. **Try a short call.** Click **开始监听** and ask a partner to say a few English sentences. Check captions, question collection and answer suggestions before using it in a formal conversation.

The [setup guide](docs/GETTING_STARTED.md) covers device selection, first-time downloads, troubleshooting and moving to another PC.

## Connect Qwen

We recommend **Qwen `qwen3.8-flash`** as a starting point for caption translation and interview answers. The [official model list](https://www.alibabacloud.com/help/en/model-studio/models) describes Flash as a fast-response option. Local Whisper handles speech recognition; Qwen handles translation and answers, so their delays have different causes.

1. Create a key in [Alibaba Cloud Model Studio](https://bailian.console.aliyun.com/). Follow the official [API key guide](https://www.alibabacloud.com/help/en/model-studio/get-api-key) if you are new to the platform.
2. In **配置千问 API Key**, choose Qwen and the region matching that key. Paste it into **API Key**. Use **记住本机默认密钥** to save it for this Windows user.
3. Set **选中模型** to `qwen3.8-flash`. Allow the question, background and retrieved excerpts to be sent. Confirm your actual free allowance or explicitly enable **允许千问使用付费余额**. Provider charges apply; the application does not supply credit.
4. Click **测试短回答**, then **使用此模型** after a successful result. If the model is unavailable, check your region and its supported model IDs.
5. For Qwen captions, go to **模型与设置**, choose **千问 Qwen · 连续字幕云翻译**, set `qwen3.8-flash`, check the matching regional endpoint, enable **允许持续英译中使用千问云服务**, and save.

Answers and continuous captions need separate consent. Enter keys inside the application; keep them out of issues, chats and screenshots.

## Make specialist answers relevant

Papers alone are not enough. Include a short account of **your work**: your role, method, results, limitations and the pages supporting them. Supported formats are PDF, Word (`.docx`), PowerPoint (`.pptx`), TXT and Markdown. Scanned PDFs need OCR first.

New files participate after import; re-import files that change. The app retrieves relevant passages when a question arrives. It does **not** retrain the model. Use the [mock interview](docs/SAMPLE_INTERVIEW.md) to check which materials the answer actually uses.

Recognition, translation and suggestions can be wrong. Rehearse with your own materials and check important claims. Capture and uploads do not start on launch. Cloud answers send the context and excerpts selected for the question. [Privacy details](docs/PRIVACY.md)

## Other models and development

DeepSeek, local Ollama, custom compatible endpoints and optional official Codex integration remain available. Each needs the appropriate account or local installation.

[Source installation and builds](docs/GETTING_STARTED.md#source-installation) · [Contributing](CONTRIBUTING.md) · [Architecture](docs/ARCHITECTURE.md) · [MIT license](LICENSE) · [Third-party licenses](THIRD_PARTY_NOTICES.md)

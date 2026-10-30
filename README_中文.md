<p align="center"><img src="docs/assets/brand.svg" width="76" alt="Prism Interview"></p>
<h1 align="center">Prism Interview</h1>
<p align="center"><strong>听懂问题，带着自己的准备回答。</strong><br>面向 Windows 的开源英语面试双语工作台。</p>
<p align="center"><a href="https://fbhSCI.github.io/prism-interview/">产品主页</a> · <a href="https://github.com/fbhSCI/prism-interview/releases">下载</a> · <a href="README.md">English</a> · <a href="docs/GETTING_STARTED.md">安装与迁移指南</a></p>

![Prism Interview 产品示意，非真实面试记录](docs/assets/workspace.svg)

把英语识别、逐段中英对照、个人资料检索和回答建议放进同一个桌面工作台。适用于求职、学术交流、技术面试等需要理解英语问题的场景。公开版保留原有能力，首次启动不包含任何人的论文、履历、研究方向或密钥。

**预览版 · Windows x64 · Python 3.11/3.12 · 应用源码采用 MIT 协议**

<details><summary>查看实际 Windows 界面（使用合成示例）</summary>

![实际桌面界面，使用合成面试内容](docs/assets/desktop.png)

</details>

## 主要功能

- **本机听写**：通过 WASAPI 回环采集选定播放设备，Whisper 在本机识别；支持 NVIDIA GPU。默认只听远端一人，也可启用远端多人匿名分组。
- **双语阅读**：英语和对应中文成组显示，可选择跟随最新、调整面板大小、浮出窗口或使用透明悬浮字幕；长内容可完整阅读。
- **结合自己的准备资料**：导入 PDF、Word、PPT，本地建立索引，按问题检索相关片段和证据卡，再交给所选模型。不会因导入他人的论文就把论文成果当作你的经历。
- **自由选择模型**：本地 Ollama、DeepSeek、千问、兼容接口和可选官方 Codex 接入；识别、翻译、回答分别配置。
- **自动或手动辅助**：支持陈述式请求、续问收集、取消、答案记录，保留人工控制。
- **可选保存对话**：手动保存，或主动开启自动保存；在本机查看历史并导出 JSON、Markdown、TXT。默认关闭自动保存，不保存原始录音。

## 安装与使用

优先从 [Releases](https://github.com/fbhSCI/prism-interview/releases) 下载 Windows 程序，完整解压后运行 `PrismInterview.exe`，不要只拷贝 EXE。语音和本地翻译模型需另行准备；完整步骤见 [安装指南](docs/GETTING_STARTED.md)。

源码安装需 Python 3.11 或 3.12（64 位）。在解压后的项目目录打开 PowerShell：

```powershell
.\setup_windows.ps1 -Gpu -ModelDirectory 'D:\PrismInterviewModels'
.\start_windows.bat
```

程序会创建独立环境并提供模型下载选项。建议为环境及模型预留至少 15 GB，安装临时目录使用英文路径。首次启动不监听、不自动调用云服务。

1. Zoom 和本软件选择同一个播放设备；先检查远端声音能否采集。
2. 配置翻译方式；使用云翻译需单独同意发送已确认的英语语段。
3. 导入自己的 CV、岗位要求、汇报材料、参考资料，填写候选人背景和目标岗位。
4. 在软件内填写所选服务的密钥；不要把密钥粘贴进公开仓库或截图。
5. 在参与者知情并符合面试规则的前提下启动监听，先做模拟面试。

## 新资料如何参与回答

导入或更新资料后，软件提取文本并更新本地检索索引；问答时检索与当前问题有关的片段，再与背景、问题、证据一起发送给所选回答模型。这属于检索增强，**没有重新训练模型，也不意味着模型永久记住整份文件**。可手动补充事实卡、明确个人贡献、核对出处，帮助模型区分自己的经历与外部资料。

## 数据与费用

语音识别留在本机；配置本地模型可离线使用。选择云回答会发送问题及相关资料片段；选择云翻译会在单独同意后发送确认的英语语段及有限上下文。API 额度、订阅和计费由对应平台决定，本项目不赠送额度，也不提供共享密钥。

保存的对话是本机明文文件，可能含个人信息，请谨慎备份和分享。公开版使用独立的数据目录及凭据命名空间，不自动继承其他安装的密钥或资料。详见 [隐私说明](docs/PRIVACY.md)。

## 已知边界与贡献

这是辅助软件：听写、说话人分组、翻译、检索和回答都可能出错。速度随电脑配置、停顿方式、网络、模型而变化，不能承诺零等待或百分之百正确。请核对关键事实，遵守面试规则，不用它伪造经历。

开发与构建步骤见 [English README](README.md)、[架构](docs/ARCHITECTURE.md)、[贡献指南](CONTRIBUTING.md)。第三方授权见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本项目与 Zoom 或各模型供应商无隶属关系。


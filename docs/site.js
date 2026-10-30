(() => {
  let language = 'en';
  let step = 0;
  const copy = {
    en: [
      ['LOCAL FIRST', 'Hear the selected system output.', 'WASAPI loopback captures the playback device you choose. Whisper recognizes English locally. Use an installed local translator or explicitly enable a cloud translation service.', 'SYSTEM AUDIO', 'LOCAL WHISPER', 'ENGLISH + 中文'],
      ['YOUR PREPARATION', 'Give the question useful context.', 'Import your CV, slides, job description or project notes. Local retrieval selects relevant excerpts and evidence; it does not retrain a model or make another person’s work yours.', 'YOUR DOCUMENTS', 'LOCAL RETRIEVAL', 'RELEVANT EVIDENCE'],
      ['YOUR MODEL CHOICE', 'Turn context into a suggestion.', 'Choose a local model or configure a supported provider. Automatic or manual assistance keeps you in control. Review the suggestion and express the answer in your own words.', 'QUESTION + EVIDENCE', 'SELECTED MODEL', 'PAIRED EN / 中文']
    ],
    zh: [
      ['优先在本机处理', '听见你选定的系统播放声。', 'WASAPI 回环采集选定播放设备的声音，Whisper 在本机识别英语。使用已安装的本地翻译模型，或主动开启云端翻译服务。', '系统播放声音', '本地 WHISPER', 'ENGLISH + 中文'],
      ['你的面试准备', '为问题提供有用的背景。', '导入履历、幻灯片、岗位说明或项目笔记。本地检索选择相关片段与证据；这不是重新训练模型，也不会把他人成果当作你的贡献。', '你的资料', '本地检索', '相关片段与证据'],
      ['你选择的模型', '从资料与问题，组织回答建议。', '选择本地模型或配置支持的服务商。可自动或手动触发辅助。请核对建议，再用自己的表达作答；软件不会自动替你说话。', '问题 + 证据', '所选回答模型', '中英配对回答']
    ]
  };
  function renderStep() {
    const values = copy[language][step];
    ['step-tag', 'step-title', 'step-copy', 'node-a', 'node-b', 'node-c'].forEach((id, i) => { document.getElementById(id).textContent = values[i]; });
    document.querySelectorAll('[data-step]').forEach((button, i) => {
      button.setAttribute('aria-selected', String(i === step));
      button.tabIndex = i === step ? 0 : -1;
    });
    document.getElementById('step-detail').setAttribute('aria-labelledby', `step-${step}`);
  }
  document.getElementById('language').addEventListener('click', () => {
    language = language === 'en' ? 'zh' : 'en';
    document.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
    document.querySelectorAll('[data-en]').forEach(el => { el.innerHTML = el.dataset[language]; });
    const button = document.getElementById('language');
    button.textContent = language === 'en' ? '中文' : 'EN';
    button.setAttribute('aria-label', language === 'en' ? 'Switch to Chinese' : '切换到英语');
    renderStep();
  });
  document.querySelectorAll('[data-step]').forEach(button => {
    button.addEventListener('click', () => { step = Number(button.dataset.step); renderStep(); });
    button.addEventListener('keydown', event => {
      if (['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) {
        event.preventDefault();
        step = event.key === 'Home' ? 0 : event.key === 'End' ? 2 : (step + (event.key === 'ArrowRight' ? 1 : 2)) % 3;
        renderStep(); document.getElementById(`step-${step}`).focus();
      }
    });
  });
  renderStep();
})();

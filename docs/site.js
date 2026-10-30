(() => {
  let language = 'zh';
  let step = 0;
  const copy = {
    en: [
      ['LOCAL FIRST', 'Choose the interviewer, your mic, or both.', 'Select the Zoom playback device and, when needed, your headset mic. Separate streams identify each side; Whisper recognizes English locally. Qwen Flash translates confirmed speech.', 'MEETING AUDIO + HEADSET', 'LOCAL WHISPER', 'ENGLISH + 中文'],
      ['YOUR PREPARATION', 'Give the question useful context.', 'Import your CV, slides, job description or project notes. Local retrieval selects relevant excerpts and evidence; it does not retrain a model or make another person’s work yours.', 'YOUR DOCUMENTS', 'LOCAL RETRIEVAL', 'RELEVANT EVIDENCE'],
      ['YOUR MODEL CHOICE', 'Get a bilingual answer with relevant sources.', 'Automatic assistance collects the interviewer’s question, retrieves your materials, then asks Qwen Flash for paired English–Chinese suggestions. Your microphone speech stays out of automatic question routing.', 'QUESTION + EVIDENCE', 'SELECTED MODEL', 'PAIRED EN / 中文']
    ],
    zh: [
      ['优先在本机处理', '选择面试官、自己的麦克风，或双方。', '选好 Zoom 正在使用的输出设备和耳麦麦克风。两路音频分开处理，Whisper 在本机识别英语，千问 Flash 翻译确认的语段。', '会议声音 + 耳麦', '本地 WHISPER', 'ENGLISH + 中文'],
      ['你的面试准备', '为问题提供有用的背景。', '导入履历、幻灯片、岗位说明或项目笔记。本地检索选择相关片段与证据；这不是重新训练模型，也不会把他人成果当作你的贡献。', '你的资料', '本地检索', '相关片段与证据'],
      ['你选择的模型', '把问题和相关资料，变成中英回答建议。', '自动辅助收集面试官的问题，检索你的材料，再交给千问 Flash 生成逐段中英对照建议。自己的麦克风发言不会重复触发自动回答。', '问题 + 证据', '所选回答模型', '中英配对回答']
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

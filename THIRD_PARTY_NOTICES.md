# Third-party components

The project-specific Python source is provided under the included MIT license.
That license does not replace licenses of bundled libraries, model weights,
NVIDIA DLLs, or the separate Codex CLI.

- **pypdf**: BSD-3-Clause. The public edition uses pypdf for PDF extraction.
  PyMuPDF/MuPDF are not dependencies and are excluded from the public build.
  The personal research edition used a different extraction adapter; its
  binary is not part of this release.
- **PySide6 Essentials / Qt**: LGPL-3.0/GPL/commercial terms, with component-specific
  exceptions. The onedir build uses dynamically linked Qt libraries. Upstream:
  <https://doc.qt.io/qtforpython-6/licenses.html>.
- **faster-whisper, CTranslate2, NumPy, SciPy, SentencePiece, SoundCard, pypdf,
  python-docx, python-pptx, PyAV and transitive dependencies**: upstream licenses
  apply. PyAV links FFmpeg; inspect the binary distributor's exact license/build.
- **NVIDIA CUDA/cuDNN wheels**: NVIDIA license terms; separate downloads by
  default. They are not covered by the project's MIT license.
- **Whisper/Argos weights**: downloaded separately from their official model
  repositories; retain each model's supplied license and metadata. Model weights
  and their download size are not part of the source archive.
- **Optional NLLB 1.3B translation weights**: CC-BY-NC-4.0, for noncommercial
  use only; separately downloaded from the OpenNMT converted model repository.
  A polished interface does not change those model-license restrictions.
- **Codex CLI**: installed and authenticated separately by the user. Subscription
  terms and account usage limits apply; no authentication data is distributed.

`build_windows.ps1` copies discoverable installed-package license texts and an
installed dependency/version inventory into the portable application's
`licenses/` directory. This inventory is a build record, not a substitute for
reviewing the license and corresponding-source requirements of redistribution.

This public edition is open source. Polished visual design does not change
third-party redistribution obligations. The dynamically linked Qt libraries
remain replaceable in the onedir layout; do not prevent user replacement or
reverse engineering for debugging modifications permitted by the LGPL.
Qt corresponding sources are available at https://download.qt.io/official_releases/qt/
and PySide sources at https://code.qt.io/pyside/pyside-setup.git/ ; use the exact
versions listed in the generated build inventory. PyAV wheel builds and their
FFmpeg source/build recipes are available at https://github.com/PyAV/pyav-wheels
and https://github.com/PyAV/PyAV . Preserve the bundled wheel notices.
# Optional TranslateGemma

The optional `translategemma:4b` model is distributed by Ollama and governed by
Google's Gemma Terms of Use. Weights are downloaded separately and are not included
in the program-update or source archives. Model and prompt format:
https://ollama.com/library/translategemma:4b
Terms: https://ai.google.dev/gemma/terms

# Incremental English captions (0.2.5)

- Runtime: [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), Apache-2.0.
- Original acoustic model: **NVIDIA FastConformer Hybrid Large Streaming**,
  published by NVIDIA under **Creative Commons Attribution 4.0 International
  (CC-BY-4.0)**. Official model card:
  <https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi>.
  The converted model also identifies the NVIDIA 480ms model at
  <https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo/models/stt_en_fastconformer_hybrid_large_streaming_480ms>.
- This application uses the CTC streaming branch converted to ONNX and INT8
  quantization, distributed by **csukuangfj / sherpa-onnx**:
  <https://huggingface.co/csukuangfj/sherpa-onnx-nemo-streaming-fast-conformer-ctc-en-480ms-int8>.
  Conversion/quantization is a modification of the original model format and
  precision; this project does not claim authorship of NVIDIA's model.
- Pinned converted-model revision:
  `66e9b61a9fe334eba579700599e391c6ee093633`.
  `model.int8.onnx`: 132,048,430 bytes; SHA256:
  `927a40a30e683e1dce4137f0e8ccae96e728ae8475f4f9798ce2aca8714272ad`.
  `tokens.txt` is checked against Git blob
  `3228fbf04d889e59138c56a1b19fd5fe795c3dbf`.
- Attribution: NVIDIA, original FastConformer/NeMo model; csukuangfj and
  sherpa-onnx contributors, ONNX conversion and runtime integration.
  License: <https://creativecommons.org/licenses/by/4.0/>. Retain attribution,
  model metadata, source links and the indication of conversion/quantization
  when transferring the weights. The project's MIT license does not supersede
  these model obligations.
- Optional alternative draft decoder: sherpa-onnx streaming English Zipformer,
  Apache-2.0, from
  <https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-26>.
  It is not the default 0.2.5 FastConformer decoder.
- Draft weights run locally on CPU. "480ms" is a model configuration, not an
  end-to-end caption, translation or scientific-answer latency guarantee.

# Local punctuation and case restoration (0.2.5)

- **Edge-Punct-Casing**, by frankyoujian and contributors, implementation of
  *A lightweight and efficient punctuation and word casing prediction model for
  on-device streaming ASR*. Code and model source:
  <https://github.com/frankyoujian/Edge-Punct-Casing> and
  <https://huggingface.co/frankyoujian/Edge-Punct-Casing>.
  Upstream repository license: **Apache-2.0**.
- The int8 ONNX model and BPE vocabulary are distributed through sherpa-onnx:
  <https://github.com/k2-fsa/sherpa-onnx/releases/tag/punctuation-models>.
  Archive `sherpa-onnx-online-punct-en-2024-08-06.tar.bz2` SHA256:
  `9f5e5a72c7d2829635bd074fce92b6bbd5b78da8a52e7ad8ed1be933f366b99d`.
  Installed `model.int8.onnx`: 7,490,500 bytes; SHA256:
  `9d611f445fe4a46186080fe161be6059d87d72eb88d3a8cb00c1a06e83a6067e`.
- The explicit installer preserves a copy of the upstream license and source
  metadata with the weights. It uses the INT8 derivative for CPU inference;
  project-specific guards reject lexical/number/formula edits. Punctuation is
  predicted, not a guarantee that every sentence boundary is correct.
- Default update/source archives do not include these separately downloaded
  weights. A runtime export containing them must retain their license and
  attribution files.

# Local anonymous speaker grouping (0.2.4)

- Runtime: [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), Apache-2.0.
- Acoustic model: NVIDIA TitaNet-Large English, about 23M parameters,
  [official NVIDIA model card](https://huggingface.co/nvidia/speakerverification_en_titanet_large).
  NVIDIA publishes these weights under **Creative Commons Attribution 4.0
  International (CC-BY-4.0)**. Toolkit code licenses do not replace this model
  attribution requirement.
- Download: <https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_large.onnx>
- SHA256: `d51abcf31717ef28162f26acb9d44dd4127c3d44c9b8624f699f3425daca8e77`.
- Attribution: NVIDIA, Nithin Rao Koluguri, Taejin Park and Boris Ginsburg,
  *TitaNet: Neural Model for Speaker Representation with 1D Depth-wise Separable
  Convolutions and Global Context*, ICASSP 2022, <https://arxiv.org/abs/2110.04410>.
- The model runs on CPU and creates session-only anonymous acoustic profiles.
  It does not identify names, guarantee speaker separation, or separate
  simultaneous speakers. Short/mixed/ambiguous audio is labeled unconfirmed.


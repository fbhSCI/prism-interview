# 0.4.0 — public interview edition

The public edition is a separate installation derived from the earlier research workspace. It keeps local speech recognition, paired bilingual reading, transparent overlay, resizable/floating panels, model selection, document retrieval and automatic/manual answer assistance.

New: a redesigned interview workspace; explicit manual saving; optional default-off auto-save; a read-only local history browser; JSON/Markdown/TXT export; independent public-edition storage and credentials; general candidate/role context; pypdf-based document extraction; safer settings validation and endpoint changes.

The release does not contain a candidate's biography, papers, recordings, keys or preset cloud authorization. Model weights remain separate from the public application download. A full local transfer bundle may include allowlisted public speech/translation model files with their licenses.

Validation on the development Windows machine (RTX 3080 Ti): **1722 tests passed, 37 skipped**. Skipped cases include licensed external recording fixtures that are not redistributed. The frozen application also passed CUDA Whisper recognition, local Argos translation, streaming-model loading, punctuation, anonymous speaker runtime and PDF extraction checks using synthetic speech, with no capture or cloud requests. Idle startup was checked with an empty isolated data directory.

These checks are not a new two-end Zoom interview or a measured RTX 4060 laptop acceptance test. No live cloud-provider accuracy benchmark was performed for this public edition. Recognition, translation and generated answers remain fallible.

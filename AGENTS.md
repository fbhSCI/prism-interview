# Prism Interview contributor instructions

This is an independent public, general-purpose edition. Do not modify or migrate the private ResearchCopilot installation.
- Preserve capture, ASR, bilingual translation, automatic/manual answers, model selection, document retrieval, floating panels and accessibility.
- First launch never starts listening or cloud calls. Cloud use requires explicit user configuration and consent.
- No personal profiles, documents, transcripts, credentials, private endpoints or third-party interview recordings belong in source control or release artifacts.
- Use a separate PrismInterview data directory and credential namespace.
- Conversation persistence is explicit, local, schema-versioned, atomic and excludes audio/settings/secrets. Auto-save defaults off.
- Keep complete bilingual text readable without mandatory expansion; preserve stable rendering and optional follow-latest.
- Do not promise correct answers or fixed latency. Candidate claims must be supported by user-provided material.
- Run meaningful regression tests with isolated data paths. Use synthetic public examples.
- Package source and application separately; never redistribute model weights without license review.

# Privacy and data flow

No telemetry or hosted backend is required by Prism Interview. Provider SDKs, model download hosts and optional CLIs have their own behavior and terms.

| Activity | Local data | Network boundary |
| --- | --- | --- |
| Speech recognition | Selected system output and/or microphone, plus a local speech model | Audio is not sent by the recognition pipeline |
| Local translation | Confirmed text and installed translation weights | None for inference |
| Cloud caption translation | Confirmed English segments and bounded same-card context | Sent to the configured translation endpoint only with consent |
| Document import | Extracted text, local search index and evidence cards | No automatic full-corpus upload |
| Cloud answers | Question, selected excerpts, configured candidate background and limited conversation context | Sent to the selected answer provider |
| Local answers | Same prompt with local inference | Local model endpoint only; review your endpoint configuration |
| Session history | Speaker labels, English/Chinese conversation, questions, suggestions, retrieved evidence excerpts and source metadata | Local save/export only; no raw audio |

Audio sources are explicit choices: computer/meeting output, microphone/headset, or both. In both-source mode, source labels distinguish the remote audio from your microphone. Microphone captions do not trigger automatic interview answers. Loopback may include other applications playing through the selected output device.

Auto-save is off on fresh installs. Manual saving and enabling auto-save are explicit controls. Saved sessions are plaintext and not encrypted; file-system backups or synchronization tools can copy them. Delete/manage exported files using normal file tools. A JSON export is data, not an instruction for executing code.

Preferences do not contain API keys. On Windows, saved provider keys use this application's own Credential Manager namespace, scoped to provider and HTTPS endpoint. Changing a custom endpoint does not silently give it a key saved for another endpoint. This edition does not migrate another application's vault or data directory.

The public repository and release exclude candidate profiles, private papers, indexes, saved sessions, logs, recordings, runtime paths and credentials. General diagnostics must not contain conversation text. Report accidental exposure privately and revoke exposed credentials at their provider.


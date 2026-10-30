# Security and privacy reports

Do not publish API keys, account tokens, interviews, candidate documents, or exploitable personal data in an issue. Use GitHub's private vulnerability reporting when available. Otherwise open an issue containing only a minimal, non-sensitive description and ask the maintainer for a private channel.

If a key was exposed, revoke it at the provider first. Removing a file or commit alone does not revoke a credential. This project does not ship shared API keys. Windows credentials are scoped to this application and endpoint; local history is plaintext and should be handled as private user data.

Only install releases from the project repository or build from reviewed source. Unsigned Windows preview binaries may trigger SmartScreen; the project does not claim code signing or a security audit.


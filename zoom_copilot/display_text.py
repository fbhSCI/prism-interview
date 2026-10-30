"""Presentation-only cleanup of machine source IDs in bilingual answer strings.

The evidence objects/citations remain untouched. Scientific brackets, including
``[S0,S1,S2]``, numerical references and intervals are not treated as source IDs.
"""
from __future__ import annotations

import re
from typing import Iterable


_HASH_CITATION = re.compile(r"\[\s*[0-9a-fA-F]{16,64}\s*\]")


def clean_answer_text(text: str, streaming: bool = False,
                      citation_ids: Iterable[str] = ()) -> str:
    """Hide opaque citations, including a still-incomplete streamed hash marker.

    Only exact single known IDs are removed. In particular ``[S0,S1,S2]`` must
    survive even when S1 or S2 happens to be an evidence ID.
    """
    value = str(text or "")
    value = _HASH_CITATION.sub("", value)
    identities = {str(item) for item in citation_ids if str(item)}
    for identity in identities:
        value = re.sub(r"\[\s*" + re.escape(identity) + r"\s*\]", "", value)
    if streaming:
        start = value.rfind("[")
        if start >= 0 and "]" not in value[start:]:
            suffix = value[start + 1:].strip()
            # Delay possible citation prefixes until they resolve; ordinary
            # mathematical text is displayed as soon as it differs from an ID.
            hash_prefix = len(suffix) <= 64 and bool(re.fullmatch(r"[0-9a-fA-F]*", suffix))
            known_prefix = any(identity.startswith(suffix) for identity in identities)
            if hash_prefix or known_prefix:
                value = value[:start]
    value = re.sub(r"[ \t]+(?=[,.;:!?，。；：！？])", "", value)
    # Hold trailing whitespace until the next visible word/punctuation, keeping
    # cleaned field snapshots monotonic for a delta-only callback.
    return value.rstrip() if streaming else value.strip()

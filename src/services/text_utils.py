from __future__ import annotations

import re

_TOKENS_PER_WORD = 0.7

# A markdown heading line: leading "#" run, then the heading text (captured).
_HEADING_RE = re.compile(r"^#{1,6}\s+(.*\S.*)$", re.MULTILINE)


def extract_titles(text: str) -> list[str]:
    """Return the markdown heading texts (without the leading '#' markers)."""
    return [m.strip() for m in _HEADING_RE.findall(text)]


def truncate_tokens(text: str, max_tokens: int) -> str:
    words = text.split()
    max_words = int(max_tokens / _TOKENS_PER_WORD)
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])

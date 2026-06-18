from __future__ import annotations

_TOKENS_PER_WORD = 0.7


def truncate_tokens(text: str, max_tokens: int) -> str:
    words = text.split()
    max_words = int(max_tokens / _TOKENS_PER_WORD)
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])

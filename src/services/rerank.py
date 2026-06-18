from __future__ import annotations

from core.config import settings
from core.llm import generate_structured
from core.text import truncate_tokens
from models.schemas import Relevance

_RERANK_PROMPT = """You are ranking a section of a clinical trial protocol by how relevant to a query.

QUERY:
{query}

Rate how relevant the text below, using an integer score from 0 to 10:
- 0 = the section is unrelated.
- 10 = the section directly contains the information.

Return ONLY valid JSON in the following format:
{{"relevance": <integer 0-10>}}

SECTION:
{section}

RELEVANCE:
"""


def rerank_score(query: str, text: str) -> int:
    section = truncate_tokens(text.strip(), settings.max_tokens)
    prompt = _RERANK_PROMPT.format(query=query.strip(), section=section)
    return generate_structured(prompt, Relevance).relevance


def rerank(query: str, results: list[dict], top_n: int | None = None) -> list[dict]:
    reranked = [
        {
            **r,
            "rerank_score": rerank_score(
                query, r.get("original") or r.get("matched_text", "")
            ),
        }
        for r in results
    ]
    reranked.sort(key=lambda r: (r["rerank_score"], r["score"]), reverse=True)
    return reranked[:top_n] if top_n is not None else reranked

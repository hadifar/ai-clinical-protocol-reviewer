from __future__ import annotations

from core.config import settings
from core.embeddings import embed_dense, embed_sparse
from core.text import truncate_tokens
from core.vectorstore import DENSE, SPARSE, get_client
from models.schemas import Relevance
from services.ai_agent import generate_structured

_RERANK_PROMPT = """You are reranking agent.

Rate how well the SECTION below lets to extract the information about the QUERY, using an integer score from 0 to 10:
- 0  = unrelated; garbage or noisy text; no information is here.
- 3  = same general topic, but the specific information is missing.
- 7  = the information is present but partial, implicit, or mixed with unrelated content.
- 10 = the section explicitly and completely states the information about the query.

Judge only whether the answer is present, not how well written the section is.

Return ONLY valid JSON in the following format:
{{"relevance": <integer 0-10>}}

QUERY:
{query}

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


def search(query: str, k: int = 12, top_n=4) -> list[dict]:
    from qdrant_client.models import Fusion, FusionQuery, Prefetch

    dense = embed_dense([query])[0]
    sparse = embed_sparse([query])[0]
    response = get_client().query_points(
        settings.qdrant_collection,
        prefetch=[
            Prefetch(query=dense, using=DENSE, limit=k * 4),
            Prefetch(query=sparse, using=SPARSE, limit=k * 4),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=k * 4,
        with_payload=True,
    )

    best: dict[tuple, dict] = {}
    for point in response.points:
        m = point.payload
        score = point.score
        key = (m.get("source"), m.get("chunk_index"))
        if key not in best or score > best[key]["score"]:
            best[key] = {
                "score": score,
                "source": m.get("source"),
                "chunk_index": m.get("chunk_index"),
                "matched_kind": m.get("kind"),
                "matched_text": m.get("text"),
                "original": m.get("original", m.get("text")),
            }
    ranked = sorted(best.values(), key=lambda r: r["score"], reverse=True)

    if settings.apply_reranking:
        return rerank(query, ranked[:k], top_n=top_n)
    else:
        return ranked[:k]

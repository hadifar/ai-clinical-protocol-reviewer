from __future__ import annotations

from core.config import settings
from core.embeddings import embed_dense, embed_sparse
from core.llm import generate_structured
from core.prompts import RERANK_PROMPT
from core.text_utils import truncate_tokens
from core.vectorstore import DENSE, SPARSE, get_client
from schemas.ai_types import RelevanceScoreResponse


def expand_query(query: str) -> list[str]:
    # TODO: mock QE, more info is needed
    q = query.strip()
    return [q]


def rerank_score(query: str, text: str) -> int:
    section = truncate_tokens(text.strip(), settings.max_tokens)
    prompt = RERANK_PROMPT.format(query=query.strip(), section=section)
    response = generate_structured(prompt, RelevanceScoreResponse)
    return response.relevance


def rerank(query: str, results: list[dict], top_n: int | None = None) -> list[dict]:
    reranked = [
        {
            **r,
            "rerank_score": rerank_score(
                query, r.get("section") or r.get("matched_text", "")
            ),
        }
        for r in results
    ]
    reranked.sort(key=lambda r: (r["rerank_score"], r["score"]), reverse=True)
    return reranked[:top_n] if top_n is not None else reranked


def search(query: str, k: int = 12, top_n=5) -> list[dict]:
    from qdrant_client.models import Fusion, FusionQuery, Prefetch

    queries = expand_query(query)
    dense = embed_dense(queries)
    sparse = embed_sparse(queries)
    prefetch = []
    for d, s in zip(dense, sparse, strict=True):
        prefetch.append(Prefetch(query=d, using=DENSE, limit=k * 4))
        prefetch.append(Prefetch(query=s, using=SPARSE, limit=k * 4))
    response = get_client().query_points(
        settings.qdrant_collection,
        prefetch=prefetch,
        query=FusionQuery(fusion=Fusion.RRF),
        limit=k * 4,
        with_payload=True,
    )

    best: dict[tuple, dict] = {}
    for point in response.points:
        m = point.payload or {}
        score = point.score
        key = (m.get("source"), m.get("chunk_index"))
        if key not in best or score > best[key]["score"]:
            best[key] = {
                "score": score,
                "source": m.get("source"),
                "chunk_index": m.get("chunk_index"),
                "matched_kind": m.get("kind"),
                "matched_text": m.get("text"),
                "section": m.get("section", m.get("text")),
                "summary": m.get("summary"),
                "title": m.get("title"),
            }
    ranked = sorted(best.values(), key=lambda r: r["score"], reverse=True)

    if settings.apply_reranking:
        return rerank(query, ranked[:k], top_n=top_n)
    else:
        return ranked[:k]

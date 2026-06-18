from __future__ import annotations

from core.config import settings
from core.embeddings import embed_dense, embed_sparse
from core.vectorstore import DENSE, SPARSE, get_client


def search(query: str, k: int = 5) -> list[dict]:
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
    return ranked[:k]

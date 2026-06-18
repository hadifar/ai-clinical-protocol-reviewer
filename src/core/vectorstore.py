from __future__ import annotations

from functools import lru_cache

from core.config import settings

DENSE = "dense"
SPARSE = "sparse"


@lru_cache(maxsize=1)
def get_client():
    from qdrant_client import QdrantClient

    return QdrantClient(path=str(settings.qdrant_path))


def ensure_collection() -> None:
    from qdrant_client.models import (
        Distance,
        SparseVectorParams,
        VectorParams,
    )

    client = get_client()
    if not client.collection_exists(settings.qdrant_collection):
        dim = settings.embed_dimension
        client.create_collection(
            collection_name=settings.qdrant_collection,
            vectors_config={DENSE: VectorParams(size=dim, distance=Distance.COSINE)},
            sparse_vectors_config={SPARSE: SparseVectorParams()},
        )


def vector_count() -> int:
    return get_client().count(settings.qdrant_collection).count


def index_exists() -> bool:
    try:
        client = get_client()
        return (
            client.collection_exists(settings.qdrant_collection) and vector_count() > 0
        )
    except Exception:
        return False


def source_indexed(source: str) -> int:
    from qdrant_client.models import FieldCondition, Filter, MatchValue

    try:
        client = get_client()
        if not client.collection_exists(settings.qdrant_collection):
            return 0
        return client.count(
            settings.qdrant_collection,
            count_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value=source))]
            ),
        ).count
    except Exception:
        return 0

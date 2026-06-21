from __future__ import annotations

import streamlit as st

from core.config import settings

DENSE = "dense"
SPARSE = "sparse"


@st.cache_resource
def get_client():
    """Return a process-wide singleton Qdrant client.

    Local (file-based) Qdrant allows only one client per storage folder.
    ``st.cache_resource`` shares this single instance across every page,
    session, and rerun, preventing the "already accessed by another
    instance" lock error in the multipage app.
    """
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


def get_chunk(chunk_index: int, source: str | None = None) -> str | None:
    from qdrant_client.models import Condition, FieldCondition, Filter, MatchValue

    client = get_client()
    if not client.collection_exists(settings.qdrant_collection):
        return None
    must: list[Condition] = [
        FieldCondition(key="chunk_index", match=MatchValue(value=chunk_index)),
        FieldCondition(key="kind", match=MatchValue(value="chunk")),
    ]
    if source:
        must.append(FieldCondition(key="source", match=MatchValue(value=source)))
    points, _ = client.scroll(
        settings.qdrant_collection,
        scroll_filter=Filter(must=must),
        limit=1,
        with_payload=True,
    )
    if not points:
        return None
    payload = points[0].payload

    if payload:
        return payload.get("section", payload.get("text"))
    else:
        return None


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

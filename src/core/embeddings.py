from __future__ import annotations

from functools import lru_cache

from core.config import settings


@lru_cache(maxsize=1)
def get_dense_model():
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=settings.embed_model)


def embed_dense(texts: list[str]) -> list[list[float]]:
    model = get_dense_model()
    return [e.tolist() for e in model.embed(texts)]


@lru_cache(maxsize=1)
def get_sparse_model():
    from fastembed import SparseTextEmbedding

    return SparseTextEmbedding(model_name=settings.sparse_model)


def embed_sparse(texts: list[str]):
    from qdrant_client.models import SparseVector

    model = get_sparse_model()
    return [
        SparseVector(indices=e.indices.tolist(), values=e.values.tolist())
        for e in model.embed(texts)
    ]

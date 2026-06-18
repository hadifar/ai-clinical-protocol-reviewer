from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from config import settings
from models import Query, Relevance

_TOKENS_PER_WORD = 0.7
_DENSE = "dense"
_SPARSE = "sparse"

_RERANK_PROMPT = """You are ranking sections of a clinical trial protocol by how useful each one is for extracting a specific piece of information.

INFORMATION NEED:
{query}

Rate how relevant the section below is for extracting the information need above, using an integer score from 0 to 10:
- 0 = the section is unrelated and contains none of the information needed.
- 10 = the section directly contains the information to be extracted.

Judge only whether this section helps extract the information need. Do not reward generic topical overlap.

Return ONLY valid JSON in the following format:
{{"relevance": <integer 0-10>}}

SECTION:
{section}

RELEVANCE:
"""

_QUERY_PROMPT = """You are tasked with generating a search query for a given section of a clinical trial protocol document.
The query must be a single concise sentence that reflects the main idea of the section (usually denoted by ##).
T
Requirements:
- Use only information explicitly present in the provided section.
- Do not add, infer, or assume any new information.
- Do not include explanations, comments, or extra text.


Return ONLY valid JSON in the following format:
{{"query": "<generated query>"}}

SECTION:
{section}


GENERATED QUERY:
"""
# _QUERY_PROMPT = (
#     "You tasked with generating a search query for a section of a clinical trial protocol.\n"
#     "Query must be one sentence."
#     "Do not add any new information or context that is not present in the section. "
#     'Only return the query in JSON format as `{{"query": "..."}}`. '
#     "No explanation or extra text.\n\n"
#     "SECTION:\n{section}\n\nGENERATED SEARCH QUERY:"


# )

def pdf_to_markdown(pdf_path: str | Path) -> str:
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    return result.document.export_to_markdown()


def convert_pdf(pdf_path: str | Path) -> tuple[str, Path, bool]:
    md_path = settings.output_dir / f"{Path(pdf_path).stem}.md"
    if md_path.exists():
        return md_path.read_text(), md_path, True
    raw_md = pdf_to_markdown(pdf_path)
    md_path.write_text(raw_md)
    return raw_md, md_path, False


def split_chunks(raw_file: str) -> list[str]:
    sections = re.findall(r"^##.*?(?=^##|\Z)", raw_file, re.DOTALL | re.MULTILINE)

    chunks: list[str] = []
    pending = ""
    for section in sections:
        body = section.partition("\n")[2]
        if body.strip():
            chunks.append(pending + section)
            pending = ""
        else:
            pending += section
    if pending:
        chunks.append(pending)
    return chunks


def truncate_tokens(text: str, max_tokens: int) -> str:
    words = text.split()
    max_words = int(max_tokens / _TOKENS_PER_WORD)
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])





@lru_cache(maxsize=1)
def get_ollama():
    import ollama

    return ollama.Client(host=settings.ollama_base_url)


def generate_query(section: str) -> str:
    client = get_ollama()
    section = truncate_tokens(section.strip(), settings.max_tokens)
    prompt = _QUERY_PROMPT.format(section=section)
    resp = client.generate(
        model=settings.ollama_model,
        prompt=prompt,
        options={"num_ctx": settings.ollama_num_ctx},
        format=Query.model_json_schema(),
    )
    text = Query.model_validate_json(resp.response).query

    return text.strip()


def _queries_path(source: str) -> Path:
    return settings.output_dir / f"{Path(source).stem}.queries.json"


def load_queries(source: str) -> list[str] | None:
    path = _queries_path(source)
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_queries(source: str, queries: list[str]) -> None:
    _queries_path(source).write_text(json.dumps(queries, ensure_ascii=False, indent=2))


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


@lru_cache(maxsize=1)
def get_client():
    from qdrant_client import QdrantClient

    return QdrantClient(path=str(settings.qdrant_path))


def _ensure_collection() -> None:
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
            vectors_config={_DENSE: VectorParams(size=dim, distance=Distance.COSINE)},
            sparse_vectors_config={_SPARSE: SparseVectorParams()},
        )


def index_exists() -> bool:
    try:
        client = get_client()
        return (
            client.collection_exists(settings.qdrant_collection)
            and client.count(settings.qdrant_collection).count > 0
        )
    except Exception:
        return False


def build_documents(chunks: list[str], queries: list[str], source: str) -> list[dict]:
    docs: list[dict] = []
    limit = settings.max_tokens
    for i, (chunk, query) in enumerate(zip(chunks, queries, strict=False)):
        meta = {"chunk_index": i, "source": source, "original": chunk}
        docs.append({**meta, "kind": "chunk", "text": truncate_tokens(chunk, limit)})
        if query:
            docs.append(
                {**meta, "kind": "query", "text": truncate_tokens(query, limit)}
            )
    return docs


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


def index_documents(
    docs: list[dict],
    source: str | None = None,
    batch_size: int = 16,
    progress_callback: Callable[[float], None] | None = None,
) -> tuple[int, bool]:
    from qdrant_client.models import PointStruct

    if not docs:
        if progress_callback:
            progress_callback(1.0)
        return 0, False

    if source is not None:
        existing = source_indexed(source)
        if existing > 0:
            if progress_callback:
                progress_callback(1.0)
            return existing, True

    _ensure_collection()
    client = get_client()
    total = 0
    for start in range(0, len(docs), batch_size):
        batch = docs[start : start + batch_size]
        texts = [d["text"] for d in batch]
        dense = embed_dense(texts)
        sparse = embed_sparse(texts)
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector={_DENSE: d, _SPARSE: s},
                payload=doc,
            )
            for doc, d, s in zip(batch, dense, sparse, strict=False)
        ]
        client.upsert(settings.qdrant_collection, points=points)
        total += len(points)
        if progress_callback:
            progress_callback(min(start + batch_size, len(docs)) / len(docs))
    return total, False



def search(query: str, k: int = 5) -> list[dict]:
    from qdrant_client.models import Fusion, FusionQuery, Prefetch

    dense = embed_dense([query])[0]
    sparse = embed_sparse([query])[0]
    response = get_client().query_points(
        settings.qdrant_collection,
        prefetch=[
            Prefetch(query=dense, using=_DENSE, limit=k * 4),
            Prefetch(query=sparse, using=_SPARSE, limit=k * 4),
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





def rerank_score(query: str, text: str) -> int:
    """Score a single chunk's relevance to the information need (0-10) with the LLM."""
    client = get_ollama()
    section = truncate_tokens(text.strip(), settings.max_tokens)
    prompt = _RERANK_PROMPT.format(query=query.strip(), section=section)
    resp = client.generate(
        model=settings.ollama_model,
        prompt=prompt,
        options={"num_ctx": settings.ollama_num_ctx},
        format=Relevance.model_json_schema(),
    )
    return Relevance.model_validate_json(resp.response).relevance


def rerank(query: str, results: list[dict], top_n: int | None = None) -> list[dict]:
    reranked = [
        {**r, "rerank_score": rerank_score(query, r.get("original") or r.get("matched_text", ""))}
        for r in results
    ]
    reranked.sort(key=lambda r: (r["rerank_score"], r["score"]), reverse=True)
    return reranked[:top_n] if top_n is not None else reranked

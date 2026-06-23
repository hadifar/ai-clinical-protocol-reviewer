from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable
from pathlib import Path

from core.config import settings
from core.embeddings import embed_dense, embed_sparse
from core.llm import generate_structured
from core.prompts import DOC2QUERY_PROMPT
from core.text_utils import extract_titles, truncate_tokens
from core.vectorstore import ensure_collection, get_client, source_indexed
from schemas.ai_types import GeneratedSummaryResponse


def convert_pdf(pdf_path: str | Path) -> tuple[str, Path, bool]:
    from docling.document_converter import DocumentConverter

    md_path = settings.output_dir / f"{Path(pdf_path).stem}.md"
    if md_path.exists():
        return md_path.read_text(), md_path, True

    converter = DocumentConverter()
    result = converter.convert(str(pdf_path))
    raw_md = result.document.export_to_markdown()
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


def generate_query(section: str) -> str:
    section = truncate_tokens(section.strip(), settings.max_tokens)
    title = " ".join(extract_titles(section)).strip()
    prompt = DOC2QUERY_PROMPT.format(title=title, section=section)
    response = generate_structured(prompt, GeneratedSummaryResponse)
    return response.summary.strip()


def _queries_path(source: str) -> Path:
    return settings.output_dir / f"{Path(source).stem}.queries.json"


def load_queries(source: str) -> list[str] | None:
    path = _queries_path(source)
    if path.exists():
        return json.loads(path.read_text())
    return None


def save_queries(source: str, queries: list[str]) -> None:
    _queries_path(source).write_text(json.dumps(queries, ensure_ascii=False, indent=2))


def build_documents(chunks: list[str], queries: list[str], source: str) -> list[dict]:
    docs: list[dict] = []
    for i, (chunk, query) in enumerate(zip(chunks, queries, strict=False)):
        title = " ".join(extract_titles(chunk)).strip()
        meta = {
            "chunk_index": i,
            "source": source,
            "section": chunk,
            "summary": query,
            "title": title,
        }

        docs.append(
            {
                **meta,
                "kind": "chunk",
                "text": truncate_tokens(chunk, settings.max_tokens),
            }
        )
        if title:
            docs.append(
                {
                    **meta,
                    "kind": "title",
                    "text": truncate_tokens(title, settings.max_tokens),
                }
            )
        if query:
            docs.append(
                {
                    **meta,
                    "kind": "query",
                    "text": truncate_tokens(query, settings.max_tokens),
                }
            )
    return docs


def ingest_pdf(
    pdf_path: str | Path,
    source: str | None = None,
    on_status: Callable[[str], object] | None = None,
    on_query_progress: Callable[[float], object] | None = None,
    on_index_progress: Callable[[float], object] | None = None,
) -> dict:
    """Run the full ingestion pipeline — the single source of the ingest steps.

    Used by both the HTTP API (no callbacks) and the web UI (which passes
    callbacks to drive its status log and progress bars). Reuses the on-disk
    caching for Markdown, doc2query, and the index. The returned dict also
    carries ``chunks``/``queries`` for the UI preview; the API ignores them.
    """
    status = on_status or (lambda _msg: None)
    source = source or Path(pdf_path).name

    raw_md, md_path, md_cached = convert_pdf(pdf_path)
    status(
        f"Loaded existing Markdown from {md_path} (skipped docling)"
        if md_cached
        else f"Converted with docling, saved to {md_path}"
    )

    chunks = split_chunks(raw_md)
    status(f"Split into {len(chunks)} section chunks")

    cached_queries = load_queries(source)
    queries_cached = cached_queries is not None and len(cached_queries) == len(chunks)
    if queries_cached and cached_queries is not None:
        queries = cached_queries
        status(f"Loaded {len(queries)} cached queries (skipped Ollama)")
    else:
        status("Generating searchable queries with Ollama…")
        queries = []
        for i, chunk in enumerate(chunks):
            queries.append(generate_query(chunk))
            if on_query_progress:
                on_query_progress((i + 1) / len(chunks))
        save_queries(source, queries)

    docs = build_documents(chunks, queries, source=source)
    status("Embedding & indexing chunks + queries in Qdrant…")
    n_vectors, already_indexed = index_documents(
        docs, source=source, progress_callback=on_index_progress
    )
    if already_indexed:
        status(f"Loaded existing index for {source} ({n_vectors} vectors)")

    return {
        "source": source,
        "markdown_path": str(md_path),
        "markdown_cached": md_cached,
        "n_chunks": len(chunks),
        "queries_cached": queries_cached,
        "n_vectors": n_vectors,
        "already_indexed": already_indexed,
        "chunks": chunks,
        "queries": queries,
    }


def index_documents(
    docs: list[dict],
    source: str | None = None,
    batch_size: int = 16,
    progress_callback: Callable[[float], object] | None = None,
) -> tuple[int, bool]:
    from qdrant_client.models import PointStruct

    from core.vectorstore import DENSE, SPARSE

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

    ensure_collection()
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
                vector={DENSE: d, SPARSE: s},
                payload=doc,
            )
            for doc, d, s in zip(batch, dense, sparse, strict=False)
        ]
        client.upsert(settings.qdrant_collection, points=points)
        total += len(points)
        if progress_callback:
            progress_callback(min(start + batch_size, len(docs)) / len(docs))
    return total, False

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
from models.schemas import Summary


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
    response = generate_structured(prompt, Summary)
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

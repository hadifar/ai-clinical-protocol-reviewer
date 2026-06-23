from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class IndexStatus(BaseModel):
    exists: bool
    collection: str
    vector_count: int


class IngestResponse(BaseModel):
    # ingest_pdf() also returns chunks/queries for the UI preview; drop them here.
    model_config = ConfigDict(extra="ignore")

    source: str
    markdown_path: str
    markdown_cached: bool
    n_chunks: int
    queries_cached: bool
    n_vectors: int
    already_indexed: bool


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=10, ge=1, le=50)


class SearchResult(BaseModel):
    # Ranking returns extra keys depending on settings; ignore the unknown ones.
    model_config = ConfigDict(extra="ignore")

    score: float
    source: str | None = None
    chunk_index: int | None = None
    matched_kind: str | None = None
    matched_text: str | None = None
    section: str | None = None
    summary: str | None = None
    title: str | None = None
    rerank_score: int | None = None


class SearchResponse(BaseModel):
    query: str
    count: int
    results: list[SearchResult]


class ExtractRequest(BaseModel):
    attribute_key: str = Field(min_length=1)
    include_trace: bool = False


class TraceEntry(BaseModel):
    type: Literal["tool_call", "tool_result", "ai"]
    name: str | None = None
    args: dict[str, Any] | None = None
    content: str | None = None


class ExtractResponse(BaseModel):
    attribute_key: str
    info: str
    cited_chunk_indices: list[int]
    trace: list[TraceEntry] | None = None

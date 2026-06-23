from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from core.config import settings
from core.constants import TARGET_ATTRIBUTES
from core.vectorstore import index_exists, vector_count
from schemas.api_types import (
    ExtractRequest,
    ExtractResponse,
    IndexStatus,
    IngestResponse,
    SearchRequest,
    SearchResponse,
    SearchResult,
    TraceEntry,
)
from services import ingestion_service
from services.agent_service import invoke_agent
from services.ranking_service import search

app = FastAPI(
    title="AI Clinical Protocol Reviewer API",
    description=(
        "HTTP API over the same ingestion / search / extraction services that "
        "power the Streamlit app. Lets other systems index protocols and pull "
        "structured information programmatically."
    ),
    version="1.0.0",
)

# Local prototype: allow any origin so scripts / notebooks can call it freely.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _require_index() -> None:
    if not index_exists():
        raise HTTPException(
            status_code=409,
            detail="No index found yet. Ingest a PDF first via POST /ingest.",
        )


def _serialize_trace(messages: list) -> list[TraceEntry]:
    entries: list[TraceEntry] = []
    for msg in messages:
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                entries.append(
                    TraceEntry(type="tool_call", name=call["name"], args=call["args"])
                )
        elif getattr(msg, "type", None) == "tool":
            entries.append(
                TraceEntry(type="tool_result", name=msg.name, content=msg.content)
            )
        elif getattr(msg, "type", None) == "ai" and msg.content:
            entries.append(TraceEntry(type="ai", content=msg.content))
    return entries


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/attributes")
def attributes() -> dict[str, str]:
    """The target attributes the IE agent can extract (key -> human label)."""
    return TARGET_ATTRIBUTES


@app.get("/index", response_model=IndexStatus)
def index_status() -> IndexStatus:
    exists = index_exists()
    return IndexStatus(
        exists=exists,
        collection=settings.qdrant_collection,
        vector_count=vector_count() if exists else 0,
    )


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: Annotated[UploadFile, File()]) -> IngestResponse:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    pdf_path = settings.upload_dir / Path(filename).name
    pdf_path.write_bytes(await file.read())

    try:
        summary = ingestion_service.ingest_pdf(pdf_path)
    except Exception as exc:  # noqa: BLE001 - surface pipeline errors to the client
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc

    return IngestResponse(**summary)


@app.post("/search", response_model=SearchResponse)
def search_endpoint(req: SearchRequest) -> SearchResponse:
    _require_index()
    results = search(req.query, k=req.k)
    return SearchResponse(
        query=req.query,
        count=len(results),
        results=[SearchResult(**r) for r in results],
    )


@app.post("/extract", response_model=ExtractResponse)
def extract(req: ExtractRequest) -> ExtractResponse:
    _require_index()
    if req.attribute_key not in TARGET_ATTRIBUTES:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown attribute_key '{req.attribute_key}'. "
                "See GET /attributes for valid keys."
            ),
        )

    result, messages = invoke_agent(req.attribute_key)
    return ExtractResponse(
        attribute_key=req.attribute_key,
        info=result[req.attribute_key],
        cited_chunk_indices=result["cited_chunk_indices"],
        trace=_serialize_trace(messages) if req.include_trace else None,
    )

# AI Clinical Protocol Reviewer

AI-driven tooling to improve clinical trial data quality and review.

A web app that ingests clinical trial protocol documents, makes them
searchable, and runs an LLM agent to extract structured information from them.

## Pipelines

- **📥 Ingestion** — parse documents (Docling), chunk them, and index into Qdrant
  with dense + sparse (BM25) embeddings.
- **🔍 Search** — hybrid semantic search over indexed chunks with optional reranking.
- **🤖 IE Agent** — a tool-using LLM (Ollama) that searches/reads chunks to extract
  protocol attributes as structured output.

## Stack

Python 3.12 · NiceGUI · FastAPI · LangChain / LangGraph · Ollama · Qdrant · FastEmbed · Docling

## Project layout

```
src/
  app.py        # entrypoint — mounts the NiceGUI UI onto the FastAPI app
  config.py     # app-wide settings (pydantic-settings, overridable via .env)
  adapters/     # thin clients for external services: Qdrant, FastEmbed, Ollama
  services/     # pipeline logic: ingestion, ranking/search, IE agent
                #   (+ prompts.py and text_utils.py, used only by these)
  schemas/      # pydantic data contracts (LLM I/O and HTTP request/response)
  api/          # FastAPI HTTP layer
  ui/           # NiceGUI pages, layout, and components
```

Dependency flow is one-way: `ui` / `api` → `services` → `adapters`, with
`config` and `schemas` shared across layers.

## Prerequisites

- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) for the local LLM backend

## Setup

```bash
uv sync                      # install dependencies
ollama serve                 # start the local LLM backend
ollama pull qwen2.5          # pull a tool-calling capable model (see note below)
uv run python -m uvicorn app:app --app-dir src
```

This serves the UI on `http://localhost:8000` (`/` ingestion, `/search`,
`/agent`). The HTTP API is served by the same process — see `/docs` for the
interactive reference.

## Configuration

Defaults live in [src/config.py](src/config.py) and can be overridden
via a `.env` file.

### Tracing (optional)

LLM calls (the extraction agent, doc2query, and reranking) can be traced with a
self-hosted [Langfuse](https://langfuse.com). Tracing turns on only when both
keys are set; otherwise it is a no-op. Add to `.env`:

```dotenv
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=http://localhost:3000   # default; your local Langfuse instance
```

## Contributing

Branching, commit conventions, and the release flow are documented in
[CONTRIBUTING.md](CONTRIBUTING.md). In short: branch off `main`, use
[Conventional Commits](https://www.conventionalcommits.org/) (they drive
automatic versioning), and squash-merge PRs with a conventional title.

## Disclaimer

This is a research/engineering prototype for exploring AI-assisted protocol
review. It is **not** a validated medical software and must **not** be used for
clinical, diagnostic, or regulatory decisions. Always verify extracted
information against the source document.


## Contact
Drop me an email, if you have any question.<br>
firstname.lastname@gmail.com

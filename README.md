# AI Clinical Protocol Reviewer

AI-driven tooling to improve clinical trial data quality and review.

A web app (NiceGUI) that ingests clinical trial protocol documents, makes them
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

The IE agent relies on tool calling + structured output, so the configured
model must support both. `qwen2.5` is a good default; adjust `OLLAMA_MODEL` if
you prefer another tool-capable model.

## Configuration

Defaults live in [src/config.py](src/config.py) and can be overridden
via a `.env` file. Copy [.env.example](.env.example) to `.env` and edit as needed.

## Disclaimer

This is a research/engineering prototype for exploring AI-assisted protocol
review. It is **not** a validated medical software and must **not** be used for
clinical, diagnostic, or regulatory decisions. Always verify extracted
information against the source document.


## Slide
Slide can be seen [here](https://docs.google.com/presentation/d/1nQb8Hczir33vj2YkuShekdg5CpwsSy-QnWbW3bUMzco/edit?usp=sharing)!


## Contact
Drop me an email, if you have any question.<br>
firstname.lastname@gmail.com

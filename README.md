# Cluepoints Challenge

AI-driven tooling to improve clinical trial data quality and review.

A Streamlit app that ingests clinical trial protocol documents, makes them
searchable, and runs an LLM agent to extract structured information from them.

## Pipelines

- **📥 Ingestion** — parse documents (Docling), chunk them, and index into Qdrant
  with dense + sparse (BM25) embeddings.
- **🔍 Search** — hybrid semantic search over indexed chunks with optional reranking.
- **🤖 IE Agent** — a tool-using LLM (Ollama) that searches/reads chunks to extract
  protocol attributes as structured output.

## Stack

Python 3.12 · Streamlit · LangChain / LangGraph · Ollama · Qdrant · FastEmbed · Docling

## Setup

```bash
uv sync                      # install dependencies
ollama serve                 # local LLM backend
uv run streamlit run src/app.py
```

Configuration (models, Ollama URL, collection name, etc.) lives in
[src/core/config.py](src/core/config.py) and can be overridden via a `.env` file.


## Slide
Slide can be seen [here](https://docs.google.com/presentation/d/1nQb8Hczir33vj2YkuShekdg5CpwsSy-QnWbW3bUMzco/edit?usp=sharing)!


## Contact
Drop me an email, if you have any question.<br>
firstname.lastname@gmail.com

from __future__ import annotations

import time

import streamlit as st

from core.config import settings
from core.vectorstore import source_indexed
from services import ingestion_service


def render() -> None:
    st.subheader("Upload & index a protocol PDF")
    uploaded = st.file_uploader("PDF file", type=["pdf"])

    if not (uploaded and st.button("Run ingestion", type="primary")):
        return

    pdf_path = settings.upload_dir / uploaded.name
    pdf_path.write_bytes(uploaded.getbuffer())

    with st.status("Running pipeline...", expanded=True) as status:
        t0 = time.time()
        raw_md, md_path, cached = ingestion_service.convert_pdf(pdf_path)
        if cached:
            st.write(f"Loaded existing Markdown from `{md_path}` (skipped docling)")
        else:
            st.write(
                f"Converted with docling, saved to `{md_path}` "
                f"({time.time() - t0:.1f}s)"
            )

        chunks = ingestion_service.split_chunks(raw_md)
        st.write(f"Split into {len(chunks)} section chunks")

        queries = ingestion_service.load_queries(uploaded.name)
        if queries is not None and len(queries) == len(chunks):
            st.write(f"Loaded {len(queries)} cached queries (skipped Ollama)")
        else:
            st.write("Generating searchable queries with Ollama...")
            queries = []
            progress = st.progress(0.0)

            for i, chunk in enumerate(chunks):
                queries.append(ingestion_service.generate_query(chunk))
                progress.progress((i + 1) / len(chunks))

            ingestion_service.save_queries(uploaded.name, queries)

        docs = ingestion_service.build_documents(chunks, queries, source=uploaded.name)
        if source_indexed(uploaded.name):
            n, _ = ingestion_service.index_documents(docs, source=uploaded.name)
            st.write(
                f"Loaded existing index for `{uploaded.name}` "
                f"({n} vectors, skipped embedding)"
            )
        else:
            st.write("Embedding & indexing chunks + queries in Qdrant...")
            index_progress = st.progress(0.0)
            n, _ = ingestion_service.index_documents(
                docs,
                source=uploaded.name,
                progress_callback=index_progress.progress,
            )
        status.update(
            label=f"Done - {n} vectors from {len(chunks)} chunks",
            state="complete",
        )

    with st.expander("Preview chunks & generated queries"):
        for i, (c, q) in enumerate(zip(chunks, queries, strict=False)):
            st.markdown(f"**Chunk {i}** — doc2query: _{q}_")
            st.code(c[:1000] + ("..." if len(c) > 1000 else ""))

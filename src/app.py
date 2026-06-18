from __future__ import annotations

import time

import streamlit as st

import pipeline as pl
from config import settings

st.set_page_config(page_title="Clinical Trial Ingestion", layout="wide")
st.title("Clinical Trial Document Ingestion")

ingest_tab, search_tab = st.tabs(["Ingest PDF", "Search"])


with ingest_tab:
    st.subheader("Upload & index a protocol PDF")
    uploaded = st.file_uploader("PDF file", type=["pdf"])

    if uploaded and st.button("Run ingestion", type="primary"):
        pdf_path = settings.upload_dir / uploaded.name
        pdf_path.write_bytes(uploaded.getbuffer())

        with st.status("Running pipeline...", expanded=True) as status:
            t0 = time.time()
            raw_md, md_path, cached = pl.convert_pdf(pdf_path)
            if cached:
                st.write(f"Loaded existing Markdown from `{md_path}` (skipped docling)")
            else:
                st.write(
                    f"Converted with docling, saved to `{md_path}` "
                    f"({time.time() - t0:.1f}s)"
                )

            chunks = pl.split_chunks(raw_md)
            st.write(f"Split into {len(chunks)} section chunks")

            queries = pl.load_queries(uploaded.name)
            if queries is not None and len(queries) == len(chunks):
                st.write(f"Loaded {len(queries)} cached queries (skipped Ollama)")
            else:
                st.write("Generating searchable queries with Ollama...")
                queries = []
                progress = st.progress(0.0)

                for i, chunk in enumerate(chunks):
                    q = pl.generate_query(chunk)
                    # print(f"Chunk: {chunk}\n\n query: {q}\n\n")
                    queries.append(q)
                    progress.progress((i + 1) / len(chunks))

                pl.save_queries(uploaded.name, queries)

            docs = pl.build_documents(chunks, queries, source=uploaded.name)
            if pl.source_indexed(uploaded.name):
                n, cached_index = pl.index_documents(docs, source=uploaded.name)
                st.write(
                    f"Loaded existing index for `{uploaded.name}` "
                    f"({n} vectors, skipped embedding)"
                )
            else:
                st.write("Embedding & indexing chunks + queries in Qdrant...")
                index_progress = st.progress(0.0)
                n, cached_index = pl.index_documents(
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
                st.markdown(f"**Chunk {i}** — query: _{q}_")
                st.code(c[:1000] + ("..." if len(c) > 500 else ""))


with search_tab:
    st.subheader("Search the index")
    if not pl.index_exists():
        st.info("No index found yet. Ingest a PDF first.")
    else:
        st.success(
            f"Index loaded: "
            f"{pl.get_client().count(settings.qdrant_collection).count} vectors "
            f"in collection `{settings.qdrant_collection}`."
        )
        query = st.text_input("Query", placeholder="e.g. primary efficacy endpoint")
        if query:
            candidates = pl.search(query, k=10)
            results = pl.rerank(query, candidates, top_n=5)

            st.caption(f"Top {len(results)} results")
            for rank, r in enumerate(results, 1):
                with st.container(border=True):
                    st.markdown(
                        f"**#{rank}** · score `{r['score']:.4f}` · "
                        f"rerank-score: {r['rerank_score']} . "
                        f"matched **{r['matched_kind']}** · "
                        f"chunk {r['chunk_index']} "
                    )

                    if r["matched_kind"] == "query":
                        st.caption(f"matched query: {r['matched_text']}")

                    st.markdown(r["original"])

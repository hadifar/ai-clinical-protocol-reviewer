from __future__ import annotations

import streamlit as st

from core.config import settings
from core.vectorstore import index_exists, vector_count
from services.rank import search


def render() -> None:
    st.subheader("Search the index")
    if not index_exists():
        st.info("No index found yet. Ingest a PDF first.")
        return

    st.success(
        f"Index loaded: {vector_count()} vectors "
        f"in collection `{settings.qdrant_collection}`."
    )
    query = st.text_input("Query", placeholder="e.g. primary efficacy endpoint")
    if not query:
        return

    results = search(query, k=10)

    st.caption(f"Top {len(results)} results")
    for rank, r in enumerate(results, 1):
        with st.container(border=True):
            st.markdown(
                f"**#{rank}** · score `{r['score']:.4f}` · "
                f"rerank-score: {r.get('rerank_score', '-')} . "
                f"matched **{r['matched_kind']}** · "
                f"chunk {r['chunk_index']} "
            )

            if r["matched_kind"] == "query":
                st.caption(f"matched query: {r['matched_text']}")

            st.markdown(r["original"])

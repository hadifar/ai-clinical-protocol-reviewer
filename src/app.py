from __future__ import annotations

import streamlit as st

from ui import ingest_view, search_view

st.set_page_config(page_title="Clinical Trial Ingestion", layout="wide")
st.title("Clinical Trial Document Ingestion")

ingest_tab, search_tab = st.tabs(["Ingest PDF", "Search"])

with ingest_tab:
    ingest_view.render()

with search_tab:
    search_view.render()

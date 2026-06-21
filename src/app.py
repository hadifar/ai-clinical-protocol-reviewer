from __future__ import annotations

import streamlit as st

from ui import sidebar_comp

st.set_page_config(page_title="Clinical Trial Ingestion", layout="wide")
st.title("Clinical Trial Document Analysis")

sidebar_comp.render()

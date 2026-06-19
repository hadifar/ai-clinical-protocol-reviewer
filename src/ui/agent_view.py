from __future__ import annotations

import streamlit as st

from core.vectorstore import index_exists
from services.extraction_agent import invoke_agent


def _render_trace(messages: list) -> None:
    """Show the agent's tool calls and observations."""
    with st.expander("Agent reasoning trace", expanded=False):
        for msg in messages:
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for call in tool_calls:
                    st.markdown(f"🛠️ **{call['name']}**(`{call['args']}`)")
            elif msg.type == "tool":
                st.caption(f"↳ {msg.name} returned:")
                st.code(msg.content, language="text")
            elif msg.type == "ai" and msg.content:
                st.markdown(f"🤖 {msg.content}")


def render() -> None:
    st.subheader("Information extraction agent")
    st.caption(
        "Give an attribute or query. The agent searches the index "
        "(search_chunks / read_chunk) and returns the extracted information as JSON."
    )

    if not index_exists():
        st.info("No index found yet. Ingest a PDF first.")
        return

    attribute = st.text_input(
        "Attribute / query",
        placeholder="e.g. primary study objective, Schedule of Activities",
    )

    if not attribute:
        return

    with st.spinner("Agent is searching the index…"):
        info, messages = invoke_agent(attribute)

    st.markdown("**Result**")
    st.json({attribute: info})
    _render_trace(messages)

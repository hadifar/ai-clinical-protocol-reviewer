from __future__ import annotations

import streamlit as st

from core.constants import TARGET_ATTRIBUTES
from core.vectorstore import index_exists
from services.agent_service import invoke_agent


def _render_trace(messages: list) -> None:
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
                st.caption("↳ AI returned:")
                st.markdown(f"{msg.content}")


def render() -> None:
    st.subheader("Information extraction agent")
    st.caption(
        "Give an attribute. The agent searches the index "
        "(search_chunks / read_chunk) and returns the extracted information as JSON."
    )

    if not index_exists():
        st.info("No index found yet. Ingest a PDF first.")
        return

    option = st.selectbox(
        label="Select one of the following topic for ",
        options=list(TARGET_ATTRIBUTES.keys()),
        index=None,
        placeholder="",
        format_func=lambda k: TARGET_ATTRIBUTES[k],
    )

    if option:
        with st.spinner("Agent is working…"):
            structured_info, messages = invoke_agent(option)

        st.markdown("**Result**")
        st.write("---")
        st.json(structured_info)
        st.write("---")
        _render_trace(messages)

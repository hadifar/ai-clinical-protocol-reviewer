from __future__ import annotations

import streamlit as st

from core.vectorstore import index_exists
from services.ai_agent import invoke_agent

target_attributes = {
    "primary_study_objectives": "Primary study objectives",
    "secondary_study_objective": "Secondary study objectives",
    "primary_endpoints": "Primary endpoints",
    "exploratory_endpoints": "Exploratory endpoints",
    "schedule_of_activities_table": "Schedule of Activities (SoA) as a structured table",
    "inclusion_criteria": "Inclusion Criteria",
    "exclusion_criteria": "Exclusion Criteria",
    "visit_definition": "Visit Definition",
    "visit_timing": "Visit timing",
    "Key_assessments_and_procedures": "Key assessments and procedures",
    "safety_monitoring_rules": "Safety monitoring rules",
}


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
        "Give an attribute. The agent searches the index "
        "(search_chunks / read_chunk) and returns the extracted information as JSON."
    )

    if not index_exists():
        st.info("No index found yet. Ingest a PDF first.")
        return

    option = st.selectbox(
        label="Select one of the following topic for ",
        options=list(target_attributes.keys()),
        index=None,
        placeholder="",
        format_func=lambda k: target_attributes[k],
    )

    if not option:
        return

    with st.spinner("Agent is searching the index…"):
        info, messages = invoke_agent(target_attributes[option])

    st.markdown("**Result**")
    st.write("")
    st.json({target_attributes[option]: info})
    _render_trace(messages)

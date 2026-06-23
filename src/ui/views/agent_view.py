from __future__ import annotations

import json

from nicegui import run, ui

from core.constants import TARGET_ATTRIBUTES
from services.agent_service import invoke_agent
from ui.components.guards import requires_index, safe
from ui.components.trace_view import trace_view


@requires_index
def render() -> None:
    ui.label("Information extraction agent").classes("text-h6")
    ui.label(
        "Give an attribute. The agent searches the index "
        "(search_chunks / read_chunk) and returns the extracted information as JSON."
    ).classes("text-grey-7 q-mb-md")

    select = ui.select(
        options=dict(TARGET_ATTRIBUTES),
        label="Select one of the following topics",
    ).classes("w-full max-w-md")
    result_area = ui.column().classes("w-full")

    @safe
    async def run_agent() -> None:
        if not select.value:
            return
        result_area.clear()
        with result_area:
            ui.spinner(size="lg")
            ui.label("Agent is working…").classes("text-grey-7")
        structured_info, messages = await run.io_bound(invoke_agent, select.value)
        result_area.clear()

        with result_area:
            ui.markdown("**Result**")
            ui.separator()
            ui.code(json.dumps(structured_info, indent=2, ensure_ascii=False)).classes(
                "w-full"
            )
            ui.separator()
            trace_view(messages)

    ui.button("Run extraction", on_click=run_agent).classes("q-mt-sm")

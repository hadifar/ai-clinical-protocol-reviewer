from __future__ import annotations

from nicegui import ui


def trace_view(messages: list) -> None:
    """Render the IE agent's reasoning trace (tool calls, results, AI replies)."""
    with ui.expansion("Agent reasoning trace").classes("w-full"):
        for msg in messages:
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                for call in tool_calls:
                    ui.markdown(f"🛠️ **{call['name']}**(`{call['args']}`)")
            elif getattr(msg, "type", None) == "tool":
                ui.label(f"↳ {msg.name} returned:").classes("text-grey-7")
                ui.code(str(msg.content)).classes("w-full")
            elif getattr(msg, "type", None) == "ai" and msg.content:
                ui.label("↳ AI returned:").classes("text-grey-7")
                ui.markdown(str(msg.content))

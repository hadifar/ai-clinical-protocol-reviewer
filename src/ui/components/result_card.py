from __future__ import annotations

from nicegui import ui


def result_card(rank: int, r: dict) -> None:
    """Render a single hybrid-search result as a bordered card."""
    with ui.card().classes("w-full"):
        ui.markdown(
            f"**#{rank}** · score `{r['score']:.4f}` · "
            f"rerank-score: {r.get('rerank_score', '-')} · "
            f"matched **{r['matched_kind']}** · "
            f"chunk {r['chunk_index']}"
        )
        if r.get("summary"):
            ui.label(f"Doc2query: {r['summary']}").classes("text-grey-7")
        if r["matched_kind"] == "query":
            ui.label(f"matched query: {r['matched_text']}").classes("text-grey-7")
        ui.markdown(r["section"])

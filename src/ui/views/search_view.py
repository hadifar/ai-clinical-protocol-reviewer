from __future__ import annotations

from nicegui import run, ui

from adapters.qdrant import vector_count
from config import settings
from services.ranking_service import search
from ui.components.guards import requires_index, safe
from ui.components.result_card import result_card

# Number of results to request per search.
SEARCH_K = 10


@requires_index
def render() -> None:
    ui.label("Search the index").classes("text-h6 q-mb-md")
    ui.label(
        f"Index loaded: {vector_count()} vectors "
        f"in collection '{settings.qdrant_collection}'."
    ).classes("text-positive")

    query = ui.input("Query", placeholder="e.g. primary efficacy endpoint").classes(
        "w-full max-w-xl"
    )
    results_area = ui.column().classes("w-full")

    @safe
    async def do_search() -> None:
        q = (query.value or "").strip()
        results_area.clear()
        if not q:
            return
        with results_area:
            spinner = ui.spinner(size="lg")
        results = await run.io_bound(search, q, SEARCH_K)
        spinner.delete()

        with results_area:
            ui.label(f"Top {len(results)} results").classes("text-grey-7")
            for rank, r in enumerate(results, 1):
                result_card(rank, r)

    query.on("keydown.enter", do_search)
    ui.button("Search", on_click=do_search).classes("q-mt-sm")

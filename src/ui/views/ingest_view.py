from __future__ import annotations

from pathlib import Path

from nicegui import run, ui

from config import settings
from services import ingestion_service
from ui.components.guards import safe

# Max characters of each chunk shown in the preview expander.
PREVIEW_CHARS = 1000


def render() -> None:
    ui.label("Upload & index a protocol PDF").classes("text-h6 q-mb-md")

    log = ui.column().classes("w-full")
    progress = ui.linear_progress(value=0.0, show_value=False).classes("w-full")
    progress.visible = False
    preview = ui.column().classes("w-full")

    # Holds the most recently uploaded PDF until the user starts ingestion.
    uploaded: dict = {"path": None, "name": None}

    @safe
    async def handle_upload(e) -> None:
        name = Path(e.file.name).name
        pdf_path = settings.upload_dir / name
        pdf_path.write_bytes(await e.file.read())
        uploaded.update(path=pdf_path, name=name)
        start_button.enable()
        ui.notify(f"{name} ready — click Start ingestion")

    @safe
    async def handle_ingest() -> None:
        if uploaded["path"] is None:
            ui.notify("Upload a PDF first", type="warning")
            return

        pdf_path, name = uploaded["path"], uploaded["name"]
        start_button.disable()
        log.clear()
        preview.clear()
        progress.visible = True
        progress.value = 0.0

        # The pipeline runs in a worker thread, so its callbacks push updates
        # into shared state; a ui.timer syncs that state onto the widgets.
        state: dict = {"messages": [], "progress": 0.0}
        shown = 0

        def sync() -> None:
            nonlocal shown
            while shown < len(state["messages"]):
                with log:
                    ui.label(state["messages"][shown])
                shown += 1
            progress.value = state["progress"]

        timer = ui.timer(0.1, sync)
        try:
            summary = await run.io_bound(
                ingestion_service.ingest_pdf,
                pdf_path,
                source=name,
                on_status=lambda m: state["messages"].append(m),
                on_query_progress=lambda v: state.update(progress=v),
                on_index_progress=lambda v: state.update(progress=v),
            )
        finally:
            timer.cancel()
            sync()
            start_button.enable()

        progress.value = 1.0
        with log:
            ui.label(
                f"Done — {summary['n_vectors']} vectors "
                f"from {summary['n_chunks']} chunks"
            ).classes("text-positive text-bold")

        with (
            preview,
            ui.expansion("Preview chunks & generated queries").classes("w-full"),
        ):
            chunks, queries = summary["chunks"], summary["queries"]
            for i, (c, q) in enumerate(zip(chunks, queries, strict=False)):
                ui.markdown(f"**Chunk {i}** — doc2query: _{q}_")
                snippet = c[:PREVIEW_CHARS] + ("..." if len(c) > PREVIEW_CHARS else "")
                ui.code(snippet).classes("w-full")

    ui.upload(
        label="PDF file",
        auto_upload=True,
        on_upload=handle_upload,
        max_files=1,
    ).props("accept=.pdf").classes("w-full max-w-md")

    start_button = ui.button("Start PDF ingestion", on_click=handle_ingest).classes(
        "q-mt-md"
    )
    start_button.disable()

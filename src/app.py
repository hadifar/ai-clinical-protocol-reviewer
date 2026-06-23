from __future__ import annotations

from nicegui import ui

from api.server import app
from ui import layout


def _build_page(page) -> None:
    layout.page_frame(page.path)
    page.render()


# Register one NiceGUI route per page in the registry
for _page in layout.PAGES:
    ui.page(_page.path)(lambda page=_page: _build_page(page))

# Mount the NiceGUI UI onto the existing FastAPI app, so a single Uvicorn
ui.run_with(
    app,
    title="Clinical Trial Document Analysis",
    storage_secret="ai-clinical-protocol-reviewer",
)

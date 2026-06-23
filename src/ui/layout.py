from __future__ import annotations

from nicegui import ui

from ui.types import Page
from ui.views import agent_view, ingest_view, search_view

# Single source of truth for the app's pages. The navigation drawer below and
# the route registration in app.py are both built from this list, so adding a
# page is a one-line change here.
PAGES: list[Page] = [
    Page("/", "📥", "Ingestion pipeline", ingest_view.render),
    Page("/search", "🔍", "Search pipeline", search_view.render),
    Page("/agent", "🤖", "IE agent", agent_view.render),
]

_FOOTER_HTML = (
    "Developed with ❤️ by "
    '<a href="https://hadifar.github.io/" target="_blank" '
    'style="color: white; text-decoration: none;">Amir Hadifar</a>'
)


def page_frame(active: str) -> None:
    """Render the shared header, navigation drawer and footer.

    Call once at the top of every page; ``active`` is the path of the current
    page so its nav entry can be highlighted.
    """
    with ui.header(elevated=True).classes("items-center"):
        ui.label("Clinical Trial Document Analysis").classes("text-h6")

    with ui.left_drawer(bordered=True).classes("bg-grey-1"):
        ui.label("Pipelines").classes("text-bold text-grey-7 q-mb-sm")
        for page in PAGES:
            link = ui.link(f"{page.icon}  {page.title}", page.path).classes(
                "block q-py-xs no-underline text-grey-9"
            )
            if page.path == active:
                link.classes("text-weight-bold text-primary")

    with ui.footer().classes("justify-center"):
        ui.html(_FOOTER_HTML)

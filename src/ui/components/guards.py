from __future__ import annotations

import functools
from collections.abc import Awaitable, Callable

from nicegui import ui

from core.vectorstore import index_exists

NO_INDEX_MESSAGE = "No index found yet. Ingest a PDF first."


def requires_index(render: Callable[[], None]) -> Callable[[], None]:
    """Wrap a page ``render`` so it shows a notice instead when no index exists."""

    @functools.wraps(render)
    def wrapper() -> None:
        if not index_exists():
            ui.label(NO_INDEX_MESSAGE).classes("text-warning")
            return
        render()

    return wrapper


def safe(handler: Callable[..., Awaitable[None]]) -> Callable[..., Awaitable[None]]:
    """Wrap an async UI event handler so errors surface as a notification
    instead of failing silently."""

    @functools.wraps(handler)
    async def wrapper(*args, **kwargs) -> None:
        try:
            await handler(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - report any failure to the user
            ui.notify(f"Something went wrong: {exc}", type="negative", multi_line=True)

    return wrapper

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    """A single UI page: its route, nav presentation, and render callback."""

    path: str
    icon: str
    title: str
    render: Callable[[], None]

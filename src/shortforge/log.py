"""Shared rich console and a timed step helper for readable progress output.

Progress goes to stderr so stdout stays clean for results (and for the MCP stdio
transport in the stretch phase).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from rich.console import Console

console = Console(stderr=True, highlight=False)


@contextmanager
def step(name: str) -> Iterator[None]:
    """Log the start, duration and outcome of a pipeline step.

    Example:
        >>> with step("Script"):
        ...     ...
    """
    console.print(f"[bold cyan]>[/] {name}...")
    started = time.perf_counter()
    try:
        yield
    except Exception:
        console.print(f"[bold red]x[/] {name} failed after {time.perf_counter() - started:.1f}s")
        raise
    console.print(f"[bold green]✓[/] {name} [dim]({time.perf_counter() - started:.1f}s)[/]")

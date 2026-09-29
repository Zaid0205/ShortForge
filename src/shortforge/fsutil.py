"""Filesystem helpers that make step outputs all-or-nothing.

A step writes to a temporary sibling file and renames it only after it succeeds, so a
crash never leaves a half-written file that the cache would later treat as done.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def atomic_path(path: Path) -> Iterator[Path]:
    """Yield a temporary path next to `path`; move it into place only on success.

    The temporary name keeps the real suffix (``scene_01.tmp.wav``) so libraries that
    pick a format from the extension still work.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp{path.suffix}")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_text_atomic(path: Path, text: str) -> None:
    """Write UTF-8 text to `path` atomically."""
    with atomic_path(path) as tmp:
        tmp.write_text(text, encoding="utf-8")

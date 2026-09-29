"""Shared fixtures."""

from __future__ import annotations

import pytest

from shortforge.config import Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove ShortForge variables from the process env so tests are deterministic."""
    for name in Settings.model_fields:
        monkeypatch.delenv(name.upper(), raising=False)

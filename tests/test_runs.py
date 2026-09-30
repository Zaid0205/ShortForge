"""Run store tests: IDs, manifests and listing."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from shortforge.runs import RunError, RunStore, fingerprint, new_run_id, slugify


def test_slugify_makes_folder_safe_names() -> None:
    assert slugify("How GPUs speed up AI?") == "how-gpus-speed-up-ai"
    assert slugify("C++ & Rust: 2026!") == "c-rust-2026"
    assert slugify("!!!") == "short"
    assert len(slugify("word " * 30)) <= 40


def test_run_id_is_sortable_and_readable() -> None:
    run_id = new_run_id("What is RAG", now=datetime(2026, 10, 1, 14, 32, 10))
    assert run_id == "20261001-143210-what-is-rag"


def test_fingerprint_is_stable_and_order_sensitive() -> None:
    assert fingerprint("a", 1) == fingerprint("a", 1)
    assert fingerprint("a", "b") != fingerprint("b", "a")
    assert fingerprint("ab", "c") != fingerprint("a", "bc")


def test_create_load_and_list(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    manifest = store.create("What is RAG", 6)
    assert (tmp_path / manifest.run_id / "run.json").exists()
    assert store.load(manifest.run_id) == manifest
    assert [m.run_id for m in store.list()] == [manifest.run_id]


def test_unknown_run_lists_recent_ones(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    existing = store.create("What is RAG", 6)
    with pytest.raises(RunError, match=existing.run_id):
        store.load("20990101-000000-nope")


@pytest.mark.parametrize("run_id", ["../secrets", "Run ID", "a/b", ""])
def test_run_ids_cannot_escape_output_folder(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(RunError, match="not a valid run ID"):
        RunStore(tmp_path).folder(run_id)


def test_damaged_manifest_is_reported_and_skipped_in_list(tmp_path: Path) -> None:
    store = RunStore(tmp_path)
    manifest = store.create("What is RAG", 6)
    (tmp_path / manifest.run_id / "run.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(RunError, match="damaged"):
        store.load(manifest.run_id)
    assert store.list() == []

"""Run folders, run IDs and the per-run manifest that makes steps cacheable.

Every run lives in ``output/<run_id>/`` and keeps a `run.json` manifest next to its files.
For each cached output the manifest stores a fingerprint of the inputs that produced it
(narration and voice for audio, prompt and style for images, all scene fingerprints plus
frame settings for the video). A step reuses a file only if it exists and its fingerprint
still matches, so editing one scene's narration in `script.json` and resuming redoes that
scene's audio and the final render, and nothing else.

Run `python -m shortforge.runs` to list the runs in the output folder.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from shortforge.fsutil import write_text_atomic

MANIFEST_NAME = "run.json"
SLUG_MAX_LENGTH = 40
_RUN_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,80}$")


class RunError(RuntimeError):
    """Raised when a run folder cannot be created, found or read."""


class SceneRecord(BaseModel):
    """Cache fingerprints and results for one scene's audio and image."""

    audio_key: str | None = None
    audio_seconds: float | None = None
    image_key: str | None = None
    image_source: Literal["stock", "generated"] | None = None
    image_credit: str | None = None


class RunManifest(BaseModel):
    """Everything known about a run: inputs, cache fingerprints, timings and outcome."""

    run_id: str
    topic: str
    scenes: int
    created_at: datetime
    status: Literal["running", "done", "failed"] = "running"
    error: str | None = None
    scene_records: list[SceneRecord] = Field(default_factory=list)
    video_key: str | None = None
    video_seconds: float | None = None
    youtube_id: str | None = None
    youtube_privacy: str | None = None
    youtube_video_key: str | None = None
    """The `video_key` of the render that was uploaded, to detect a changed video."""
    step_seconds: dict[str, float] = Field(default_factory=dict)


def fingerprint(*parts: object) -> str:
    """Return a short stable hash of `parts`, used to tell whether a cached file is current."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()[:16]


def slugify(text: str, max_length: int = SLUG_MAX_LENGTH) -> str:
    """Turn a topic into a lowercase, dash-separated folder-safe slug."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_length].rstrip("-") or "short"


def new_run_id(topic: str, now: datetime | None = None) -> str:
    """Return a sortable, readable run ID such as ``20261001-143210-how-gpus-speed-up-ai``."""
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{slugify(topic)}"


class RunStore:
    """Create, find and persist runs under one output folder."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def folder(self, run_id: str) -> Path:
        """Return the folder for `run_id`, rejecting IDs that could escape the output folder."""
        if not _RUN_ID_RE.fullmatch(run_id):
            raise RunError(
                f"{run_id!r} is not a valid run ID (lowercase letters, digits and dashes only)."
            )
        return self.output_dir / run_id

    def create(self, topic: str, scenes: int) -> RunManifest:
        """Start a new run folder and write its first manifest."""
        run_id = new_run_id(topic)
        folder = self.folder(run_id)
        if folder.exists():
            raise RunError(f"Run folder {folder} already exists. Wait a second and try again.")
        manifest = RunManifest(
            run_id=run_id, topic=topic, scenes=scenes, created_at=datetime.now().astimezone()
        )
        self.save(manifest)
        return manifest

    def load(self, run_id: str) -> RunManifest:
        """Read an existing run's manifest.

        Raises:
            RunError: If the run does not exist or its manifest is unreadable, listing
                recent runs to choose from.
        """
        path = self.folder(run_id) / MANIFEST_NAME
        if not path.exists():
            recent = ", ".join(m.run_id for m in self.list()[:5]) or "none yet"
            raise RunError(f"No run {run_id!r} in {self.output_dir}. Recent runs: {recent}.")
        try:
            return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as exc:
            raise RunError(f"{path} is damaged and cannot be resumed: {exc}") from exc

    def save(self, manifest: RunManifest) -> None:
        """Write `manifest` to its run folder atomically."""
        path = self.folder(manifest.run_id) / MANIFEST_NAME
        write_text_atomic(path, manifest.model_dump_json(indent=2))

    def list(self) -> list[RunManifest]:
        """Return every readable run, newest first. Unreadable folders are skipped."""
        if not self.output_dir.is_dir():
            return []
        runs = []
        for path in self.output_dir.glob(f"*/{MANIFEST_NAME}"):
            try:
                runs.append(RunManifest.model_validate_json(path.read_text(encoding="utf-8")))
            except (OSError, ValidationError):
                continue
        return sorted(runs, key=lambda m: m.created_at, reverse=True)


def _self_check() -> None:
    """Print the runs in the configured output folder."""
    from rich.table import Table

    from shortforge.config import get_settings
    from shortforge.log import console

    store = RunStore(get_settings().output_dir)
    runs = store.list()
    if not runs:
        console.print(f"No runs in {store.output_dir} yet.")
        return
    table = Table(title=f"Runs in {store.output_dir}")
    for column in ("Run ID", "Status", "Scenes", "Video", "Topic"):
        table.add_column(column)
    colors = {"done": "green", "failed": "red", "running": "yellow"}
    for run in runs:
        video = f"{run.video_seconds:.1f}s" if run.video_seconds else "-"
        status = f"[{colors[run.status]}]{run.status}[/]"
        table.add_row(run.run_id, status, str(run.scenes), video, run.topic)
    console.print(table)


if __name__ == "__main__":
    _self_check()

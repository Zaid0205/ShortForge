"""Self-check: `python -m shortforge.images` renders the self-check script's scenes.

Reads `output/_selfcheck/script.json` (written by `python -m shortforge.script`) and saves
one frame-sized PNG per scene, from Pexels or generated depending on ``IMAGE_SOURCE``, plus
`contact_sheet.png`, a grid of all scenes for review.
"""

from __future__ import annotations

import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from pydantic import ValidationError

from shortforge.config import ConfigError, get_settings
from shortforge.fsutil import atomic_path
from shortforge.images.sourcing import SceneImage, SceneImageSource
from shortforge.log import console, step
from shortforge.models import Scene
from shortforge.script import load_script

THUMB_WIDTH, THUMB_HEIGHT = 216, 384
LABEL_HEIGHT = 36
GAP = 8


def contact_sheet(rows: list[tuple[str, list[Path]]]) -> Image.Image:
    """Lay out labelled rows of frame thumbnails into one review image."""
    columns = max(len(paths) for _, paths in rows)
    width = GAP + columns * (THUMB_WIDTH + GAP)
    height = GAP + len(rows) * (LABEL_HEIGHT + THUMB_HEIGHT + GAP)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=20)
    y = GAP
    for label, paths in rows:
        draw.text((GAP, y + 6), label, fill="black", font=font)
        y += LABEL_HEIGHT
        for index, path in enumerate(paths):
            with Image.open(path) as frame:
                thumb = frame.convert("RGB").resize((THUMB_WIDTH, THUMB_HEIGHT))
            sheet.paste(thumb, (GAP + index * (THUMB_WIDTH + GAP), y))
        y += THUMB_HEIGHT + GAP
    return sheet


def render_scenes(scenes: list[Scene], folder: Path) -> tuple[list[Path], list[SceneImage]]:
    """Produce every scene's image with the configured source, returning paths and origins."""
    source = SceneImageSource.from_settings(get_settings())
    paths, results = [], []
    for index, scene in enumerate(scenes, start=1):
        out = folder / f"scene_{index:02d}.png"
        started = time.perf_counter()
        with atomic_path(out) as tmp:
            result = source.render(scene, tmp)
        detail = result.credit or f"generated from: {scene.image_prompt[:60]}"
        console.print(
            f"  scene {index}: {result.source}, {time.perf_counter() - started:.1f}s "
            f"[dim]({scene.search_query!r}) {detail}[/]"
        )
        paths.append(out)
        results.append(result)
    return paths, results


def main() -> None:
    """Produce the scene images and write the review sheet."""
    settings = get_settings()
    folder = settings.output_dir / "_selfcheck"
    script_path = folder / "script.json"
    regenerate = 'Run python -m shortforge.script "your topic" first.'
    if not script_path.exists():
        console.print(f"[bold red]Error:[/] {script_path} not found. {regenerate}")
        raise SystemExit(1)
    try:
        scenes = load_script(script_path).scenes
    except ValidationError as exc:
        console.print(
            f"[bold red]Error:[/] {script_path} no longer passes validation "
            f"({exc.error_count()} problem(s)). {regenerate}"
        )
        raise SystemExit(1) from exc

    try:
        with step(f"Images ({settings.image_source}, {len(scenes)} scenes)"):
            paths, results = render_scenes(scenes, folder)
    except (ConfigError, RuntimeError, ValueError) as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise SystemExit(1) from exc

    stock = sum(r.source == "stock" for r in results)
    label = f"{settings.image_source}: {stock} stock photos, {len(results) - stock} generated"
    sheet_path = folder / "contact_sheet.png"
    with atomic_path(sheet_path) as tmp:
        contact_sheet([(label, paths)]).save(tmp)
    console.print(f"[green]Saved[/] {sheet_path} ({label})")


if __name__ == "__main__":
    main()

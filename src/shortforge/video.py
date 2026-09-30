"""Video assembly: Ken Burns motion, timed captions and per-scene audio into one MP4.

Each scene lasts exactly as long as its audio clip. Its image gets a slow zoom computed
from a fractional crop box (sub-pixel accurate, so no jitter from integer rounding), and its
captions are overlaid as transparent image clips. Scenes are joined with hard cuts so
audio and captions stay in sync.

Run `python -m shortforge.video` to build `output/_selfcheck/final.mp4`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from moviepy import (
    AudioFileClip,
    CompositeVideoClip,
    ImageClip,
    VideoClip,
    concatenate_videoclips,
)
from PIL import Image

from shortforge.captions import caption_position, render_caption, scene_captions
from shortforge.config import Settings

ZOOM = 1.08
"""Scale at the zoomed-in end of each scene's Ken Burns move."""
ANCHORS = [(0.5, 0.5), (0.35, 0.4), (0.65, 0.55), (0.5, 0.35), (0.4, 0.6), (0.6, 0.45)]
"""Normalized points the zoom moves toward; cycled so consecutive scenes differ."""


@dataclass(frozen=True)
class SceneAssets:
    """Everything needed to render one scene."""

    narration: str
    image: Path
    audio: Path


def ken_burns_frames(
    image: Image.Image, duration: float, index: int
) -> Callable[[float], np.ndarray]:
    """Return a frame function that zooms slowly toward this scene's anchor point.

    Even scenes zoom in, odd scenes zoom out, and the anchor cycles through `ANCHORS`.
    Keeping the anchor fixed in the frame means the crop never leaves the image.
    """
    source = image.convert("RGB")
    width, height = source.size
    anchor_x, anchor_y = ANCHORS[index % len(ANCHORS)]
    zoom_in = index % 2 == 0

    def frame(t: float) -> np.ndarray:
        progress = min(max(t / duration, 0.0), 1.0) if duration > 0 else 0.0
        if not zoom_in:
            progress = 1.0 - progress
        scale = 1.0 / (1.0 + (ZOOM - 1.0) * progress)
        x0 = anchor_x * width * (1.0 - scale)
        y0 = anchor_y * height * (1.0 - scale)
        box = (x0, y0, x0 + width * scale, y0 + height * scale)
        return np.asarray(source.resize((width, height), Image.Resampling.BICUBIC, box=box))

    return frame


def build_scene(assets: SceneAssets, index: int, settings: Settings) -> CompositeVideoClip:
    """Compose one scene: moving image, timed captions and its audio."""
    audio = AudioFileClip(str(assets.audio))
    duration = audio.duration
    with Image.open(assets.image) as raw:
        if raw.size != (settings.video_width, settings.video_height):
            raise ValueError(
                f"{assets.image.name} is {raw.size[0]}x{raw.size[1]}, expected "
                f"{settings.video_width}x{settings.video_height}"
            )
        background = VideoClip(ken_burns_frames(raw, duration, index), duration=duration)

    layers: list[VideoClip] = [background]
    for caption in scene_captions(assets.narration, duration):
        overlay = render_caption(caption.text, settings.font_path, settings.video_width)
        y = caption_position(overlay.height, settings.video_height)
        layers.append(
            ImageClip(np.asarray(overlay), transparent=True)
            .with_start(caption.start)
            .with_duration(caption.duration)
            .with_position(("center", y))
        )
    size = (settings.video_width, settings.video_height)
    return CompositeVideoClip(layers, size=size).with_duration(duration).with_audio(audio)


def assemble_video(
    scenes: list[SceneAssets], out_path: Path, settings: Settings, progress: bool = True
) -> float:
    """Render all scenes into an H.264/AAC MP4 at `out_path` and return its duration.

    Args:
        scenes: Scene assets in playback order.
        out_path: Destination ``.mp4`` file.
        settings: Frame size, fps and font come from here.
        progress: Show MoviePy's progress bar.
    """
    if not scenes:
        raise ValueError("No scenes to assemble.")
    if out_path.suffix.lower() != ".mp4":
        raise ValueError(f"Video output must be an .mp4 file, got {out_path.name}")
    clips = [build_scene(scene, index, settings) for index, scene in enumerate(scenes)]
    video = concatenate_videoclips(clips, method="chain")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        video.write_videofile(
            str(out_path),
            fps=settings.video_fps,
            codec="libx264",
            audio_codec="aac",
            audio_bitrate="192k",
            preset="medium",
            pixel_format="yuv420p",
            ffmpeg_params=["-crf", "20", "-movflags", "+faststart"],
            temp_audiofile_path=str(out_path.parent),
            logger="bar" if progress else None,
        )
        return float(video.duration)
    finally:
        video.close()
        for clip in clips:
            clip.close()


def _self_check() -> None:
    """Assemble the self-check script's audio and images into final.mp4."""
    import time

    from shortforge.config import get_settings
    from shortforge.fsutil import atomic_path
    from shortforge.log import console, step
    from shortforge.script import load_script

    settings = get_settings()
    folder = settings.output_dir / "_selfcheck"
    try:
        script = load_script(folder / "script.json")
    except OSError as exc:
        console.print(f"[bold red]Error:[/] no self-check script ({exc}).")
        raise SystemExit(1) from exc
    except ValueError as exc:
        console.print(
            "[bold red]Error:[/] the self-check script no longer passes validation. "
            'Run python -m shortforge.script "your topic" first.'
        )
        raise SystemExit(1) from exc
    scenes = [
        SceneAssets(scene.narration, folder / f"scene_{i:02d}.png", folder / f"scene_{i:02d}.wav")
        for i, scene in enumerate(script.scenes, start=1)
    ]
    missing = [p.name for s in scenes for p in (s.image, s.audio) if not p.exists()]
    if missing:
        console.print(
            f"[bold red]Error:[/] missing {', '.join(missing)}. Run python -m shortforge.tts "
            "and python -m shortforge.images for the current script first."
        )
        raise SystemExit(1)

    out = folder / "final.mp4"
    started = time.perf_counter()
    try:
        with step(f"Rendering {len(scenes)} scenes"), atomic_path(out) as tmp:
            duration = assemble_video(scenes, tmp, settings)
    except (OSError, ValueError) as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise SystemExit(1) from exc
    size_mb = out.stat().st_size / 1_000_000
    console.print(
        f"[green]Saved[/] {out}: {duration:.1f}s, {size_mb:.1f} MB, "
        f"rendered in {time.perf_counter() - started:.0f}s"
    )


if __name__ == "__main__":
    _self_check()

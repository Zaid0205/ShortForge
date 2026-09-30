"""Video assembly tests: Ken Burns frames and a small real render with ffmpeg."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from moviepy import VideoFileClip
from PIL import Image

from shortforge.config import Settings
from shortforge.video import SceneAssets, assemble_video, ken_burns_frames

FONT = Path(__file__).resolve().parents[1] / "assets" / "fonts" / "Montserrat-ExtraBold.ttf"


def gradient(width: int = 72, height: int = 128) -> Image.Image:
    x = np.linspace(0, 255, width, dtype=np.uint8)
    return Image.fromarray(np.tile(x, (height, 1))).convert("RGB")


def test_ken_burns_frames_have_frame_size_and_move() -> None:
    frame = ken_burns_frames(gradient(), duration=2.0, index=0)
    first, last = frame(0.0), frame(2.0)
    assert first.shape == last.shape == (128, 72, 3)
    assert not np.array_equal(first, last)


def test_zoom_in_starts_unscaled_and_zoom_out_ends_unscaled() -> None:
    image = gradient()
    original = np.asarray(image)
    assert np.abs(ken_burns_frames(image, 2.0, index=0)(0.0).astype(int) - original).max() <= 1
    assert np.abs(ken_burns_frames(image, 2.0, index=1)(2.0).astype(int) - original).max() <= 1


@pytest.fixture
def scene_files(tmp_path: Path) -> list[SceneAssets]:
    scenes = []
    for index, seconds in enumerate([1.2, 0.8], start=1):
        image = tmp_path / f"scene_{index:02d}.png"
        gradient(144, 256).save(image)
        audio = tmp_path / f"scene_{index:02d}.wav"
        t = np.arange(int(seconds * 24_000)) / 24_000
        sf.write(audio, 0.3 * np.sin(2 * np.pi * 220 * t), 24_000)
        scenes.append(SceneAssets(f"Scene number {index} has words to show", image, audio))
    return scenes


def test_assemble_video_matches_audio_length(
    tmp_path: Path, scene_files: list[SceneAssets]
) -> None:
    settings = Settings(_env_file=None, video_width=144, video_height=256, font_path=FONT)
    out = tmp_path / "final.mp4"
    duration = assemble_video(scene_files, out, settings, progress=False)
    assert duration == pytest.approx(2.0, abs=0.05)
    with VideoFileClip(str(out)) as clip:
        assert clip.size == [144, 256]
        assert clip.fps == pytest.approx(24)
        assert clip.audio is not None
        assert clip.duration == pytest.approx(2.0, abs=0.1)


def test_wrong_image_size_is_rejected(tmp_path: Path, scene_files: list[SceneAssets]) -> None:
    settings = Settings(_env_file=None, video_width=720, video_height=1280, font_path=FONT)
    with pytest.raises(ValueError, match="expected 720x1280"):
        assemble_video(scene_files, tmp_path / "final.mp4", settings, progress=False)

"""Caption chunking, timing and rendering tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shortforge.captions import (
    MAX_WIDTH_RATIO,
    caption_position,
    chunk_words,
    display_text,
    render_caption,
    scene_captions,
    time_chunks,
)
from shortforge.tts.base import EDGE_MARGIN_SECONDS, SCENE_PAUSE_SECONDS

FONT = Path(__file__).resolve().parents[1] / "assets" / "fonts" / "Montserrat-ExtraBold.ttf"
NARRATION = (
    "A CPU has a few powerful cores, while a GPU holds thousands of tiny ones "
    "that run the same operation on many numbers at once."
)


def test_chunks_are_three_to_five_words() -> None:
    chunks = chunk_words(NARRATION)
    assert all(3 <= len(chunk.split()) <= 5 for chunk in chunks)
    assert " ".join(chunks) == NARRATION


def test_chunks_break_at_punctuation() -> None:
    chunks = chunk_words("First comes the hook, then the payoff lands hard.")
    assert chunks[0] == "First comes the hook,"


def test_short_tail_is_merged() -> None:
    assert chunk_words("one two three four five six") == ["one two three", "four five six"]
    assert chunk_words("one two three four") == ["one two three four"]


def test_very_short_text_is_one_chunk() -> None:
    assert chunk_words("Hello there") == ["Hello there"]


def test_timing_is_proportional_contiguous_and_holds_to_scene_end() -> None:
    chunks = ["aaaa", "aaaaaaaa", "aaaa"]
    captions = time_chunks(chunks, speech_start=1.0, speech_end=5.0, scene_end=6.0)
    assert [c.start for c in captions] == pytest.approx([1.0, 2.0, 4.0])
    assert [c.end for c in captions] == pytest.approx([2.0, 4.0, 6.0])


def test_invalid_timing_is_rejected() -> None:
    with pytest.raises(ValueError):
        time_chunks(["a b c"], speech_start=3.0, speech_end=2.0, scene_end=4.0)


def test_scene_captions_cover_the_speech() -> None:
    duration = 7.0
    captions = scene_captions(NARRATION, duration)
    assert captions[0].start == pytest.approx(EDGE_MARGIN_SECONDS)
    assert captions[-1].end == pytest.approx(duration)
    speech_end = duration - SCENE_PAUSE_SECONDS - EDGE_MARGIN_SECONDS
    assert captions[-2].end < speech_end
    for before, after in zip(captions, captions[1:], strict=False):
        assert before.end == pytest.approx(after.start)


def test_render_caption_is_transparent_white_on_black() -> None:
    image = render_caption("Thousands of tiny cores", FONT, 720)
    pixels = np.asarray(image)
    assert image.mode == "RGBA"
    assert pixels[0, 0, 3] == 0
    opaque = pixels[pixels[..., 3] == 255][:, :3]
    assert (opaque == 255).all(axis=1).any() and (opaque == 0).all(axis=1).any()
    assert image.width <= 720 * MAX_WIDTH_RATIO + 24


def test_long_caption_wraps_and_fits() -> None:
    single = render_caption("GPUs", FONT, 720)
    wrapped = render_caption("Extraordinarily parallelized computational workloads", FONT, 720)
    assert wrapped.height > single.height
    assert wrapped.width <= 720 * MAX_WIDTH_RATIO + 24


def test_missing_font_is_reported(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="FONT_PATH"):
        render_caption("hello there friend", tmp_path / "missing.ttf", 720)


def test_caption_centered_at_65_percent() -> None:
    assert caption_position(100, 1280) == 782


@pytest.mark.parametrize(
    ("raw", "shown"),
    [
        ("the hook,", "the hook"),
        ("it lands.", "it lands"),
        ("at a time?", "at a time?"),
        ("wow!", "wow!"),
    ],
)
def test_display_text_drops_trailing_clutter(raw: str, shown: str) -> None:
    assert display_text(raw) == shown

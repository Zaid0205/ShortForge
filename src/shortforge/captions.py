"""Caption chunking, timing and rendering.

Narration is split into short chunks (3 to 5 words, breaking at punctuation where
possible). Each chunk gets a share of the scene's speaking time equal to its share of the
scene's characters, so no speech recognition is needed. Chunks are drawn with Pillow as
transparent PNGs, which avoids MoviePy's TextClip and its ImageMagick dependency.

Run `python -m shortforge.captions` to render a preview frame for review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from shortforge.tts.base import EDGE_MARGIN_SECONDS, SCENE_PAUSE_SECONDS

MIN_WORDS = 3
MAX_WORDS = 5
MAX_LINES = 2
MAX_WIDTH_RATIO = 0.86
"""Captions may use at most this fraction of the frame width."""
CENTER_Y_RATIO = 0.65
"""Vertical center of captions: low on screen, above the YouTube Shorts interface."""
FONT_SIZE_RATIO = 0.085
MIN_FONT_SIZE_RATIO = 0.05
STROKE_RATIO = 0.1
LINE_SPACING = 1.12
PADDING = 12

_BREAK_AFTER = re.compile(r"[,.;:!?]$")


@dataclass(frozen=True)
class Caption:
    """One on-screen caption and when it is visible, in seconds from the scene start."""

    text: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        """Seconds the caption is on screen."""
        return self.end - self.start


def chunk_words(text: str, min_words: int = MIN_WORDS, max_words: int = MAX_WORDS) -> list[str]:
    """Split `text` into chunks of `min_words` to `max_words` words.

    A chunk ends early at punctuation once it has `min_words` words, so captions follow
    the phrasing. A short leftover tail is merged into, or rebalanced with, the chunk
    before it, so a lone word never flashes up by itself.
    """
    words = text.split()
    chunks: list[list[str]] = []
    current: list[str] = []
    for word in words:
        current.append(word)
        if len(current) == max_words or (len(current) >= min_words and _BREAK_AFTER.search(word)):
            chunks.append(current)
            current = []
    if current:
        if chunks and len(current) < min_words:
            combined = chunks.pop() + current
            if len(combined) <= max_words:
                chunks.append(combined)
            else:
                half = (len(combined) + 1) // 2
                chunks.extend([combined[:half], combined[half:]])
        else:
            chunks.append(current)
    return [" ".join(chunk) for chunk in chunks]


def time_chunks(
    chunks: list[str], speech_start: float, speech_end: float, scene_end: float
) -> list[Caption]:
    """Spread `chunks` over the speech proportionally to their character counts.

    The last caption stays on until `scene_end`, through the pause after the speech, so
    the screen never goes blank between scenes.
    """
    if not chunks:
        return []
    if not 0 <= speech_start < speech_end <= scene_end:
        raise ValueError(
            f"invalid timing: speech {speech_start:.2f} to {speech_end:.2f}, scene {scene_end:.2f}"
        )
    total_chars = sum(len(chunk) for chunk in chunks)
    span = speech_end - speech_start
    captions = []
    cursor = speech_start
    for index, chunk in enumerate(chunks):
        end = cursor + span * len(chunk) / total_chars
        if index == len(chunks) - 1:
            end = scene_end
        captions.append(Caption(chunk, cursor, end))
        cursor = end
    return captions


def scene_captions(narration: str, audio_duration: float) -> list[Caption]:
    """Build timed captions for one scene from its narration and audio length.

    The TTS step's audio layout is known exactly: a short margin, the speech, a short
    margin, then the fixed scene pause. Captions cover the speech; the last one holds to the end.
    """
    speech_start = min(EDGE_MARGIN_SECONDS, audio_duration / 4)
    speech_end = audio_duration - SCENE_PAUSE_SECONDS - EDGE_MARGIN_SECONDS
    if speech_end <= speech_start:
        speech_end = audio_duration
    return time_chunks(chunk_words(narration), speech_start, speech_end, audio_duration)


def display_text(text: str) -> str:
    """Drop trailing commas, periods, colons and semicolons, which look like clutter on screen.

    Question and exclamation marks stay because they change how a line reads.
    """
    return text.rstrip(",.;: ") or text


@lru_cache(maxsize=16)
def _font(font_path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(font_path, size)


def _text_width(text: str, font: ImageFont.FreeTypeFont, stroke: int) -> float:
    return font.getlength(text) + 2 * stroke


def _layout(
    words: list[str], font: ImageFont.FreeTypeFont, stroke: int, max_width: float
) -> list[str] | None:
    """Return 1 or 2 balanced lines that fit `max_width`, or None if nothing fits."""
    one_line = " ".join(words)
    if _text_width(one_line, font, stroke) <= max_width:
        return [one_line]
    if MAX_LINES < 2 or len(words) < 2:
        return None
    best: list[str] | None = None
    best_width = float("inf")
    for split in range(1, len(words)):
        lines = [" ".join(words[:split]), " ".join(words[split:])]
        widest = max(_text_width(line, font, stroke) for line in lines)
        if widest <= max_width and widest < best_width:
            best, best_width = lines, widest
    return best


def render_caption(text: str, font_path: Path, frame_width: int) -> Image.Image:
    """Draw `text` as white bold type with a black outline on a transparent image.

    The font starts at `FONT_SIZE_RATIO` of the frame width and shrinks until the text
    fits in at most two lines within `MAX_WIDTH_RATIO` of the frame.

    Raises:
        FileNotFoundError: If the font file is missing.
    """
    if not font_path.is_file():
        raise FileNotFoundError(f"Caption font not found at {font_path}. Check FONT_PATH in .env.")
    words = display_text(text).split()
    max_width = frame_width * MAX_WIDTH_RATIO
    size = round(frame_width * FONT_SIZE_RATIO)
    min_size = round(frame_width * MIN_FONT_SIZE_RATIO)
    while True:
        font = _font(str(font_path), size)
        stroke = max(2, round(size * STROKE_RATIO))
        lines = _layout(words, font, stroke, max_width)
        if lines is not None or size <= min_size:
            break
        size -= 2
    if lines is None:
        lines = [" ".join(words)]

    ascent, descent = font.getmetrics()
    line_height = round((ascent + descent) * LINE_SPACING)
    width = round(max(_text_width(line, font, stroke) for line in lines)) + 2 * PADDING
    height = line_height * len(lines) + 2 * stroke + 2 * PADDING
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        draw.text(
            (width / 2, PADDING + stroke + index * line_height),
            line,
            font=font,
            fill=(255, 255, 255, 255),
            stroke_width=stroke,
            stroke_fill=(0, 0, 0, 255),
            anchor="ma",
        )
    return image


def caption_position(caption_height: int, frame_height: int) -> int:
    """Top y coordinate that centers a caption at `CENTER_Y_RATIO` of the frame height."""
    return round(frame_height * CENTER_Y_RATIO - caption_height / 2)


def _self_check() -> None:
    """Print scene 1's caption timings and save a preview frame with its longest caption."""
    from rich.table import Table

    from shortforge.config import get_settings
    from shortforge.fsutil import atomic_path
    from shortforge.log import console
    from shortforge.script import load_script
    from shortforge.tts import wav_duration

    settings = get_settings()
    folder = settings.output_dir / "_selfcheck"
    image_path, audio_path = folder / "scene_01.png", folder / "scene_01.wav"
    try:
        narration = load_script(folder / "script.json").scenes[0].narration
    except OSError as exc:
        console.print(f"[bold red]Error:[/] no self-check script ({exc}).")
        raise SystemExit(1) from exc
    except ValueError as exc:
        console.print(
            "[bold red]Error:[/] the self-check script no longer passes validation. "
            'Run python -m shortforge.script "your topic" first.'
        )
        raise SystemExit(1) from exc
    if not image_path.exists() or not audio_path.exists():
        console.print(
            "[bold red]Error:[/] scene_01.wav and scene_01.png are needed. "
            "Run python -m shortforge.tts and python -m shortforge.images first."
        )
        raise SystemExit(1)

    captions = scene_captions(narration, wav_duration(audio_path))
    table = Table(title="Scene 1 captions")
    for column in ("Start", "End", "Text"):
        table.add_column(column, justify="right" if column != "Text" else "left")
    for caption in captions:
        table.add_row(f"{caption.start:.2f}", f"{caption.end:.2f}", caption.text)
    console.print(table)

    longest = max(captions, key=lambda c: len(c.text))
    with Image.open(image_path) as base:
        frame = base.convert("RGBA")
    overlay = render_caption(longest.text, settings.font_path, frame.width)
    x = (frame.width - overlay.width) // 2
    frame.alpha_composite(overlay, (x, caption_position(overlay.height, frame.height)))
    out = folder / "caption_preview.png"
    with atomic_path(out) as tmp:
        frame.convert("RGB").save(tmp)
    console.print(f"[green]Saved[/] {out} with caption {longest.text!r}")


if __name__ == "__main__":
    _self_check()

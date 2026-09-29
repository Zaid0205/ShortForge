"""Abstract image generation interface shared by all image providers."""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

from PIL import Image, ImageOps

from shortforge.config import Settings


def fit_to_frame(image: Image.Image, width: int, height: int) -> Image.Image:
    """Center-crop and resize `image` to exactly ``width x height`` without distortion.

    A square 1024x1024 source becomes a centered 576x1024 strip, upscaled to 720x1280.
    """
    return ImageOps.fit(
        image.convert("RGB"), (width, height), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)
    )


class ImageGenerator(ABC):
    """Base class for image providers.

    Subclasses implement `_generate`, which returns encoded image bytes in any format
    Pillow can read. The base class appends the shared style suffix, crops to the video
    frame and saves a PNG, so every provider yields identical, frame-ready files.
    """

    name: str = "base"

    def __init__(self, style: str, width: int, height: int) -> None:
        self.style = style
        self.width = width
        self.height = height

    @classmethod
    @abstractmethod
    def from_settings(cls, settings: Settings) -> Self:
        """Build the provider from settings, raising `ConfigError` if a key is missing."""

    @abstractmethod
    def _generate(self, prompt: str) -> bytes:
        """Return encoded image bytes for the fully styled `prompt`."""

    def styled_prompt(self, prompt: str) -> str:
        """Append the channel's visual style suffix to a scene prompt."""
        return f"{prompt.rstrip(' .,')}, {self.style}" if self.style else prompt

    def generate(self, prompt: str, out_path: Path) -> None:
        """Generate an image for `prompt` and save it to `out_path` as a frame-sized PNG.

        Args:
            prompt: Scene image prompt, without the style suffix.
            out_path: Destination file; must end in ``.png``.
        """
        if out_path.suffix.lower() != ".png":
            raise ValueError(f"Image output must be a .png file, got {out_path.name}")
        data = self._generate(self.styled_prompt(prompt))
        try:
            with Image.open(io.BytesIO(data)) as raw:
                framed = fit_to_frame(raw, self.width, self.height)
        except (OSError, ValueError) as exc:
            raise RuntimeError(f"{self.name} returned data that is not a valid image") from exc
        out_path.parent.mkdir(parents=True, exist_ok=True)
        framed.save(out_path, format="PNG", optimize=True)

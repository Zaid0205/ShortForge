"""Choose where each scene's image comes from.

In ``generated`` mode (the default) every scene image is generated. In ``hybrid`` mode a
scene first gets a real Pexels photo for its `search_query`; only when nothing suitable is
found is an image generated from its `image_prompt`. The generator is created lazily, so a
hybrid run where every scene finds a photo never needs image-generation credentials.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Self

from shortforge.config import Settings
from shortforge.images import create_image_generator
from shortforge.images.base import ImageGenerator
from shortforge.images.pexels import PexelsPhotos
from shortforge.models import Scene


@dataclass(frozen=True)
class SceneImage:
    """Where a scene's image came from, and the credit line if it needs one."""

    source: Literal["stock", "generated"]
    credit: str | None = None


class SceneImageSource:
    """Produce one frame-sized PNG per scene, preferring real stock photos."""

    def __init__(
        self,
        settings: Settings,
        stock: PexelsPhotos | None = None,
        generator: ImageGenerator | None = None,
    ) -> None:
        self._settings = settings
        self._stock = stock
        self._generator = generator
        self._used_photo_ids: set[int] = set()

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """Use Pexels when ``IMAGE_SOURCE=hybrid``, otherwise generate every image."""
        stock = PexelsPhotos.from_settings(settings) if settings.image_source == "hybrid" else None
        return cls(settings, stock=stock)

    @property
    def generator(self) -> ImageGenerator:
        """The image generator, created on first use."""
        if self._generator is None:
            self._generator = create_image_generator(self._settings)
        return self._generator

    def render(self, scene: Scene, out_path: Path) -> SceneImage:
        """Write the scene's image to `out_path` and report its source."""
        if self._stock is not None:
            photo = self._stock.fetch(scene.search_query, out_path, self._used_photo_ids)
            if photo is not None:
                self._used_photo_ids.add(photo.id)
                return SceneImage("stock", photo.credit)
        self.generator.generate(scene.image_prompt, out_path)
        return SceneImage("generated")

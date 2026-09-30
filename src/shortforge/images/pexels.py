"""Real stock photos from the Pexels API (free key, photos free to use).

Each scene searches Pexels with its short `search_query`, keeps only portrait photos,
skips photos already used in the same video, and saves the best match cropped to the
video frame. Photographer credits are kept so they can go into the video description.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import httpx
from PIL import Image

from shortforge.config import ConfigError, Settings
from shortforge.images.base import fit_to_frame
from shortforge.retry import call_with_retries

SEARCH_URL = "https://api.pexels.com/v1/search"
MIN_ASPECT = 1.3
"""Minimum height/width ratio; anything squarer loses too much when cropped to 9:16."""
RESULTS_PER_SEARCH = 15
REQUEST_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class StockPhoto:
    """One Pexels search result and the details needed to credit it."""

    id: int
    width: int
    height: int
    download_url: str
    page_url: str
    photographer: str
    alt: str

    @property
    def credit(self) -> str:
        """Attribution line in the form Pexels asks for."""
        return f"Photo by {self.photographer} on Pexels: {self.page_url}"


class PexelsPhotos:
    """Search and download portrait stock photos."""

    name = "pexels"

    def __init__(
        self,
        api_key: str,
        width: int,
        height: int,
        *,
        max_attempts: int = 4,
        client: httpx.Client | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.max_attempts = max_attempts
        self._client = client or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)
        self._headers = {"Authorization": api_key}

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """Build from ``PEXELS_API_KEY`` and the video frame size."""
        return cls(
            settings.require_secret("pexels_api_key"),
            settings.video_width,
            settings.video_height,
            max_attempts=settings.max_retries,
        )

    def _get(self, url: str, **kwargs: Any) -> httpx.Response:
        response = self._client.get(url, **kwargs)
        response.raise_for_status()
        return response

    def search(self, query: str) -> list[StockPhoto]:
        """Return portrait photos for `query`, best matches first.

        Raises:
            ConfigError: If Pexels rejects the API key.
            RuntimeError: If the search fails after retries.
        """
        params = {
            "query": query,
            "orientation": "portrait",
            "size": "large",
            "per_page": RESULTS_PER_SEARCH,
        }
        try:
            response = call_with_retries(
                lambda: self._get(SEARCH_URL, params=params, headers=self._headers),
                what="Pexels search",
                attempts=self.max_attempts,
            )
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status in (401, 403):
                raise ConfigError(
                    f"Pexels rejected PEXELS_API_KEY (HTTP {status}). Check the key in .env."
                ) from exc
            raise RuntimeError(f"Pexels search for {query!r} failed (HTTP {status})") from exc

        photos = []
        for item in response.json().get("photos", []):
            try:
                photo = StockPhoto(
                    id=int(item["id"]),
                    width=int(item["width"]),
                    height=int(item["height"]),
                    download_url=item["src"]["large2x"],
                    page_url=item["url"],
                    photographer=item.get("photographer") or "unknown",
                    alt=item.get("alt") or "",
                )
            except (KeyError, TypeError, ValueError):
                continue
            if photo.width and photo.height / photo.width >= MIN_ASPECT:
                photos.append(photo)
        return photos

    def fetch(self, query: str, out_path: Path, exclude_ids: set[int]) -> StockPhoto | None:
        """Save the best unused portrait photo for `query` to `out_path` as a PNG.

        Returns:
            The photo used, or None if the search found nothing suitable.
        """
        candidates = [p for p in self.search(query) if p.id not in exclude_ids]
        if not candidates:
            return None
        photo = candidates[0]
        try:
            response = call_with_retries(
                lambda: self._get(photo.download_url),
                what="Pexels download",
                attempts=self.max_attempts,
            )
            with Image.open(io.BytesIO(response.content)) as raw:
                framed = fit_to_frame(raw, self.width, self.height)
        except (httpx.HTTPError, OSError) as exc:
            raise RuntimeError(f"Could not download Pexels photo {photo.id}") from exc
        out_path.parent.mkdir(parents=True, exist_ok=True)
        framed.save(out_path, format="PNG", optimize=True)
        return photo

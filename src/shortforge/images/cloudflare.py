"""FLUX.1-schnell on Cloudflare Workers AI, covered by Cloudflare's free daily allocation.

One POST per image. The API returns a base64-encoded square JPEG, which the base class
crops to the 9:16 video frame.
"""

from __future__ import annotations

import base64
import binascii
from typing import Any, Self

import httpx

from shortforge.config import ConfigError, Settings
from shortforge.images.base import ImageGenerator
from shortforge.retry import call_with_retries

API_URL = "https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
MAX_PROMPT_CHARS = 2048
REQUEST_TIMEOUT_SECONDS = 90.0


class CloudflareImages(ImageGenerator):
    """Workers AI image generation."""

    name = "cloudflare"

    def __init__(
        self,
        style: str,
        width: int,
        height: int,
        *,
        account_id: str,
        api_token: str,
        model: str,
        steps: int = 4,
        max_attempts: int = 4,
        client: httpx.Client | None = None,
    ) -> None:
        super().__init__(style, width, height)
        self.url = API_URL.format(account_id=account_id, model=model)
        self.steps = steps
        self.max_attempts = max_attempts
        self._client = client or httpx.Client(timeout=REQUEST_TIMEOUT_SECONDS)
        self._headers = {"Authorization": f"Bearer {api_token}"}

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """Build from the ``CLOUDFLARE_*`` settings and ``IMAGE_STYLE``."""
        return cls(
            settings.image_style,
            settings.video_width,
            settings.video_height,
            account_id=settings.require_secret("cloudflare_account_id"),
            api_token=settings.require_secret("cloudflare_api_token"),
            model=settings.cloudflare_image_model,
            steps=settings.cloudflare_steps,
            max_attempts=settings.max_retries,
        )

    def _post(self, payload: dict[str, Any]) -> httpx.Response:
        """Send one request; raise `httpx.HTTPStatusError` on a non-2xx status."""
        response = self._client.post(self.url, json=payload, headers=self._headers)
        response.raise_for_status()
        return response

    def _generate(self, prompt: str) -> bytes:
        """Request one image and return its decoded bytes."""
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError(
                f"Styled prompt is {len(prompt)} characters; Cloudflare accepts at most "
                f"{MAX_PROMPT_CHARS}. Shorten IMAGE_STYLE."
            )
        payload = {"prompt": prompt, "steps": self.steps}
        try:
            response = call_with_retries(
                lambda: self._post(payload),
                what="Cloudflare image request",
                attempts=self.max_attempts,
            )
        except httpx.HTTPStatusError as exc:
            raise _explain(exc) from exc

        body = response.json()
        if not body.get("success", False):
            raise RuntimeError(f"Cloudflare reported a failure: {_error_text(body)}")
        try:
            return base64.b64decode(body["result"]["image"], validate=True)
        except (KeyError, TypeError, binascii.Error) as exc:
            raise RuntimeError("Cloudflare response did not contain a base64 image") from exc


def _error_text(body: Any) -> str:
    """Pull Cloudflare's own error messages out of a response body, if present."""
    errors = body.get("errors") if isinstance(body, dict) else None
    if errors:
        return "; ".join(
            str(e.get("message", e)) if isinstance(e, dict) else str(e) for e in errors
        )
    return "no error details returned"


def _explain(exc: httpx.HTTPStatusError) -> Exception:
    """Turn a final HTTP error into a message that says what to do about it."""
    status = exc.response.status_code
    try:
        detail = _error_text(exc.response.json())
    except ValueError:
        detail = exc.response.text[:200] or "no error details returned"
    if status in (401, 403):
        return ConfigError(
            f"Cloudflare rejected the credentials (HTTP {status}: {detail}). Check "
            "CLOUDFLARE_ACCOUNT_ID, and that CLOUDFLARE_API_TOKEN has the Workers AI permission."
        )
    if status == 429:
        return RuntimeError(
            f"Cloudflare kept rate limiting after retries (HTTP 429: {detail}). The free "
            "allocation is 10,000 neurons per day; if it is used up, try again after it resets "
            "(00:00 UTC). Finished images are cached, so a re-run only generates what is missing."
        )
    return RuntimeError(f"Cloudflare image request failed (HTTP {status}: {detail})")

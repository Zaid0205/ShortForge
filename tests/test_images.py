"""Cloudflare image provider and review-sheet tests. HTTP is mocked; no network access."""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from shortforge.config import ConfigError, Settings
from shortforge.images import create_image_generator
from shortforge.images.__main__ import contact_sheet
from shortforge.images.cloudflare import CloudflareImages


def jpeg_b64(size: tuple[int, int] = (1024, 1024)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, "teal").save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode()


def make_provider(handler, style: str = "flat style", attempts: int = 3) -> CloudflareImages:
    """Provider wired to a mock transport; `handler` receives each request."""
    return CloudflareImages(
        style,
        720,
        1280,
        account_id="acc123",
        api_token="tok456",
        model="@cf/black-forest-labs/flux-1-schnell",
        steps=6,
        max_attempts=attempts,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )


def ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200, json={"success": True, "result": {"image": jpeg_b64()}, "errors": []}
    )


def test_generates_frame_sized_png(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return ok(request)

    out = tmp_path / "scene_01.png"
    make_provider(handler).generate("a lighthouse in fog", out)
    with Image.open(out) as image:
        assert image.size == (720, 1280)
    request = requests[0]
    assert str(request.url).endswith("/accounts/acc123/ai/run/@cf/black-forest-labs/flux-1-schnell")
    assert request.headers["authorization"] == "Bearer tok456"
    body = json.loads(request.content)
    assert body == {
        "prompt": "a lighthouse in fog, flat style",
        "steps": 6,
    }


def test_rate_limit_is_retried(tmp_path: Path) -> None:
    replies = [
        httpx.Response(429, headers={"retry-after": "0"}, json={"errors": [{"message": "slow"}]}),
        None,
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        reply = replies.pop(0)
        return reply if reply is not None else ok(request)

    make_provider(handler).generate("a lighthouse in fog", tmp_path / "scene_01.png")
    assert replies == []


def test_exhausted_rate_limit_explains_free_allocation(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429, headers={"retry-after": "0"}, json={"errors": [{"message": "limit"}]}
        )

    with pytest.raises(RuntimeError, match="10,000 neurons"):
        make_provider(handler, attempts=2).generate("a lighthouse", tmp_path / "s.png")


@pytest.mark.parametrize("status", [401, 403])
def test_bad_credentials_become_config_error(tmp_path: Path, status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={"errors": [{"message": "Authentication error"}]})

    with pytest.raises(ConfigError, match="Authentication error"):
        make_provider(handler).generate("a lighthouse", tmp_path / "s.png")


def test_unsuccessful_body_is_reported(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"success": False, "errors": [{"message": "NSFW"}]})

    with pytest.raises(RuntimeError, match="NSFW"):
        make_provider(handler).generate("a lighthouse", tmp_path / "s.png")


def test_missing_image_is_reported(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"success": True, "result": {"image": "%%%"}})

    with pytest.raises(RuntimeError, match="base64 image"):
        make_provider(handler).generate("a lighthouse", tmp_path / "s.png")


def test_overlong_prompt_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="IMAGE_STYLE"):
        make_provider(ok, style="x" * 2100).generate("a lighthouse", tmp_path / "s.png")


def test_factory_requires_cloudflare_credentials() -> None:
    with pytest.raises(ConfigError, match="CLOUDFLARE_ACCOUNT_ID"):
        create_image_generator(Settings(_env_file=None))


def test_factory_builds_cloudflare() -> None:
    settings = Settings(
        _env_file=None,
        cloudflare_account_id="0123456789abcdef0123456789abcdef",
        cloudflare_api_token="t",
        cloudflare_steps=8,
    )
    provider = create_image_generator(settings)
    assert isinstance(provider, CloudflareImages)
    assert provider.steps == 8


def test_contact_sheet_layout(tmp_path: Path) -> None:
    paths = []
    for index in range(3):
        path = tmp_path / f"scene_{index}.png"
        Image.new("RGB", (720, 1280), "navy").save(path)
        paths.append(path)
    sheet = contact_sheet([("row a", paths), ("row b", paths[:2])])
    assert sheet.size == (8 + 3 * (216 + 8), 8 + 2 * (36 + 384 + 8))

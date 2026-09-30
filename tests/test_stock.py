"""Pexels stock photo search, hybrid image sourcing and the FLUX.2 request format."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import httpx
import pytest
from PIL import Image

from shortforge.config import ConfigError, Settings
from shortforge.images.base import ImageGenerator
from shortforge.images.cloudflare import CloudflareImages
from shortforge.images.pexels import PexelsPhotos
from shortforge.images.sourcing import SceneImageSource
from shortforge.models import Scene

ACCOUNT = "0123456789abcdef0123456789abcdef"


def jpeg(size: tuple[int, int], color: str = "teal") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def photo(photo_id: int, width: int = 1000, height: int = 1500) -> dict:
    return {
        "id": photo_id,
        "width": width,
        "height": height,
        "url": f"https://www.pexels.com/photo/{photo_id}/",
        "photographer": f"Person {photo_id}",
        "alt": "a server room",
        "src": {"large2x": f"https://images.pexels.com/photos/{photo_id}.jpeg"},
    }


def pexels_client(results: list[dict], log: list[httpx.Request] | None = None) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        if log is not None:
            log.append(request)
        if request.url.host == "api.pexels.com":
            return httpx.Response(200, json={"photos": results})
        return httpx.Response(200, content=jpeg((866, 1300)))

    return httpx.Client(transport=httpx.MockTransport(handler))


def make_pexels(results: list[dict], log: list[httpx.Request] | None = None) -> PexelsPhotos:
    return PexelsPhotos("pk_test", 720, 1280, client=pexels_client(results, log))


SCENE = Scene(
    narration="Data centers run day and night.",
    search_query="server room",
    image_prompt="rows of servers in a data center",
)


def test_search_sends_key_and_portrait_filter() -> None:
    log: list[httpx.Request] = []
    make_pexels([photo(1)], log).search("server room")
    request = log[0]
    assert request.headers["authorization"] == "pk_test"
    assert request.url.params["query"] == "server room"
    assert request.url.params["orientation"] == "portrait"


def test_search_drops_landscape_and_square_photos() -> None:
    results = [photo(1, 1600, 1000), photo(2, 1000, 1000), photo(3, 1000, 1500)]
    assert [p.id for p in make_pexels(results).search("x")] == [3]


def test_fetch_saves_frame_and_skips_used_photos(tmp_path: Path) -> None:
    out = tmp_path / "scene_01.png"
    chosen = make_pexels([photo(1), photo(2)]).fetch("server room", out, exclude_ids={1})
    assert chosen is not None and chosen.id == 2
    assert chosen.credit == "Photo by Person 2 on Pexels: https://www.pexels.com/photo/2/"
    with Image.open(out) as image:
        assert image.size == (720, 1280)


def test_fetch_returns_none_without_results(tmp_path: Path) -> None:
    assert make_pexels([]).fetch("quantum foam", tmp_path / "s.png", set()) is None


def test_bad_key_becomes_config_error() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(401)))
    with pytest.raises(ConfigError, match="PEXELS_API_KEY"):
        PexelsPhotos("bad", 720, 1280, client=client).search("x")


class RecordingGenerator(ImageGenerator):
    name = "recording"

    def __init__(self) -> None:
        super().__init__("style", 720, 1280)
        self.prompts: list[str] = []

    @classmethod
    def from_settings(cls, settings: Settings) -> RecordingGenerator:
        return cls()

    def _generate(self, prompt: str) -> bytes:
        self.prompts.append(prompt)
        return jpeg((720, 1280), "navy")


def test_hybrid_prefers_stock_and_never_repeats_a_photo(tmp_path: Path) -> None:
    generator = RecordingGenerator()
    source = SceneImageSource(
        Settings(_env_file=None), stock=make_pexels([photo(7)]), generator=generator
    )
    first = source.render(SCENE, tmp_path / "scene_01.png")
    second = source.render(SCENE, tmp_path / "scene_02.png")
    assert first.source == "stock" and "Person 7" in (first.credit or "")
    assert second.source == "generated" and second.credit is None
    assert generator.prompts == ["rows of servers in a data center, style"]


def test_generated_mode_skips_stock(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, image_source="generated")
    generator = RecordingGenerator()
    source = SceneImageSource.from_settings(settings)
    source._generator = generator
    assert source.render(SCENE, tmp_path / "scene_01.png").source == "generated"


def test_hybrid_requires_pexels_key() -> None:
    with pytest.raises(ConfigError, match="PEXELS_API_KEY"):
        SceneImageSource.from_settings(Settings(_env_file=None, image_source="hybrid"))


def test_flux2_sends_multipart_with_frame_size(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        image = base64.b64encode(jpeg((720, 1280))).decode()
        return httpx.Response(200, json={"result": {"image": image}})

    provider = CloudflareImages(
        "realistic photo",
        720,
        1280,
        account_id=ACCOUNT,
        api_token="tok",
        model="@cf/black-forest-labs/flux-2-klein-4b",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    provider.generate("a technician in a server room", tmp_path / "scene.png")
    request = seen[0]
    body = request.read()
    assert request.headers["content-type"].startswith("multipart/form-data")
    assert b'name="width"' in body and b"720" in body and b'name="height"' in body
    assert b"a technician in a server room, realistic photo" in body

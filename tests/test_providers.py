"""Provider base-class and factory tests using in-memory fake providers."""

from __future__ import annotations

import io
import sys
import types
from pathlib import Path
from typing import Self

import numpy as np
import pytest
import soundfile as sf
from PIL import Image

from shortforge import images, tts
from shortforge.config import ConfigError, ImageProviderName, Settings, TTSProviderName
from shortforge.images.base import ImageGenerator, fit_to_frame
from shortforge.tts.base import TextToSpeech, wav_duration


class FakeTTS(TextToSpeech):
    name = "fake"

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls()

    def _synthesize(self, text: str) -> tuple[np.ndarray, int]:
        return np.zeros(24_000 * 2, dtype=np.float32), 24_000


class FakeImages(ImageGenerator):
    name = "fake"
    last_prompt = ""

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        return cls(settings.image_style, settings.video_width, settings.video_height)

    def _generate(self, prompt: str) -> bytes:
        FakeImages.last_prompt = prompt
        buffer = io.BytesIO()
        Image.new("RGB", (1024, 1024), "navy").save(buffer, format="JPEG")
        return buffer.getvalue()


@pytest.fixture
def fake_module(monkeypatch: pytest.MonkeyPatch) -> str:
    """Register a throwaway module holding the fake providers."""
    module = types.ModuleType("shortforge_fake_providers")
    module.FakeTTS = FakeTTS
    module.FakeImages = FakeImages
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module.__name__


def test_tts_writes_wav_and_reports_duration(tmp_path: Path) -> None:
    out = tmp_path / "scene_01.wav"
    duration = FakeTTS().synthesize("hello there", out)
    assert duration == pytest.approx(2.0)
    assert wav_duration(out) == pytest.approx(2.0)
    assert sf.info(out).subtype == "PCM_16"


def test_tts_rejects_non_wav_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        FakeTTS().synthesize("hello", tmp_path / "scene.mp3")


def test_fit_to_frame_crops_square_to_vertical() -> None:
    framed = fit_to_frame(Image.new("RGB", (1024, 1024)), 720, 1280)
    assert framed.size == (720, 1280)


def test_image_generator_appends_style_and_saves_frame(tmp_path: Path) -> None:
    generator = FakeImages("flat vector style", 720, 1280)
    out = tmp_path / "scene_01.png"
    generator.generate("a robot reading a book.", out)
    assert FakeImages.last_prompt == "a robot reading a book, flat vector style"
    with Image.open(out) as saved:
        assert saved.size == (720, 1280)


def test_tts_factory_uses_registry(monkeypatch: pytest.MonkeyPatch, fake_module: str) -> None:
    monkeypatch.setitem(tts.PROVIDERS, TTSProviderName.KOKORO, f"{fake_module}:FakeTTS")
    provider = tts.create_tts(Settings(_env_file=None, tts_provider="kokoro"))
    assert isinstance(provider, FakeTTS)


def test_image_factory_uses_registry(monkeypatch: pytest.MonkeyPatch, fake_module: str) -> None:
    monkeypatch.setitem(images.PROVIDERS, ImageProviderName.CLOUDFLARE, f"{fake_module}:FakeImages")
    provider = images.create_image_generator(Settings(_env_file=None))
    assert isinstance(provider, FakeImages)
    assert (provider.width, provider.height) == (720, 1280)


def test_factory_reports_unloadable_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(tts.PROVIDERS, TTSProviderName.KOKORO, "shortforge.missing:Nope")
    with pytest.raises(ConfigError, match="failed to load"):
        tts.create_tts(Settings(_env_file=None))

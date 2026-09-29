"""Config loading and validation tests. None of these read the real `.env`."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shortforge.config import ConfigError, ImageProviderName, Settings, TTSProviderName


def test_defaults_are_free_tier_providers() -> None:
    settings = Settings(_env_file=None)
    assert settings.tts_provider is TTSProviderName.KOKORO
    assert settings.image_provider is ImageProviderName.CLOUDFLARE
    assert (settings.video_width, settings.video_height, settings.video_fps) == (720, 1280, 24)


def test_provider_names_are_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_PROVIDER", " Replicate ")
    monkeypatch.setenv("TTS_PROVIDER", "ELEVENLABS")
    settings = Settings(_env_file=None)
    assert settings.image_provider is ImageProviderName.REPLICATE
    assert settings.tts_provider is TTSProviderName.ELEVENLABS


def test_unknown_provider_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGE_PROVIDER", "cloudflareREPLICATE_API_TOKEN=oops")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_empty_values_count_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "")
    assert Settings(_env_file=None).groq_api_key is None


def test_require_secret_returns_plain_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    assert Settings(_env_file=None).require_secret("groq_api_key") == "gsk_test"


def test_require_secret_names_missing_variable() -> None:
    with pytest.raises(ConfigError, match="GROQ_API_KEY is not set"):
        Settings(_env_file=None).require_secret("groq_api_key")


def test_secrets_are_masked_in_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "super-secret-token")
    assert "super-secret-token" not in repr(Settings(_env_file=None))

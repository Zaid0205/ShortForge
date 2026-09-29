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
    monkeypatch.setenv("IMAGE_PROVIDER", " Cloudflare ")
    monkeypatch.setenv("TTS_PROVIDER", "KOKORO")
    settings = Settings(_env_file=None)
    assert settings.image_provider is ImageProviderName.CLOUDFLARE
    assert settings.tts_provider is TTSProviderName.KOKORO


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


def test_cloudflare_account_id_accepts_hex_id(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "0123456789abcdef0123456789abcdef")
    assert Settings(_env_file=None).cloudflare_account_id == "0123456789abcdef0123456789abcdef"


def test_cloudflare_account_id_rejects_dashboard_url(monkeypatch: pytest.MonkeyPatch) -> None:
    url = "https://dash.cloudflare.com/0123456789abcdef0123456789abcdef/home/ai"
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", url)
    with pytest.raises(ValidationError, match="looks like the ID itself"):
        Settings(_env_file=None)

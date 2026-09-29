"""Typed application settings loaded from environment variables and `.env`.

Credentials are optional at load time so that tests, self-checks and providers you
are not using never fail on a missing key. Code that needs a secret asks for it
through `Settings.require_secret`, which raises a `ConfigError` naming the exact
variable to set.

Run `python -m shortforge.config` to print the resolved settings with secrets masked.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from shortforge.models import MAX_SCENES, MIN_SCENES


class ConfigError(RuntimeError):
    """Raised when configuration is missing or invalid, with a fix-it message."""


class TTSProviderName(StrEnum):
    """Available text-to-speech backends."""

    KOKORO = "kokoro"
    ELEVENLABS = "elevenlabs"


class ImageProviderName(StrEnum):
    """Available image generation backends."""

    CLOUDFLARE = "cloudflare"
    REPLICATE = "replicate"


class Settings(BaseSettings):
    """All runtime settings. Field names map to upper-case environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8-sig",
        env_ignore_empty=True,
        extra="ignore",
        str_strip_whitespace=True,
    )

    groq_api_key: SecretStr | None = None
    groq_model: str = "llama-3.3-70b-versatile"

    channel_niche: str = "AI tools and tech concepts, explained in 45 seconds"

    tts_provider: TTSProviderName = TTSProviderName.KOKORO
    kokoro_voice: str = "af_heart"
    kokoro_speed: float = Field(default=1.0, ge=0.5, le=2.0)
    elevenlabs_api_key: SecretStr | None = None
    elevenlabs_voice_id: str | None = None
    elevenlabs_model: str = "eleven_multilingual_v2"

    image_provider: ImageProviderName = ImageProviderName.CLOUDFLARE
    cloudflare_account_id: str | None = None
    cloudflare_api_token: SecretStr | None = None
    cloudflare_image_model: str = "@cf/black-forest-labs/flux-1-schnell"
    replicate_api_token: SecretStr | None = None
    replicate_image_model: str = "black-forest-labs/flux-schnell"
    image_style: str = (
        "clean modern tech illustration, soft cinematic lighting, deep blue and cyan palette, "
        "high detail, vertical composition, subject centered, no text, no letters, no watermark"
    )

    video_width: int = Field(default=720, gt=0)
    video_height: int = Field(default=1280, gt=0)
    video_fps: int = Field(default=24, gt=0)
    font_path: Path = Path("assets/fonts/Montserrat-ExtraBold.ttf")

    youtube_client_secret: Path = Path("client_secret.json")
    youtube_token: Path = Path("token.json")
    youtube_privacy: Literal["private", "unlisted", "public"] = "private"

    output_dir: Path = Path("output")
    default_scenes: int = Field(default=6, ge=MIN_SCENES, le=MAX_SCENES)
    max_retries: int = Field(default=4, ge=1, le=10)

    @field_validator("tts_provider", "image_provider", "youtube_privacy", mode="before")
    @classmethod
    def _normalize_choice(cls, value: object) -> object:
        """Accept provider names regardless of case or stray whitespace."""
        return value.strip().lower() if isinstance(value, str) else value

    def require_secret(self, field: str) -> str:
        """Return the plain value of a credential field, or raise a `ConfigError`.

        Args:
            field: Settings attribute name, for example ``"groq_api_key"``.

        Raises:
            ConfigError: If the value is unset.
        """
        value = getattr(self, field)
        if value is None:
            raise ConfigError(
                f"{field.upper()} is not set. Add it to your .env file (see .env.example)."
            )
        return value.get_secret_value() if isinstance(value, SecretStr) else str(value)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once per process.

    Raises:
        ConfigError: If a value in the environment or `.env` fails validation.
    """
    try:
        return Settings()
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']).upper()}: {err['msg']}" for err in exc.errors()
        )
        raise ConfigError(f"Invalid configuration in .env or environment: {problems}") from exc


def _self_check() -> None:
    """Print resolved settings with secrets masked and credential status."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    try:
        settings = get_settings()
    except ConfigError as exc:
        console.print(f"[bold red]Config error:[/] {exc}")
        raise SystemExit(1) from exc

    table = Table(title="ShortForge settings", show_lines=False)
    table.add_column("Setting", style="cyan")
    table.add_column("Value")
    for name, value in settings.model_dump().items():
        if isinstance(getattr(settings, name), SecretStr):
            shown = "[green]set[/]"
        elif value is None:
            shown = "[dim]not set[/]"
        else:
            shown = str(value)
        table.add_row(name.upper(), shown)
    console.print(table)

    required = ["groq_api_key"]
    if settings.tts_provider is TTSProviderName.ELEVENLABS:
        required += ["elevenlabs_api_key", "elevenlabs_voice_id"]
    if settings.image_provider is ImageProviderName.CLOUDFLARE:
        required += ["cloudflare_account_id", "cloudflare_api_token"]
    else:
        required += ["replicate_api_token"]

    missing = [name.upper() for name in required if getattr(settings, name) is None]
    if missing:
        console.print(f"[bold red]Missing for selected providers:[/] {', '.join(missing)}")
        raise SystemExit(1)
    console.print(
        f"[bold green]OK[/] TTS={settings.tts_provider.value}, "
        f"images={settings.image_provider.value}, all required keys present."
    )


if __name__ == "__main__":
    _self_check()

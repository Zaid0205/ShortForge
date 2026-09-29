"""Text-to-speech providers and the factory that selects one from settings.

Providers are imported lazily, so choosing ElevenLabs never loads torch and Kokoro.
To add a provider: implement `TextToSpeech`, add a `TTSProviderName` member, and
register its import path below. Pipeline code does not change.
"""

from __future__ import annotations

import importlib

from shortforge.config import ConfigError, Settings, TTSProviderName, get_settings
from shortforge.tts.base import TextToSpeech, wav_duration

PROVIDERS: dict[TTSProviderName, str] = {
    TTSProviderName.KOKORO: "shortforge.tts.kokoro:KokoroTTS",
    TTSProviderName.ELEVENLABS: "shortforge.tts.elevenlabs:ElevenLabsTTS",
}


def create_tts(settings: Settings | None = None) -> TextToSpeech:
    """Instantiate the TTS provider named by ``TTS_PROVIDER``.

    Raises:
        ConfigError: If the provider is unregistered or cannot be loaded.
    """
    settings = settings or get_settings()
    target = PROVIDERS.get(settings.tts_provider)
    if target is None:
        raise ConfigError(f"No TTS provider registered for {settings.tts_provider!r}")
    module_name, class_name = target.split(":")
    try:
        provider_cls = getattr(importlib.import_module(module_name), class_name)
    except (ImportError, AttributeError) as exc:
        raise ConfigError(
            f"TTS provider {settings.tts_provider.value!r} failed to load: {exc}"
        ) from exc
    return provider_cls.from_settings(settings)


__all__ = ["PROVIDERS", "TextToSpeech", "create_tts", "wav_duration"]

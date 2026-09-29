"""Kokoro-82M text-to-speech, running locally on CPU.

The model (about 330 MB) is downloaded from Hugging Face on first use and cached in
the user's Hugging Face cache. It is loaded once per provider instance and reused for
every scene. Kokoro uses espeak-ng as a fallback phonemizer for unknown words.
"""

from __future__ import annotations

import os
import warnings
from typing import TYPE_CHECKING, Any, Self

import httpx
import numpy as np

from shortforge.config import Settings
from shortforge.tts.base import TextToSpeech

if TYPE_CHECKING:
    from kokoro import KPipeline

REPO_ID = "hexgrad/Kokoro-82M"
SAMPLE_RATE = 24_000
LANG_CODES = {"a": "American English", "b": "British English"}


class KokoroTTS(TextToSpeech):
    """Local Kokoro voice. Voice names start with their language code, e.g. ``af_heart``."""

    name = "kokoro"

    def __init__(self, voice: str, speed: float = 1.0, pipeline: KPipeline | Any = None) -> None:
        """Create the provider. The model loads lazily on the first `synthesize` call.

        Args:
            voice: Kokoro voice ID; its first letter selects the language (``a`` or ``b``).
            speed: Speaking rate multiplier.
            pipeline: Optional preloaded pipeline, used by tests.
        """
        lang_code = voice[:1]
        if lang_code not in LANG_CODES:
            raise ValueError(
                f"Unsupported Kokoro voice {voice!r}: voice IDs must start with "
                f"{' or '.join(repr(c) for c in LANG_CODES)} (for example 'af_heart')."
            )
        self.voice = voice
        self.speed = speed
        self.lang_code = lang_code
        self._pipeline = pipeline

    @classmethod
    def from_settings(cls, settings: Settings) -> Self:
        """Build from ``KOKORO_VOICE`` and ``KOKORO_SPEED``."""
        return cls(settings.kokoro_voice, settings.kokoro_speed)

    @property
    def pipeline(self) -> KPipeline | Any:
        """The loaded Kokoro pipeline, downloading the model on first access."""
        if self._pipeline is None:
            os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning, module="torch")
                warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
                from kokoro import KPipeline

                try:
                    self._pipeline = KPipeline(lang_code=self.lang_code, repo_id=REPO_ID)
                except (OSError, httpx.HTTPError) as exc:
                    raise RuntimeError(
                        f"Could not load the Kokoro model from Hugging Face ({REPO_ID}). "
                        "The first run needs internet access to download it."
                    ) from exc
        return self._pipeline

    def _synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Run Kokoro and join its segments into one clip."""
        try:
            results = list(self.pipeline(text, voice=self.voice, speed=self.speed))
        except (OSError, httpx.HTTPError) as exc:
            raise RuntimeError(
                f"Kokoro could not load voice {self.voice!r}. Check KOKORO_VOICE in .env "
                "against the voice list at https://huggingface.co/hexgrad/Kokoro-82M."
            ) from exc
        chunks = [np.asarray(r.audio, dtype=np.float32) for r in results if r.audio is not None]
        if not chunks:
            raise RuntimeError(f"Kokoro produced no audio for {text[:60]!r}")
        return np.concatenate(chunks), SAMPLE_RATE

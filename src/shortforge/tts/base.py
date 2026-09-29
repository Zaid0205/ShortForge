"""Abstract text-to-speech interface shared by all TTS providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import numpy as np
import soundfile as sf

from shortforge.config import Settings


class TextToSpeech(ABC):
    """Base class for TTS providers.

    Subclasses implement `_synthesize`, which returns raw mono samples. The base class
    owns writing the WAV file and measuring its duration, so every provider produces
    identical output files and the pipeline never depends on provider details.
    """

    name: str = "base"

    @classmethod
    @abstractmethod
    def from_settings(cls, settings: Settings) -> Self:
        """Build the provider from settings, raising `ConfigError` if a key is missing."""

    @abstractmethod
    def _synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Return ``(samples, sample_rate)`` for `text` as a mono float or int16 array."""

    def synthesize(self, text: str, out_path: Path) -> float:
        """Speak `text`, write it to `out_path` as 16-bit PCM WAV, and return its duration.

        Args:
            text: Narration for one scene.
            out_path: Destination file; must end in ``.wav``.

        Returns:
            Audio duration in seconds.
        """
        if out_path.suffix.lower() != ".wav":
            raise ValueError(f"TTS output must be a .wav file, got {out_path.name}")
        samples, sample_rate = self._synthesize(text)
        samples = np.asarray(samples).squeeze()
        if samples.ndim != 1 or samples.size == 0:
            raise RuntimeError(f"{self.name} returned no usable audio for: {text[:60]!r}")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, samples, sample_rate, subtype="PCM_16")
        return samples.size / sample_rate


def wav_duration(path: Path) -> float:
    """Return the duration in seconds of an existing audio file."""
    return float(sf.info(path).duration)

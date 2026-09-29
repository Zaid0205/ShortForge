"""Abstract text-to-speech interface shared by all TTS providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Self

import numpy as np
import soundfile as sf

from shortforge.config import Settings

SILENCE_THRESHOLD_DB = -40.0
"""Samples quieter than this, relative to the clip's peak, count as silence."""
EDGE_MARGIN_SECONDS = 0.05
"""Silence kept before and after speech so onsets and word endings are not clipped."""
SCENE_PAUSE_SECONDS = 0.3
"""Silence appended after every scene so all cuts breathe the same way."""


def tidy_audio(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    """Trim uneven edge silence and append the fixed scene pause.

    Args:
        samples: Mono audio as float in ``[-1, 1]`` or int16.
        sample_rate: Samples per second.

    Returns:
        Float32 audio: a short margin, the speech, then `SCENE_PAUSE_SECONDS` of silence.

    Raises:
        RuntimeError: If the clip is empty or contains no audible sound.
    """
    audio = np.asarray(samples).squeeze()
    if audio.ndim != 1 or audio.size == 0:
        raise RuntimeError("audio must be a non-empty mono array")
    if np.issubdtype(audio.dtype, np.integer):
        audio = audio.astype(np.float32) / np.iinfo(audio.dtype).max
    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)

    peak = float(np.abs(audio).max())
    if peak == 0.0:
        raise RuntimeError("audio is completely silent")
    loud = np.flatnonzero(np.abs(audio) >= peak * 10 ** (SILENCE_THRESHOLD_DB / 20))
    margin = int(EDGE_MARGIN_SECONDS * sample_rate)
    start = max(0, loud[0] - margin)
    end = min(audio.size, loud[-1] + 1 + margin)
    pause = np.zeros(int(SCENE_PAUSE_SECONDS * sample_rate), dtype=np.float32)
    return np.concatenate([audio[start:end], pause])


class TextToSpeech(ABC):
    """Base class for TTS providers.

    Subclasses implement `_synthesize`, which returns raw mono samples. The base class
    tidies the silence, writes the WAV file and measures its duration, so every provider
    produces identically paced files and the pipeline never depends on provider details.
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
            Audio duration in seconds, including the trailing scene pause.
        """
        if out_path.suffix.lower() != ".wav":
            raise ValueError(f"TTS output must be a .wav file, got {out_path.name}")
        samples, sample_rate = self._synthesize(text)
        try:
            audio = tidy_audio(samples, sample_rate)
        except RuntimeError as exc:
            raise RuntimeError(
                f"{self.name} returned unusable audio ({exc}) for {text[:60]!r}"
            ) from exc
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, audio, sample_rate, subtype="PCM_16")
        return audio.size / sample_rate


def wav_duration(path: Path) -> float:
    """Return the duration in seconds of an existing audio file."""
    return float(sf.info(path).duration)

"""Tests for audio tidying and the Kokoro provider, using a fake pipeline (no model download)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from shortforge.config import Settings
from shortforge.tts import create_tts
from shortforge.tts.base import EDGE_MARGIN_SECONDS, SCENE_PAUSE_SECONDS, tidy_audio
from shortforge.tts.kokoro import SAMPLE_RATE, KokoroTTS

SR = 1000


def tone(seconds: float, sr: int = SR, amplitude: float = 0.5) -> np.ndarray:
    """A sine tone standing in for speech."""
    t = np.arange(int(seconds * sr)) / sr
    return (amplitude * np.sin(2 * np.pi * 50 * t)).astype(np.float32)


def silence(seconds: float, sr: int = SR) -> np.ndarray:
    return np.zeros(int(seconds * sr), dtype=np.float32)


def test_tidy_trims_uneven_edges_and_adds_fixed_pause() -> None:
    clip = np.concatenate([silence(0.6), tone(1.0), silence(0.9)])
    tidied = tidy_audio(clip, SR)
    expected = 1.0 + 2 * EDGE_MARGIN_SECONDS + SCENE_PAUSE_SECONDS
    assert tidied.size / SR == pytest.approx(expected, abs=0.02)
    assert np.all(tidied[-int(SCENE_PAUSE_SECONDS * SR) :] == 0)


def test_tidy_gives_every_scene_the_same_pause() -> None:
    a = tidy_audio(np.concatenate([silence(0.1), tone(1.0), silence(0.2)]), SR)
    b = tidy_audio(np.concatenate([silence(0.7), tone(1.0), silence(1.5)]), SR)
    assert a.size == b.size


def test_tidy_keeps_quiet_speech_above_threshold() -> None:
    clip = np.concatenate([tone(0.2, amplitude=0.05), tone(1.0), tone(0.2, amplitude=0.05)])
    tidied = tidy_audio(clip, SR)
    assert tidied.size / SR == pytest.approx(1.4 + SCENE_PAUSE_SECONDS, abs=0.02)


def test_tidy_converts_int16() -> None:
    clip = (tone(1.0) * 32767).astype(np.int16)
    tidied = tidy_audio(clip, SR)
    assert tidied.dtype == np.float32
    assert np.abs(tidied).max() == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("bad", [np.zeros(1000, dtype=np.float32), np.array([], dtype=np.float32)])
def test_tidy_rejects_silent_or_empty_audio(bad: np.ndarray) -> None:
    with pytest.raises(RuntimeError):
        tidy_audio(bad, SR)


class FakePipeline:
    """Mimics KPipeline: yields results with an `.audio` attribute."""

    def __init__(self, segments: list[np.ndarray | None], error: Exception | None = None) -> None:
        self.segments = segments
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def __call__(self, text: str, **kwargs: Any) -> Any:
        self.calls.append({"text": text, **kwargs})
        if self.error:
            raise self.error
        return (SimpleNamespace(audio=segment) for segment in self.segments)


def test_kokoro_joins_segments_and_writes_wav(tmp_path: Path) -> None:
    pipeline = FakePipeline([tone(1.0, SAMPLE_RATE), None, tone(0.5, SAMPLE_RATE)])
    tts = KokoroTTS("af_heart", speed=1.1, pipeline=pipeline)
    out = tmp_path / "scene_01.wav"
    seconds = tts.synthesize("Hello there.", out)
    assert pipeline.calls == [{"text": "Hello there.", "voice": "af_heart", "speed": 1.1}]
    assert seconds == pytest.approx(1.5 + SCENE_PAUSE_SECONDS, abs=0.01)
    info = sf.info(out)
    assert (info.samplerate, info.channels, info.subtype) == (SAMPLE_RATE, 1, "PCM_16")


def test_kokoro_reports_empty_output(tmp_path: Path) -> None:
    tts = KokoroTTS("af_heart", pipeline=FakePipeline([None]))
    with pytest.raises(RuntimeError, match="no audio"):
        tts.synthesize("Hello there.", tmp_path / "scene_01.wav")


def test_kokoro_reports_unknown_voice(tmp_path: Path) -> None:
    tts = KokoroTTS("af_nobody", pipeline=FakePipeline([], error=FileNotFoundError("404")))
    with pytest.raises(RuntimeError, match="KOKORO_VOICE"):
        tts.synthesize("Hello there.", tmp_path / "scene_01.wav")


def test_kokoro_rejects_unsupported_language_prefix() -> None:
    with pytest.raises(ValueError, match="must start with"):
        KokoroTTS("zf_xiaobei")


def test_kokoro_lang_code_follows_voice() -> None:
    assert KokoroTTS("bm_george").lang_code == "b"


def test_factory_builds_kokoro_from_settings() -> None:
    tts = create_tts(Settings(_env_file=None, kokoro_voice="am_adam", kokoro_speed=1.2))
    assert isinstance(tts, KokoroTTS)
    assert (tts.voice, tts.speed, tts.lang_code) == ("am_adam", 1.2, "a")

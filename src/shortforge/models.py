"""Pydantic schemas for the generated script.

The LLM returns JSON that must validate against `Script` before anything else runs.
Limits mirror real constraints: YouTube caps titles at 100 characters and tags at
500 characters total, and the word budget keeps narration inside the 30 to 50 second
target at a typical TTS pace of about 2.5 words per second.

Pass ``context={"expected_scenes": n}`` to `Script.model_validate` to also enforce an
exact scene count (used by the ``--scenes`` CLI option).

Run `python -m shortforge.models` for a quick self-check.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

MIN_SCENES = 5
MAX_SCENES = 7
MIN_WORDS = 80
MAX_WORDS = 125
MAX_TAGS_CHARS = 500

_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['.-][A-Za-z0-9]+)*")


def count_words(text: str) -> int:
    """Count spoken words, treating tokens like "GPT-4" or "don't" as one word."""
    return len(_WORD_RE.findall(text))


class Scene(BaseModel):
    """One beat of the Short: a line of narration and the image shown while it plays."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    narration: str = Field(min_length=10, max_length=400)
    image_prompt: str = Field(min_length=10, max_length=600)


class Script(BaseModel):
    """A complete Short script plus the YouTube metadata that goes with it."""

    model_config = ConfigDict(str_strip_whitespace=True, extra="ignore")

    title: str = Field(min_length=5, max_length=100)
    description: str = Field(min_length=20, max_length=5000)
    tags: list[str] = Field(min_length=3, max_length=15)
    scenes: list[Scene] = Field(min_length=MIN_SCENES, max_length=MAX_SCENES)

    @field_validator("tags")
    @classmethod
    def _clean_tags(cls, tags: list[str]) -> list[str]:
        """Strip hashes and whitespace, drop empties and case-insensitive duplicates."""
        seen: set[str] = set()
        cleaned: list[str] = []
        for tag in tags:
            tag = tag.strip().lstrip("#").strip()
            if tag and tag.lower() not in seen:
                seen.add(tag.lower())
                cleaned.append(tag)
        if len(cleaned) < 3:
            raise ValueError("need at least 3 distinct non-empty tags")
        if sum(len(t) for t in cleaned) > MAX_TAGS_CHARS:
            raise ValueError(f"tags exceed YouTube's {MAX_TAGS_CHARS} character limit")
        return cleaned

    @model_validator(mode="after")
    def _check_length(self, info: ValidationInfo) -> Script:
        """Enforce the total word budget and, if requested, an exact scene count."""
        words = self.word_count
        if not MIN_WORDS <= words <= MAX_WORDS:
            raise ValueError(
                f"total narration is {words} words; it must be between {MIN_WORDS} and "
                f"{MAX_WORDS} words to land in the 30 to 50 second range"
            )
        expected = (info.context or {}).get("expected_scenes")
        if expected is not None and len(self.scenes) != expected:
            raise ValueError(f"expected exactly {expected} scenes, got {len(self.scenes)}")
        return self

    @property
    def word_count(self) -> int:
        """Total spoken words across all scenes."""
        return sum(count_words(scene.narration) for scene in self.scenes)


def _self_check() -> None:
    """Validate a sample script, then show that a too-short one is rejected."""
    from pydantic import ValidationError
    from rich.console import Console

    console = Console()
    line = "Vector databases store meaning as numbers so AI can find related ideas fast."
    sample = {
        "title": "Vector databases in 45 seconds",
        "description": "What a vector database is and why every AI app seems to use one.",
        "tags": ["#AI", "vector database", "RAG", "ai"],
        "scenes": [
            {"narration": f"{line} Scene {i}.", "image_prompt": "glowing points in 3D space"}
            for i in range(1, 7)
        ],
    }
    script = Script.model_validate(sample, context={"expected_scenes": 6})
    console.print(
        f"[green]valid[/] {len(script.scenes)} scenes, {script.word_count} words, "
        f"tags={script.tags}"
    )

    sample["scenes"] = sample["scenes"][:5]
    sample["scenes"][0]["narration"] = "Too short."
    try:
        Script.model_validate(sample)
    except ValidationError as exc:
        console.print(f"[green]rejected as expected[/]: {exc.errors()[0]['msg']}")
    else:
        console.print("[red]a too-short script was accepted[/]")
        raise SystemExit(1)


if __name__ == "__main__":
    _self_check()

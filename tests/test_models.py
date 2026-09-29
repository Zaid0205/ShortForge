"""Schema validation tests for the generated script."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shortforge.models import MAX_WORDS, MIN_WORDS, Script, count_words

LINE = "Vector databases store meaning as numbers so AI can find related ideas fast."


def make_script(scenes: int = 6, words_per_scene: int = 15, **overrides: object) -> dict:
    """Build a raw script payload with a controllable word count."""
    narration = " ".join(["word"] * words_per_scene)
    payload = {
        "title": "Vector databases in 45 seconds",
        "description": "What a vector database is and why every AI app uses one.",
        "tags": ["AI", "vector database", "RAG"],
        "scenes": [{"narration": narration, "image_prompt": "glowing points in space"}] * scenes,
    }
    payload.update(overrides)
    return payload


def test_count_words_keeps_compound_tokens_together() -> None:
    assert count_words("GPT-4 isn't magic, it's math.") == 5


def test_valid_script_passes() -> None:
    script = Script.model_validate(make_script())
    assert len(script.scenes) == 6
    assert script.word_count == 90


@pytest.mark.parametrize("scenes", [4, 8])
def test_scene_count_outside_range_fails(scenes: int) -> None:
    with pytest.raises(ValidationError):
        Script.model_validate(make_script(scenes=scenes, words_per_scene=18))


def test_expected_scene_count_is_enforced() -> None:
    with pytest.raises(ValidationError, match="expected exactly 7 scenes"):
        Script.model_validate(make_script(scenes=6), context={"expected_scenes": 7})


@pytest.mark.parametrize("words_per_scene", [MIN_WORDS // 6 - 1, MAX_WORDS // 6 + 2])
def test_word_budget_is_enforced(words_per_scene: int) -> None:
    with pytest.raises(ValidationError, match="total narration"):
        Script.model_validate(make_script(words_per_scene=words_per_scene))


def test_tags_are_cleaned_and_deduplicated() -> None:
    script = Script.model_validate(make_script(tags=["#AI", " ai ", "RAG", "#LLM", ""]))
    assert script.tags == ["AI", "RAG", "LLM"]


def test_too_few_distinct_tags_fails() -> None:
    with pytest.raises(ValidationError, match="3 distinct"):
        Script.model_validate(make_script(tags=["AI", "ai", "#AI"]))


def test_title_over_youtube_limit_fails() -> None:
    with pytest.raises(ValidationError):
        Script.model_validate(make_script(title="x" * 101))


def test_whitespace_is_stripped() -> None:
    script = Script.model_validate(make_script(title="  Padded title  "))
    assert script.title == "Padded title"

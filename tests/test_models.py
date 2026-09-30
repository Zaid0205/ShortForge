"""Schema validation tests for the generated script."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from shortforge.models import MAX_WORDS, MIN_WORDS, Scene, Script, clean_text, count_words

LINE = "Vector databases store meaning as numbers so AI can find related ideas fast."


def make_script(scenes: int = 6, words_per_scene: int = 15, **overrides: object) -> dict:
    """Build a raw script payload with a controllable word count."""
    narration = " ".join(["word"] * words_per_scene)
    payload = {
        "title": "Vector databases in 45 seconds",
        "description": "What a vector database is and why every AI app uses one.",
        "tags": ["AI", "vector database", "RAG"],
        "scenes": [
            {
                "narration": narration,
                "search_query": "server room",
                "image_prompt": "rows of servers in a data center",
            }
        ]
        * scenes,
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


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("vectors\u2014numeric fingerprints", "vectors, numeric fingerprints"),
        ("fast \u2014 and cheap", "fast, and cheap"),
        ("fast \u2013 and cheap", "fast, and cheap"),
        ("fast - and cheap", "fast, and cheap"),
        ("high\u2011dimensional space", "high-dimensional space"),
        ("5\u20137 scenes", "5-7 scenes"),
        ("it\u2019s \u201csmart\u201d", 'it\'s "smart"'),
        ("wait\u2026 what", "wait... what"),
        ("GPT-4 stays GPT-4", "GPT-4 stays GPT-4"),
        ("ends with a dash\u2014", "ends with a dash"),
        ("odd\u00a0 spaces ,here", "odd spaces,here"),
    ],
)
def test_clean_text(raw: str, expected: str) -> None:
    assert clean_text(raw) == expected


def test_all_text_fields_are_cleaned() -> None:
    payload = make_script(title="Vectors\u2014explained", tags=["AI", "high\u2011dim", "RAG"])
    payload["scenes"] = [
        {
            "narration": "A vector\u2014a list of numbers " + " ".join(["word"] * 12),
            "search_query": "lighthouse\u2019s beam",
            "image_prompt": "a lighthouse\u2019s beam in fog",
        }
    ] * 6
    script = Script.model_validate(payload)
    assert script.title == "Vectors, explained"
    assert script.tags[1] == "high-dim"
    assert script.scenes[0].narration.startswith("A vector, a list of numbers")
    assert script.scenes[0].image_prompt == "a lighthouse's beam in fog"
    dump = script.model_dump_json()
    assert all(ch not in dump for ch in "\u2014\u2011\u2019")


@pytest.mark.parametrize(
    "prompt",
    [
        "a person at a desk as glowing words float up from a keyboard",
        "wooden blocks, each painted with a subword fragment",
        "colorful ribbons, each ribbon labeled with a number",
        "a sunrise over a horizon made of scrolling text lines",
        "a neon sign above a doorway",
    ],
)
def test_image_prompt_asking_for_text_is_rejected(prompt: str) -> None:
    with pytest.raises(ValidationError, match="asks for visible text"):
        Scene(narration="Tokens are pieces of words.", search_query="library", image_prompt=prompt)


@pytest.mark.parametrize(
    "prompt",
    [
        "glowing beads on a string drifting through a dark room",
        "threads of light connecting floating glass cubes, context everywhere",
        "a lighthouse beam cutting through dense fog at night",
    ],
)
def test_image_prompt_without_text_is_accepted(prompt: str) -> None:
    assert (
        Scene(
            narration="Tokens are pieces of words.", search_query="library", image_prompt=prompt
        ).image_prompt
        == prompt
    )


@pytest.mark.parametrize("query", ["", "a b c d e f g", "x" * 70])
def test_search_query_must_be_short(query: str) -> None:
    with pytest.raises(ValidationError):
        Scene(narration="Tokens are pieces of words.", search_query=query, image_prompt="a library")


def test_symbols_and_logos_count_as_text() -> None:
    for prompt in ("floating matrix symbols around a chip", "a graphics card with a glowing logo"):
        with pytest.raises(ValidationError, match="asks for visible text"):
            Scene(narration="Tokens are pieces of words.", search_query="chip", image_prompt=prompt)


@pytest.mark.parametrize("narration", ["A GPU (graphics card) is fast.", "RAM [memory] is quick."])
def test_brackets_in_narration_are_rejected(narration: str) -> None:
    with pytest.raises(ValidationError, match="must not contain brackets"):
        Scene(narration=narration, search_query="computer", image_prompt="a desk with a computer")

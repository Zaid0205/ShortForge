"""Script generation tests with a fake Groq client. No network access."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import groq
import httpx
import pytest

from shortforge.config import ConfigError, Settings
from shortforge.script import (
    SCRIPT_JSON_SCHEMA,
    ScriptGenerationError,
    build_system_prompt,
    generate_script,
    load_script,
    save_script,
)


@pytest.fixture
def settings() -> Settings:
    """Settings isolated from the real .env and shell environment."""
    return Settings(_env_file=None, max_retries=3)


def script_json(scenes: int = 6, words_per_scene: int = 16) -> str:
    """Return a valid script payload as the model would send it."""
    narration = " ".join(["word"] * words_per_scene)
    return json.dumps(
        {
            "title": "RAG in 45 seconds",
            "description": "How retrieval augmented generation keeps AI answers grounded.",
            "tags": ["AI", "RAG", "LLM"],
            "scenes": [
                {
                    "narration": narration,
                    "search_query": "library shelves",
                    "image_prompt": "a quiet library with tall shelves",
                }
            ]
            * scenes,
        }
    )


def api_error(cls: type[groq.APIStatusError], status: int, message: str, **headers: str) -> Any:
    """Build a Groq SDK status error with a real httpx response attached."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status, request=request, headers=headers)
    return cls(message, response=response, body=None)


class FakeClient:
    """Returns queued replies (strings or exceptions) and records every request."""

    def __init__(self, *replies: str | Exception, finish_reason: str = "stop") -> None:
        self.replies = list(replies)
        self.finish_reason = finish_reason
        self.requests: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.requests.append(copy.deepcopy(kwargs))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        message = SimpleNamespace(content=reply)
        choice = SimpleNamespace(message=message, finish_reason=self.finish_reason)
        return SimpleNamespace(choices=[choice])


def test_valid_reply_on_first_try(settings: Settings) -> None:
    client = FakeClient(script_json())
    script = generate_script("RAG", 6, settings, client)
    assert len(script.scenes) == 6
    request = client.requests[0]
    assert request["model"] == "openai/gpt-oss-120b"
    json_schema = request["response_format"]["json_schema"]
    assert request["response_format"]["type"] == "json_schema"
    assert json_schema["strict"] is True
    assert json_schema["schema"] == SCRIPT_JSON_SCHEMA
    assert request["include_reasoning"] is False


def test_validation_errors_are_fed_back(settings: Settings) -> None:
    too_short = script_json(words_per_scene=5)
    client = FakeClient(too_short, script_json())
    script = generate_script("RAG", 6, settings, client)
    assert script.word_count == 96
    retry_messages = client.requests[1]["messages"]
    assert retry_messages[-2] == {"role": "assistant", "content": too_short}
    assert "total narration is 30 words" in retry_messages[-1]["content"]


def test_wrong_scene_count_is_rejected_then_fixed(settings: Settings) -> None:
    client = FakeClient(script_json(scenes=5, words_per_scene=19), script_json(scenes=6))
    assert len(generate_script("RAG", 6, settings, client).scenes) == 6
    assert "expected exactly 6 scenes" in client.requests[1]["messages"][-1]["content"]


def test_broken_json_is_fed_back(settings: Settings) -> None:
    client = FakeClient('{"title": ', script_json())
    generate_script("RAG", 6, settings, client)
    assert "not valid JSON" in client.requests[1]["messages"][-1]["content"]


def test_groq_json_mode_failure_counts_as_attempt(settings: Settings) -> None:
    failure = api_error(groq.BadRequestError, 400, "json_validate_failed: Failed to generate JSON")
    client = FakeClient(failure, script_json())
    assert generate_script("RAG", 6, settings, client).title == "RAG in 45 seconds"


def test_gives_up_after_max_attempts(settings: Settings) -> None:
    bad = script_json(words_per_scene=5)
    client = FakeClient(bad, bad, bad)
    with pytest.raises(ScriptGenerationError, match="after 3 attempts"):
        generate_script("RAG", 6, settings, client)


def test_rate_limit_is_retried_honoring_retry_after(settings: Settings) -> None:
    limited = api_error(groq.RateLimitError, 429, "rate limited", **{"retry-after": "0"})
    client = FakeClient(limited, script_json())
    generate_script("RAG", 6, settings, client)
    assert len(client.requests) == 2


def test_bad_key_becomes_config_error(settings: Settings) -> None:
    client = FakeClient(api_error(groq.AuthenticationError, 401, "invalid api key"))
    with pytest.raises(ConfigError, match="GROQ_API_KEY"):
        generate_script("RAG", 6, settings, client)


def test_retired_model_becomes_config_error(settings: Settings) -> None:
    client = FakeClient(api_error(groq.NotFoundError, 404, "model_not_found"))
    with pytest.raises(ConfigError, match="GROQ_MODEL"):
        generate_script("RAG", 6, settings, client)


def test_request_too_large_is_explained(settings: Settings) -> None:
    client = FakeClient(api_error(groq.APIStatusError, 413, "rate_limit_exceeded: TPM"))
    with pytest.raises(ScriptGenerationError, match="tokens-per-minute"):
        generate_script("RAG", 6, settings, client)


def test_completion_budget_fits_groq_free_tier() -> None:
    from shortforge.script import MAX_COMPLETION_TOKENS, build_system_prompt

    prompt_tokens_estimate = len(build_system_prompt("AI tools explained", 7)) // 3
    assert MAX_COMPLETION_TOKENS + prompt_tokens_estimate < 8000


def test_truncated_reply_is_reported(settings: Settings) -> None:
    client = FakeClient('{"title": "cut off', finish_reason="length")
    with pytest.raises(ScriptGenerationError, match="token limit"):
        generate_script("RAG", 6, settings, client)


def test_strict_schema_closes_every_object() -> None:
    scene = SCRIPT_JSON_SCHEMA["properties"]["scenes"]["items"]
    for obj in (SCRIPT_JSON_SCHEMA, scene):
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])


def test_missing_key_is_reported_before_any_request(settings: Settings) -> None:
    with pytest.raises(ConfigError, match="GROQ_API_KEY is not set"):
        generate_script("RAG", 6, settings)


@pytest.mark.parametrize(("topic", "scenes"), [("  ", 6), ("RAG", 4), ("RAG", 8)])
def test_bad_arguments_are_rejected(topic: str, scenes: int, settings: Settings) -> None:
    with pytest.raises(ValueError):
        generate_script(topic, scenes, settings, FakeClient())


def test_prompt_mentions_niche_and_scene_count() -> None:
    prompt = build_system_prompt("AI tools explained", 7)
    assert "AI tools explained" in prompt
    assert "Exactly 7 scenes" in prompt
    assert "technically correct" in prompt
    assert "Never ask for diagrams" in prompt
    assert "search_query" in prompt
    assert "within one second" in prompt
    assert "beads" not in prompt


def test_save_and_load_round_trip(tmp_path: Path, settings: Settings) -> None:
    script = generate_script("RAG", 6, settings, FakeClient(script_json()))
    path = tmp_path / "run" / "script.json"
    save_script(script, path)
    assert load_script(path) == script
    assert not list(path.parent.glob("*.tmp.*"))

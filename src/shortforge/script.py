"""Script generation with Groq (GPT-OSS 120B by default) and two layers of validation.

1. Groq's strict structured outputs constrain decoding to `SCRIPT_JSON_SCHEMA`, so the
   reply always has the right keys and types.
2. Pydantic then checks what a JSON schema can't express: the total word budget, the
   exact scene count and YouTube's metadata limits. If that fails, the exact errors are
   sent back as a follow-up turn so the model fixes its own output instead of starting
   over.

Transient API errors are retried separately by `call_with_retries`.

Run `python -m shortforge.script "your topic" [scenes]` to generate one and save it
to `output/_selfcheck/script.json`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Protocol

import groq
from pydantic import ValidationError

from shortforge.config import ConfigError, Settings, get_settings
from shortforge.fsutil import write_text_atomic
from shortforge.log import console, step
from shortforge.models import MAX_SCENES, MAX_WORDS, MIN_SCENES, MIN_WORDS, Script, count_words
from shortforge.retry import call_with_retries

TARGET_WORDS = (MIN_WORDS + MAX_WORDS) // 2
TEMPERATURE = 0.7
MAX_COMPLETION_TOKENS = 8192  # reasoning models spend part of this budget thinking

_STRING = {"type": "string"}
SCRIPT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "title": _STRING,
        "description": _STRING,
        "tags": {"type": "array", "items": _STRING},
        "scenes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"narration": _STRING, "image_prompt": _STRING},
                "required": ["narration", "image_prompt"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["title", "description", "tags", "scenes"],
    "additionalProperties": False,
}
"""Shape-only schema for Groq strict mode, which accepts a subset of JSON Schema.
Length and count rules live in `shortforge.models.Script`."""


class ScriptGenerationError(RuntimeError):
    """Raised when no valid script could be produced within the attempt budget."""


class ChatClient(Protocol):
    """The slice of the Groq client this module uses, so tests can pass a fake."""

    @property
    def chat(self) -> Any: ...


def build_system_prompt(niche: str, scenes: int) -> str:
    """Return the system prompt describing the channel, the rules and the JSON shape."""
    per_scene = round(TARGET_WORDS / scenes)
    return f"""You write scripts for a YouTube Shorts channel: "{niche}".
Viewers are curious non-experts scrolling fast. Be concrete and punchy, but accuracy comes first.

Fields:
- title: catchy, under 70 characters, no hashtags.
- description: one or two sentences summarizing the video.
- tags: 5 to 10 short search tags, no # symbols.
- scenes: each has "narration" (what the voiceover says) and "image_prompt" (what the image
  shows).

Rules:
- Exactly {scenes} scenes.
- Total narration between {MIN_WORDS + 5} and {MAX_WORDS - 5} words, about {per_scene} words per
  scene. Count carefully. This keeps the video between 30 and 50 seconds.
- Every claim must be technically correct. Comparisons must be fair: explain the real
  difference instead of exaggerating one side. Never invent statistics; if you are not sure a
  number is right, make the point without it. A true, specific fact beats hype.
- Scene 1 is a hook: a surprising true fact, a sharp question or a relatable problem. The last
  scene lands one clear takeaway. No "like and subscribe".
- Narration is spoken aloud by a TTS voice and shown as captions: plain sentences punctuated
  with commas and periods only, no dashes, emojis, markdown or lists. Write numbers and symbols
  as words where a voice would ("fifty percent", "C plus plus").
- image_prompt describes ONE concrete scene or visual metaphor for an image model: a physical
  subject in a setting, with lighting and mood (for example "a lone lighthouse beam cutting
  through dense fog at night"). Never ask for diagrams, charts, graphs, grids of numbers,
  icons, arrows, screens, user interfaces, split screens, timelines, text, logos or real
  people; image models render those as garbled fake text. Do not describe an art style; a
  shared style is added automatically."""


def build_user_prompt(topic: str) -> str:
    """Return the user turn for `topic`."""
    return f"Topic: {topic}"


def _feedback(errors: str) -> str:
    """Return the follow-up turn asking the model to correct its previous output."""
    return (
        "Your previous JSON did not pass validation:\n"
        f"{errors}\n"
        "Return the full corrected JSON object only, keeping everything that was fine."
    )


def _format_validation_error(exc: ValidationError) -> str:
    """Turn a Pydantic error into short bullet lines the model can act on."""
    lines = []
    for err in exc.errors():
        location = ".".join(str(part) for part in err["loc"]) or "script"
        lines.append(f"- {location}: {err['msg']}")
    return "\n".join(lines)


def generate_script(
    topic: str,
    scenes: int | None = None,
    settings: Settings | None = None,
    client: ChatClient | None = None,
) -> Script:
    """Generate and validate a Short script for `topic`.

    Args:
        topic: What the Short is about.
        scenes: Exact number of scenes (5 to 7); defaults to ``DEFAULT_SCENES``.
        settings: Settings to use; loaded from `.env` if omitted.
        client: A Groq-compatible client; created from ``GROQ_API_KEY`` if omitted.

    Returns:
        A validated `Script`.

    Raises:
        ValueError: If `topic` is empty or `scenes` is out of range.
        ConfigError: If the Groq key is missing or rejected, or the model is unavailable.
        ScriptGenerationError: If no valid script was produced within ``MAX_RETRIES`` tries.
    """
    settings = settings or get_settings()
    scenes = scenes or settings.default_scenes
    topic = topic.strip()
    if not topic:
        raise ValueError("Topic must not be empty.")
    if not MIN_SCENES <= scenes <= MAX_SCENES:
        raise ValueError(f"Scenes must be between {MIN_SCENES} and {MAX_SCENES}, got {scenes}.")
    if client is None:
        client = groq.Groq(api_key=settings.require_secret("groq_api_key"), max_retries=0)

    messages: list[dict[str, str]] = [
        {"role": "system", "content": build_system_prompt(settings.channel_niche, scenes)},
        {"role": "user", "content": build_user_prompt(topic)},
    ]
    last_problem = "no attempts made"

    for attempt in range(1, settings.max_retries + 1):
        try:
            response = call_with_retries(
                lambda: client.chat.completions.create(
                    model=settings.groq_model,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "short_script",
                            "strict": True,
                            "schema": SCRIPT_JSON_SCHEMA,
                        },
                    },
                    reasoning_effort=settings.groq_reasoning_effort,
                    include_reasoning=False,
                    temperature=TEMPERATURE,
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                ),
                what="Groq request",
                attempts=settings.max_retries,
            )
        except groq.AuthenticationError as exc:
            raise ConfigError("Groq rejected GROQ_API_KEY (401). Check the key in .env.") from exc
        except groq.NotFoundError as exc:
            raise ConfigError(
                f"Groq model {settings.groq_model!r} is not available (it may have been retired). "
                "Set GROQ_MODEL in .env to a current model from https://console.groq.com/docs/models."
            ) from exc
        except groq.BadRequestError as exc:
            if "json_validate_failed" not in str(exc):
                raise
            last_problem = "the model's reply was not valid JSON"
            messages.append({"role": "user", "content": _feedback(f"- {last_problem}")})
            console.print(f"  [yellow]![/] attempt {attempt}: {last_problem}, asking for a fix")
            continue

        choice = response.choices[0]
        content = choice.message.content or ""
        if choice.finish_reason == "length":
            raise ScriptGenerationError(
                f"Groq stopped at the {MAX_COMPLETION_TOKENS}-token limit before finishing. "
                "Lower GROQ_REASONING_EFFORT or raise MAX_COMPLETION_TOKENS."
            )
        try:
            return Script.model_validate(json.loads(content), context={"expected_scenes": scenes})
        except json.JSONDecodeError as exc:
            last_problem = f"- reply is not valid JSON ({exc.msg} at char {exc.pos})"
        except ValidationError as exc:
            last_problem = _format_validation_error(exc)

        first_line = last_problem.splitlines()[0].lstrip("- ")
        console.print(f"  [yellow]![/] attempt {attempt} invalid: {first_line}")
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": _feedback(last_problem)})

    raise ScriptGenerationError(
        f"No valid script after {settings.max_retries} attempts. Last problem:\n{last_problem}"
    )


def save_script(script: Script, path: Path) -> None:
    """Write `script` to `path` as pretty-printed JSON, atomically."""
    write_text_atomic(path, script.model_dump_json(indent=2))


def load_script(path: Path) -> Script:
    """Load and re-validate a script previously written by `save_script`."""
    return Script.model_validate_json(path.read_text(encoding="utf-8"))


def _self_check() -> None:
    """Generate a script for a topic from argv, print it, and save it for inspection."""
    from rich.table import Table

    topic = sys.argv[1] if len(sys.argv) > 1 else "What is retrieval augmented generation?"
    scenes = int(sys.argv[2]) if len(sys.argv) > 2 else None
    try:
        with step(f"Script for {topic!r}"):
            script = generate_script(topic, scenes)
    except (ConfigError, ScriptGenerationError, ValueError, groq.APIError) as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise SystemExit(1) from exc

    table = Table(title=script.title, show_lines=True)
    table.add_column("#", justify="right")
    table.add_column("Narration")
    table.add_column("Words", justify="right")
    table.add_column("Image prompt", style="dim")
    for index, scene in enumerate(script.scenes, start=1):
        words = count_words(scene.narration)
        table.add_row(str(index), scene.narration, str(words), scene.image_prompt)
    console.print(table)
    console.print(f"[bold]Description:[/] {script.description}")
    console.print(f"[bold]Tags:[/] {', '.join(script.tags)}")
    console.print(
        f"[bold]Total:[/] {script.word_count} words, about {script.estimated_seconds:.0f}s spoken"
    )

    out = get_settings().output_dir / "_selfcheck" / "script.json"
    save_script(script, out)
    console.print(f"[green]Saved[/] {out}")


if __name__ == "__main__":
    _self_check()

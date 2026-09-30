"""Script generation with Groq (GPT-OSS 120B by default): a writer pass, an editor pass,
and two layers of validation on each.

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
from collections.abc import Callable
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
TEMPERATURE = 0.4
MAX_COMPLETION_TOKENS = 4000
"""Room for hidden reasoning plus the JSON reply (a script is under 1,000 tokens). Groq counts
each request's prompt plus this limit against the free tier's 8,000 tokens per minute, so one
request always fits. The writer and editor requests together can exceed it; Groq then answers
429 with a Retry-After and the retry policy waits instead of failing."""

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
                "properties": {
                    "narration": _STRING,
                    "search_query": _STRING,
                    "image_prompt": _STRING,
                },
                "required": ["narration", "search_query", "image_prompt"],
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


_SHARED_RULES = """- Exactly {scenes} scenes.
- Total narration between {low} and {high} words, about {per_scene} words per scene. Count
  carefully. This keeps the video between 30 and 50 seconds.
- Scene 1 is the hook: one true, concrete fact or question about the topic that makes a
  viewer want the answer. It must agree with everything that follows. If it compares two
  things, the difference it names must be real and specific to them, not something both do.
- The middle scenes explain the idea in plain words, one point per scene. You may use one
  everyday comparison for the single core idea, in one or two scenes at most, and only if it
  behaves like the real thing on that point. Do not extend it to other details; explain those
  directly. Every sentence must make sense to someone who takes it literally.
- The last scene says why this matters to the viewer. No recap, no "like and subscribe".
- The viewer is a curious non-expert. Use everyday words. Leave out any technical term the
  viewer does not need. If one is needed, build its meaning into the sentence so it reads
  naturally aloud, never as a comma aside, a list of synonyms or brackets.
- Every claim must be technically correct and every comparison fair. Never invent numbers.
- Narration is spoken by a TTS voice and shown as captions: short plain sentences, commas and
  periods only, no brackets, dashes, emojis, markdown or lists. Write numbers and symbols as a
  voice would say them ("fifty percent", "C plus plus").
- Each scene's picture must show what its narration is about, so the link is obvious in one
  second.
- search_query: 2 to 4 plain words naming concrete things a camera can photograph, like a
  stock photo search. No abstract ideas, brand names or jargon. Different for every scene.
- image_prompt: a realistic photograph of that scene: subject, action, real setting, natural
  light. Never ask for diagrams, charts, icons, arrows, screen contents, user interfaces,
  logos, symbols or real people; show devices from angles where no screen is visible. Nothing
  may be written on anything: no words, letters, numbers, labels or signs, and do not use
  those words at all. Do not describe an art style; a shared style is added automatically."""

_FIELDS = """Fields:
- title: catchy, under 70 characters, no hashtags.
- description: one or two sentences summarizing the video.
- tags: 5 to 10 short search tags, no # symbols.
- scenes: each has "narration" (what the voiceover says), "search_query" (words to find a real
  stock photo for the scene) and "image_prompt" (a photo description of the scene)."""


def _rules(scenes: int) -> str:
    return _SHARED_RULES.format(
        scenes=scenes,
        low=MIN_WORDS + 5,
        high=MAX_WORDS - 5,
        per_scene=round(TARGET_WORDS / scenes),
    )


def build_system_prompt(niche: str, scenes: int) -> str:
    """Return the writer's system prompt: the channel, the fields and the rules."""
    return f"""You write scripts for a YouTube Shorts channel: "{niche}".
Viewers are curious non-experts scrolling fast. Be concrete and punchy, but accuracy comes first.

{_FIELDS}

Rules:
{_rules(scenes)}"""


def build_editor_prompt(niche: str, scenes: int) -> str:
    """Return the editor's system prompt: fix concrete problems in a draft, keep the rest."""
    return f"""You are the senior editor of a YouTube Shorts channel: "{niche}".
You receive a draft script as JSON. Read every narration line literally, as a viewer would,
and fix only real problems:
1. Hook: is scene 1 a true, concrete fact or question that the rest of the script supports?
2. Analogy: if there is a comparison, does it hold on the point it explains, and is it kept to
   one or two scenes? If it breaks there or is stretched to other details, say those details
   plainly instead.
3. Jargon: is every technical term needed, and explained so the sentence still reads
   naturally aloud? Drop terms the viewer can do without instead of tacking on definitions.
4. Accuracy: is every claim correct and every comparison fair?
5. Ending: does the last scene say why this matters instead of recapping?
6. Pictures: does each search_query and image_prompt show what its narration says?
Leave lines that already work unchanged. Return the final script as JSON in the same shape.

{_FIELDS}

Rules the final script must follow:
{_rules(scenes)}"""


def build_user_prompt(topic: str) -> str:
    """Return the writer's user turn for `topic`."""
    return f"Topic: {topic}"


def build_editor_request(topic: str, draft: Script) -> str:
    """Return the editor's user turn: the topic and the draft to improve."""
    return f"Topic: {topic}\n\nDraft script:\n{draft.model_dump_json(indent=2)}"


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


def _complete_script(
    client: ChatClient, settings: Settings, messages: list[dict[str, str]], scenes: int, label: str
) -> Script:
    """Ask the model for a script and loop on validation feedback until one passes.

    Raises:
        ConfigError: If the Groq key is rejected or the model is unavailable.
        ScriptGenerationError: If no valid script arrives within ``MAX_RETRIES`` attempts.
    """
    last_problem = "no attempts made"
    effort = settings.groq_reasoning_effort
    for attempt in range(1, settings.max_retries + 1):
        try:
            response = call_with_retries(
                lambda effort=effort: client.chat.completions.create(
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
                    reasoning_effort=effort,
                    include_reasoning=False,
                    temperature=TEMPERATURE,
                    max_completion_tokens=MAX_COMPLETION_TOKENS,
                ),
                what=f"Groq {label} request",
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
            console.print(f"  [yellow]![/] {label} attempt {attempt}: {last_problem}, retrying")
            continue
        except groq.APIStatusError as exc:
            if exc.status_code != 413:
                raise
            raise ScriptGenerationError(
                "Groq rejected the request as too large for your tokens-per-minute limit "
                f"(HTTP 413). The prompt plus MAX_COMPLETION_TOKENS ({MAX_COMPLETION_TOKENS}) "
                "in script.py must fit your Groq tier's per-minute token limit."
            ) from exc

        choice = response.choices[0]
        content = choice.message.content or ""
        if choice.finish_reason == "length":
            if effort == "low":
                raise ScriptGenerationError(
                    f"Groq stopped at the {MAX_COMPLETION_TOKENS}-token limit even with low "
                    "reasoning effort. Raise MAX_COMPLETION_TOKENS in script.py."
                )
            last_problem = f"reply cut off at {MAX_COMPLETION_TOKENS} tokens"
            console.print(
                f"  [yellow]![/] {label} attempt {attempt}: {last_problem} while reasoning, "
                "retrying with low reasoning effort"
            )
            effort = "low"
            continue
        try:
            return Script.model_validate(json.loads(content), context={"expected_scenes": scenes})
        except json.JSONDecodeError as exc:
            last_problem = f"- reply is not valid JSON ({exc.msg} at char {exc.pos})"
        except ValidationError as exc:
            last_problem = _format_validation_error(exc)

        first_line = last_problem.splitlines()[0].lstrip("- ")
        console.print(f"  [yellow]![/] {label} attempt {attempt} invalid: {first_line}")
        messages.append({"role": "assistant", "content": content})
        messages.append({"role": "user", "content": _feedback(last_problem)})

    raise ScriptGenerationError(
        f"No valid {label} script after {settings.max_retries} attempts. "
        f"Last problem:\n{last_problem}"
    )


def generate_script(
    topic: str,
    scenes: int | None = None,
    settings: Settings | None = None,
    client: ChatClient | None = None,
    on_draft: Callable[[Script], None] | None = None,
) -> Script:
    """Write a Short script for `topic`, then have an editor pass critique and rewrite it.

    The editor pass runs when ``SCRIPT_EDITOR`` is on (the default). If the editor fails to
    produce a valid script, the validated draft is used and a warning is printed, so a
    working script is never thrown away.

    Args:
        topic: What the Short is about.
        scenes: Exact number of scenes (5 to 7); defaults to ``DEFAULT_SCENES``.
        settings: Settings to use; loaded from `.env` if omitted.
        client: A Groq-compatible client; created from ``GROQ_API_KEY`` if omitted.
        on_draft: Called with the validated draft before the editor pass, for example to save
            it so the two versions can be compared.

    Returns:
        A validated `Script`.

    Raises:
        ValueError: If `topic` is empty or `scenes` is out of range.
        ConfigError: If the Groq key is missing or rejected, or the model is unavailable.
        ScriptGenerationError: If no valid draft was produced within ``MAX_RETRIES`` tries.
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

    niche = settings.channel_niche
    draft = _complete_script(
        client,
        settings,
        [
            {"role": "system", "content": build_system_prompt(niche, scenes)},
            {"role": "user", "content": build_user_prompt(topic)},
        ],
        scenes,
        label="draft",
    )
    if on_draft is not None:
        on_draft(draft)
    if not settings.script_editor:
        return draft
    console.print(f"  draft ready ({draft.word_count} words), editor pass...")
    try:
        return _complete_script(
            client,
            settings,
            [
                {"role": "system", "content": build_editor_prompt(niche, scenes)},
                {"role": "user", "content": build_editor_request(topic, draft)},
            ],
            scenes,
            label="editor",
        )
    except ScriptGenerationError as exc:
        console.print(f"  [yellow]![/] editor pass failed, keeping the draft: {exc}")
        return draft


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
    folder = get_settings().output_dir / "_selfcheck"
    drafts: list[Script] = []

    def keep_draft(draft: Script) -> None:
        drafts.append(draft)
        save_script(draft, folder / "draft.json")

    try:
        with step(f"Script for {topic!r}"):
            script = generate_script(topic, scenes, on_draft=keep_draft)
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

    if drafts and drafts[0] != script:
        pairs = zip(drafts[0].scenes, script.scenes, strict=True)
        changed = [str(i) for i, (a, b) in enumerate(pairs, start=1) if a.narration != b.narration]
        console.print(
            f"[bold]Editor rewrote narration in scenes:[/] {', '.join(changed) or 'none'}"
        )
    out = folder / "script.json"
    save_script(script, out)
    console.print(f"[green]Saved[/] {out} (draft before editing: {folder / 'draft.json'})")


if __name__ == "__main__":
    _self_check()

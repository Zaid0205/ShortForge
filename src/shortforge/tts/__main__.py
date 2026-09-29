"""Self-check: `python -m shortforge.tts` voices the last self-check script scene by scene.

Reads `output/_selfcheck/script.json` (written by `python -m shortforge.script`), or a
built-in sample if it does not exist, and writes `scene_XX.wav` next to it. Prints each
scene's duration and the real speaking rate, so the word budget can be checked against
the 30 to 50 second target.
"""

from __future__ import annotations

from rich.table import Table

from shortforge.config import ConfigError, get_settings
from shortforge.fsutil import atomic_path
from shortforge.log import console, step
from shortforge.models import count_words
from shortforge.script import load_script
from shortforge.tts import create_tts

TARGET_SECONDS = (30.0, 50.0)
SAMPLE_NARRATION = [
    "Your phone recognizes your face in a fraction of a second.",
    "It does that by turning your face into a list of numbers, called an embedding.",
]


def main() -> None:
    """Synthesize every scene and report durations."""
    settings = get_settings()
    folder = settings.output_dir / "_selfcheck"
    script_path = folder / "script.json"
    if script_path.exists():
        narrations = [scene.narration for scene in load_script(script_path).scenes]
        console.print(f"Using {script_path} ({len(narrations)} scenes)")
    else:
        narrations = SAMPLE_NARRATION
        console.print("No self-check script found, using a built-in two-line sample")

    try:
        tts = create_tts(settings)
        with step(f"Loading {tts.name} (first run downloads the model, about 330 MB)"):
            tts.synthesize("Ready.", folder / "warmup.tmp.wav")
    except (ConfigError, RuntimeError, ValueError) as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        raise SystemExit(1) from exc
    finally:
        (folder / "warmup.tmp.wav").unlink(missing_ok=True)

    table = Table(title=f"TTS: {tts.name}")
    for column in ("#", "Words", "Seconds", "Words/s", "File"):
        table.add_column(column, justify="right" if column != "File" else "left")
    total_seconds = total_words = 0.0
    with step(f"Voicing {len(narrations)} scenes"):
        for index, text in enumerate(narrations, start=1):
            out = folder / f"scene_{index:02d}.wav"
            with atomic_path(out) as tmp:
                seconds = tts.synthesize(text, tmp)
            words = count_words(text)
            total_seconds += seconds
            total_words += words
            table.add_row(
                str(index), str(words), f"{seconds:.2f}", f"{words / seconds:.2f}", str(out)
            )
    console.print(table)

    if not script_path.exists():
        console.print(f"Total {total_seconds:.1f}s for the sample")
        return
    low, high = TARGET_SECONDS
    verdict = "[green]within[/]" if low <= total_seconds <= high else "[yellow]outside[/]"
    rate = total_words / total_seconds
    console.print(
        f"Total {total_seconds:.1f}s for {total_words:.0f} words ({rate:.2f} words/s), "
        f"{verdict} the {low:.0f} to {high:.0f}s target"
    )


if __name__ == "__main__":
    main()

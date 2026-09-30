"""Command-line interface: ``shortforge "topic" [--upload] [--resume RUN_ID] [--scenes N]``.

Progress goes to stderr; only results go to stdout (the video path, then the YouTube link
when uploading), so the command composes with other tools.
"""

from __future__ import annotations

import time
from typing import Annotated

import typer
from rich.table import Table

from shortforge.config import ConfigError, get_settings
from shortforge.log import console
from shortforge.models import MAX_SCENES, MIN_SCENES
from shortforge.pipeline import Pipeline, RunResult
from shortforge.runs import RunError

app = typer.Typer(
    add_completion=False,
    help="Turn a topic into a captioned vertical Short.",
    pretty_exceptions_enable=False,
)


def _summary(result: RunResult, elapsed: float) -> Table:
    """Build the end-of-run table."""
    manifest = result.manifest
    sources = [r.image_source or "generated" for r in manifest.scene_records]
    table = Table(title=result.script.title, show_header=False)
    table.add_column(style="cyan")
    table.add_column()
    table.add_row("Run ID", manifest.run_id)
    table.add_row("Video", str(result.video))
    table.add_row("Length", f"{manifest.video_seconds or 0:.1f}s, {len(sources)} scenes")
    table.add_row("Size", f"{result.video.stat().st_size / 1_000_000:.1f} MB")
    table.add_row("Script", f"{result.script.word_count} words")
    table.add_row("Images", ", ".join(f"{sources.count(s)} {s}" for s in dict.fromkeys(sources)))
    if result.youtube_url:
        privacy = manifest.youtube_privacy or "private"
        table.add_row("YouTube", f"{result.youtube_url} ({privacy})")
    if result.credits:
        table.add_row("Credits", "\n".join(result.credits))
    table.add_row("Time", f"{elapsed:.0f}s this run")
    return table


@app.command()
def main(
    topic: Annotated[
        str | None, typer.Argument(help="What the Short is about. Omit when resuming.")
    ] = None,
    scenes: Annotated[
        int | None,
        typer.Option(
            "--scenes",
            "-n",
            min=MIN_SCENES,
            max=MAX_SCENES,
            help="Number of scenes for a new run (default: DEFAULT_SCENES in .env).",
        ),
    ] = None,
    resume: Annotated[
        str | None,
        typer.Option("--resume", "-r", help="Continue an earlier run by its ID."),
    ] = None,
    upload: Annotated[
        bool, typer.Option("--upload", "-u", help="Upload the finished video to YouTube.")
    ] = False,
    debug: Annotated[bool, typer.Option("--debug", help="Show full tracebacks on errors.")] = False,
) -> None:
    """Generate a Short for TOPIC, or finish an earlier run with --resume.

    With --upload the video also goes to YouTube (private by default) and its link is
    printed after the video path.
    """
    started = time.perf_counter()
    pipeline: Pipeline | None = None
    try:
        pipeline = Pipeline(get_settings())
        result = pipeline.run(topic, scenes, resume, upload)
    except KeyboardInterrupt:
        console.print("[yellow]Interrupted.[/]")
        _resume_hint(pipeline)
        raise typer.Exit(130) from None
    except (ConfigError, RunError) as exc:
        console.print(f"[bold red]Error:[/] {exc}")
        _resume_hint(pipeline)
        raise typer.Exit(1) from None
    except Exception as exc:
        if debug:
            raise
        console.print(f"[bold red]Error:[/] {type(exc).__name__}: {exc}")
        _resume_hint(pipeline)
        raise typer.Exit(1) from None

    console.print(_summary(result, time.perf_counter() - started))
    typer.echo(str(result.video))
    if result.youtube_url:
        typer.echo(result.youtube_url)


def _resume_hint(pipeline: Pipeline | None) -> None:
    """Tell the user how to continue the run that just stopped, if one was started."""
    if pipeline is not None and pipeline.run_id:
        console.print(
            f"Finished steps are saved. Continue with: shortforge --resume {pipeline.run_id}",
            soft_wrap=True,
        )


if __name__ == "__main__":
    app()

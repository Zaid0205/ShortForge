"""Pipeline orchestration tests with fake script, voice, image and render steps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf
from PIL import Image
from typer.testing import CliRunner

from shortforge import cli
from shortforge.config import Settings
from shortforge.images.sourcing import SceneImage
from shortforge.models import Scene, Script
from shortforge.pipeline import Pipeline
from shortforge.runs import RunError
from shortforge.video import SceneAssets

SAMPLE_RATE = 8_000


def make_script(scenes: int = 6) -> Script:
    return Script.model_validate(
        {
            "title": "RAG in 45 seconds",
            "description": "How retrieval keeps AI answers grounded.",
            "tags": ["AI", "RAG", "LLM"],
            "scenes": [
                {
                    "narration": f"Scene {i} explains one idea in plain words, so a curious "
                    "viewer can follow along without any background.",
                    "search_query": f"library shelf {i}",
                    "image_prompt": f"a quiet library aisle, view {i}, in soft daylight",
                }
                for i in range(1, scenes + 1)
            ],
        }
    )


class FakeWriter:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, topic: str, scenes: int, settings: Settings, on_draft: Any) -> Script:
        self.calls += 1
        script = make_script(scenes)
        on_draft(script)
        return script


class FakeTTS:
    def __init__(self, seconds: float = 7.0) -> None:
        self.seconds = seconds
        self.spoken: list[str] = []

    def synthesize(self, text: str, out_path: Path) -> float:
        self.spoken.append(text)
        sf.write(out_path, np.zeros(int(self.seconds * SAMPLE_RATE)), SAMPLE_RATE)
        return self.seconds


class FakeImages:
    def __init__(self, fail_on: int | None = None) -> None:
        self.fail_on = fail_on
        self.prompts: list[str] = []

    def render(self, scene: Scene, out_path: Path) -> SceneImage:
        if self.fail_on is not None and len(self.prompts) + 1 == self.fail_on:
            raise RuntimeError("Cloudflare reported a failure: capacity")
        self.prompts.append(scene.image_prompt)
        Image.new("RGB", (72, 128), "gray").save(out_path)
        return SceneImage("generated")


class FakeRender:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(
        self, scenes: list[SceneAssets], out_path: Path, settings: Settings, progress: bool
    ) -> float:
        self.calls += 1
        out_path.write_bytes(b"fake mp4")
        return sum(sf.info(scene.audio).duration for scene in scenes)


class Harness:
    """A pipeline wired to fakes, plus handles to inspect what each fake did."""

    def __init__(self, tmp_path: Path, **settings: Any) -> None:
        self.settings = Settings(_env_file=None, output_dir=tmp_path, **settings)
        self.writer, self.tts, self.images, self.render = (
            FakeWriter(),
            FakeTTS(),
            FakeImages(),
            FakeRender(),
        )

    def pipeline(self, settings: Settings | None = None) -> Pipeline:
        return Pipeline(
            settings or self.settings,
            write_script=self.writer,
            tts_factory=lambda _: self.tts,
            image_factory=lambda _: self.images,
            render=self.render,
            progress=False,
        )


@pytest.fixture
def harness(tmp_path: Path) -> Harness:
    return Harness(tmp_path)


def test_new_run_produces_every_file(harness: Harness) -> None:
    result = harness.pipeline().run("What is RAG")
    names = sorted(p.name for p in result.folder.iterdir())
    assert names == sorted(
        ["draft.json", "script.json", "run.json", "final.mp4"]
        + [f"scene_{i:02d}.{ext}" for i in range(1, 7) for ext in ("wav", "png")]
    )
    manifest = result.manifest
    assert manifest.status == "done"
    assert manifest.video_seconds == pytest.approx(42.0)
    assert set(manifest.step_seconds) == {"script", "voice", "images", "video"}
    assert not list(result.folder.glob("*.tmp.*"))


def test_resuming_a_finished_run_does_no_work(harness: Harness) -> None:
    run_id = harness.pipeline().run("What is RAG").manifest.run_id
    harness.pipeline().run(resume=run_id)
    assert harness.writer.calls == 1
    assert len(harness.tts.spoken) == 6
    assert len(harness.images.prompts) == 6
    assert harness.render.calls == 1


def test_edited_narration_redoes_only_that_scene(harness: Harness) -> None:
    result = harness.pipeline().run("What is RAG")
    path = result.folder / "script.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["scenes"][1]["narration"] = "A rewritten second line that still reads well aloud today."
    path.write_text(json.dumps(data), encoding="utf-8")

    harness.pipeline().run(resume=result.manifest.run_id)
    assert harness.tts.spoken[6:] == [data["scenes"][1]["narration"]]
    assert len(harness.images.prompts) == 6
    assert harness.render.calls == 2


def test_changed_image_style_redoes_images_but_not_voice(harness: Harness) -> None:
    run_id = harness.pipeline().run("What is RAG").manifest.run_id
    restyled = harness.settings.model_copy(update={"image_style": "black and white film photo"})
    harness.pipeline(restyled).run(resume=run_id)
    assert len(harness.tts.spoken) == 6
    assert len(harness.images.prompts) == 12
    assert harness.render.calls == 2


def test_failed_run_resumes_without_redoing_finished_work(harness: Harness) -> None:
    harness.images = FakeImages(fail_on=3)
    pipeline = harness.pipeline()
    with pytest.raises(RuntimeError, match="capacity"):
        pipeline.run("What is RAG")
    run_id = pipeline.run_id
    assert run_id is not None
    failed = pipeline.store.load(run_id)
    assert failed.status == "failed"
    assert "capacity" in (failed.error or "")

    harness.images = FakeImages()
    result = harness.pipeline().run(resume=run_id)
    assert result.manifest.status == "done"
    assert harness.writer.calls == 1
    assert len(harness.tts.spoken) == 6
    assert harness.images.prompts == [s.image_prompt for s in result.script.scenes[2:]]


def test_short_voiceover_warns_but_renders(
    harness: Harness, capsys: pytest.CaptureFixture[str]
) -> None:
    harness.tts = FakeTTS(seconds=3.0)
    result = harness.pipeline().run("What is RAG")
    assert result.video.exists()
    assert "outside the 30 to 50s target" in capsys.readouterr().err


def test_invalid_cached_script_is_explained(harness: Harness) -> None:
    result = harness.pipeline().run("What is RAG")
    (result.folder / "script.json").write_text('{"title": "x"}', encoding="utf-8")
    with pytest.raises(RunError, match="Fix it or delete it"):
        harness.pipeline().run(resume=result.manifest.run_id)


def test_argument_conflicts_are_rejected(harness: Harness) -> None:
    run_id = harness.pipeline().run("What is RAG").manifest.run_id
    with pytest.raises(RunError, match="Give a topic"):
        harness.pipeline().run()
    with pytest.raises(RunError, match="not 'Something else'"):
        harness.pipeline().run("Something else", resume=run_id)
    with pytest.raises(RunError, match="cannot change a resumed run"):
        harness.pipeline().run(scenes=7, resume=run_id)


def test_credits_are_collected_once(harness: Harness) -> None:
    class StockImages(FakeImages):
        def render(self, scene: Scene, out_path: Path) -> SceneImage:
            super().render(scene, out_path)
            return SceneImage("stock", "Photo by Ana on Pexels")

    harness.images = StockImages()
    result = harness.pipeline().run("What is RAG")
    assert result.credits == ["Photo by Ana on Pexels"]


def test_cli_prints_only_the_video_path_to_stdout(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: harness.settings)
    monkeypatch.setattr(cli, "Pipeline", lambda settings: harness.pipeline(settings))
    outcome = CliRunner().invoke(cli.app, ["What is RAG", "--scenes", "5"])
    assert outcome.exit_code == 0, outcome.output
    video = Path(outcome.stdout.strip())
    assert video.name == "final.mp4" and video.exists()
    assert len(harness.tts.spoken) == 5


def test_cli_reports_errors_and_resume_hint(
    harness: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness.images = FakeImages(fail_on=1)
    monkeypatch.setattr(cli, "get_settings", lambda: harness.settings)
    monkeypatch.setattr(cli, "Pipeline", lambda settings: harness.pipeline(settings))
    outcome = CliRunner().invoke(cli.app, ["What is RAG"])
    assert outcome.exit_code == 1
    assert "capacity" in outcome.stderr
    assert "shortforge --resume " in outcome.stderr
    assert outcome.stdout == ""


def test_cli_needs_a_topic_or_resume(harness: Harness, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "get_settings", lambda: harness.settings)
    outcome = CliRunner().invoke(cli.app, [])
    assert outcome.exit_code == 1
    assert "Give a topic" in outcome.stderr

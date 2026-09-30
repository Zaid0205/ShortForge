"""End-to-end orchestration: topic to finished Short, with step-level caching.

A run goes through four steps, each writing into ``output/<run_id>/``:

1. Script: `script.json` (plus `draft.json`, the version before the editor pass).
2. Voice: `scene_XX.wav` per scene; each duration sets that scene's length.
3. Images: `scene_XX.png` per scene.
4. Video: `final.mp4`.

A step reuses an existing file when the fingerprint stored in `run.json` still matches
its inputs (see `shortforge.runs`), so a failed or interrupted run resumes where it
stopped and never pays for the same image twice.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from shortforge.config import Settings, get_settings
from shortforge.fsutil import atomic_path
from shortforge.images.sourcing import SceneImageSource
from shortforge.log import console, step
from shortforge.models import MAX_SCENES, MIN_SCENES, Scene, Script
from shortforge.runs import RunError, RunManifest, RunStore, SceneRecord, fingerprint
from shortforge.script import generate_script, load_script, save_script
from shortforge.tts import TextToSpeech, create_tts, wav_duration
from shortforge.video import SceneAssets, assemble_video

TARGET_SECONDS = (30.0, 50.0)
"""The Short's intended length. Outside this range the run warns but still renders."""

ScriptWriter = Callable[..., Script]
Renderer = Callable[[list[SceneAssets], Path, Settings, bool], float]


@dataclass(frozen=True)
class RunResult:
    """The outcome of a finished run."""

    manifest: RunManifest
    script: Script
    folder: Path

    @property
    def video(self) -> Path:
        """Path to the finished MP4."""
        return self.folder / "final.mp4"

    @property
    def credits(self) -> list[str]:
        """Distinct credit lines for stock photos used, in scene order."""
        lines = [r.image_credit for r in self.manifest.scene_records if r.image_credit]
        return list(dict.fromkeys(lines))


def scene_file(folder: Path, index: int, suffix: str) -> Path:
    """Return the path of scene `index` (1-based) with `suffix`, for example ``scene_03.wav``."""
    return folder / f"scene_{index:02d}{suffix}"


class Pipeline:
    """Run or resume the full topic-to-video pipeline.

    Collaborators are injectable so tests can replace the network and model calls with
    fakes; by default they come from settings.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        store: RunStore | None = None,
        write_script: ScriptWriter = generate_script,
        tts_factory: Callable[[Settings], TextToSpeech] = create_tts,
        image_factory: Callable[[Settings], SceneImageSource] = SceneImageSource.from_settings,
        render: Renderer = assemble_video,
        progress: bool = True,
    ) -> None:
        self.settings = settings or get_settings()
        self.store = store or RunStore(self.settings.output_dir)
        self._write_script = write_script
        self._tts_factory = tts_factory
        self._image_factory = image_factory
        self._render = render
        self._progress = progress
        self.run_id: str | None = None
        """ID of the run most recently started or resumed, set before any step runs."""

    def run(
        self, topic: str | None = None, scenes: int | None = None, resume: str | None = None
    ) -> RunResult:
        """Produce a finished Short for `topic`, or continue the run `resume`.

        Args:
            topic: What the Short is about. Optional when resuming; must match if given.
            scenes: Number of scenes (5 to 7) for a new run; defaults to ``DEFAULT_SCENES``.
            resume: ID of an existing run to continue.

        Raises:
            RunError: If the arguments conflict or the run cannot be found.
            ConfigError: If a required key is missing or rejected.
        """
        manifest = self._open(topic, scenes, resume)
        self.run_id = manifest.run_id
        folder = self.store.folder(manifest.run_id)
        console.print(f"[bold]Run[/] {manifest.run_id} [dim]({folder})[/]")
        manifest.status, manifest.error = "running", None
        self.store.save(manifest)
        try:
            script = self._script_step(manifest, folder)
            self._voice_step(manifest, folder, script)
            self._image_step(manifest, folder, script)
            self._video_step(manifest, folder, script)
        except BaseException as exc:
            manifest.status = "failed"
            manifest.error = "interrupted" if isinstance(exc, KeyboardInterrupt) else str(exc)
            self.store.save(manifest)
            raise
        manifest.status = "done"
        self.store.save(manifest)
        return RunResult(manifest, script, folder)

    def _open(self, topic: str | None, scenes: int | None, resume: str | None) -> RunManifest:
        """Load the run to resume, or create a new one, after checking the arguments agree."""
        topic = topic.strip() if topic else None
        if resume is None:
            if not topic:
                raise RunError("Give a topic for a new run, or --resume RUN_ID.")
            scenes = scenes or self.settings.default_scenes
            if not MIN_SCENES <= scenes <= MAX_SCENES:
                raise RunError(f"Scenes must be {MIN_SCENES} to {MAX_SCENES}, got {scenes}.")
            return self.store.create(topic, scenes)

        manifest = self.store.load(resume)
        if topic and topic != manifest.topic:
            raise RunError(
                f"Run {resume} is about {manifest.topic!r}, not {topic!r}. "
                "Leave out the topic when resuming, or start a new run."
            )
        if scenes and scenes != manifest.scenes:
            raise RunError(
                f"Run {resume} has {manifest.scenes} scenes; --scenes cannot change a "
                "resumed run. Start a new run instead."
            )
        return manifest

    def _script_step(self, manifest: RunManifest, folder: Path) -> Script:
        path = folder / "script.json"
        with step("Script"):
            if path.exists():
                try:
                    script = load_script(path)
                except (ValidationError, ValueError) as exc:
                    raise RunError(
                        f"{path} no longer passes validation. Fix it or delete it to write a "
                        f"new script. Details: {exc}"
                    ) from exc
                console.print(f"  cached: {script.title!r}")
            else:
                started = time.perf_counter()
                script = self._write_script(
                    manifest.topic,
                    manifest.scenes,
                    self.settings,
                    on_draft=lambda draft: save_script(draft, folder / "draft.json"),
                )
                save_script(script, path)
                manifest.step_seconds["script"] = round(time.perf_counter() - started, 1)
                console.print(f"  {script.title!r}, {script.word_count} words")
        count = len(script.scenes)
        manifest.scenes = count
        records = manifest.scene_records[:count]
        manifest.scene_records = records + [SceneRecord() for _ in range(count - len(records))]
        self.store.save(manifest)
        return script

    def _audio_key(self, scene: Scene) -> str:
        s = self.settings
        return fingerprint(scene.narration, s.tts_provider, s.kokoro_voice, s.kokoro_speed)

    def _image_key(self, scene: Scene) -> str:
        s = self.settings
        query = scene.search_query if s.image_source == "hybrid" else ""
        return fingerprint(
            s.image_source,
            query,
            scene.image_prompt,
            s.image_provider,
            s.cloudflare_image_model,
            s.cloudflare_steps,
            s.image_style,
            s.video_width,
            s.video_height,
        )

    def _voice_step(self, manifest: RunManifest, folder: Path, script: Script) -> None:
        tts: TextToSpeech | None = None
        started, worked = time.perf_counter(), False
        with step(f"Voice for {len(script.scenes)} scenes"):
            for index, (scene, record) in enumerate(
                zip(script.scenes, manifest.scene_records, strict=True), start=1
            ):
                path = scene_file(folder, index, ".wav")
                key = self._audio_key(scene)
                if _is_current(path, record.audio_key, key):
                    record.audio_key = key
                    record.audio_seconds = wav_duration(path)
                    console.print(f"  scene {index}: {record.audio_seconds:.1f}s [dim](cached)[/]")
                    continue
                tts = tts or self._tts_factory(self.settings)
                with atomic_path(path) as tmp:
                    record.audio_seconds = round(tts.synthesize(scene.narration, tmp), 3)
                record.audio_key, worked = key, True
                self.store.save(manifest)
                console.print(f"  scene {index}: {record.audio_seconds:.1f}s")
        if worked:
            manifest.step_seconds["voice"] = round(time.perf_counter() - started, 1)
        self.store.save(manifest)

        total = sum(r.audio_seconds or 0.0 for r in manifest.scene_records)
        low, high = TARGET_SECONDS
        if not low <= total <= high:
            console.print(
                f"  [yellow]![/] voiceover is {total:.1f}s, outside the {low:.0f} to "
                f"{high:.0f}s target. Rendering anyway."
            )
        else:
            console.print(f"  total voiceover {total:.1f}s")

    def _image_step(self, manifest: RunManifest, folder: Path, script: Script) -> None:
        source: SceneImageSource | None = None
        started, worked = time.perf_counter(), False
        with step(f"Images for {len(script.scenes)} scenes"):
            for index, (scene, record) in enumerate(
                zip(script.scenes, manifest.scene_records, strict=True), start=1
            ):
                path = scene_file(folder, index, ".png")
                key = self._image_key(scene)
                if _is_current(path, record.image_key, key):
                    record.image_key = key
                    record.image_source = record.image_source or "generated"
                    console.print(f"  scene {index}: {record.image_source} [dim](cached)[/]")
                    continue
                source = source or self._image_factory(self.settings)
                scene_started = time.perf_counter()
                with atomic_path(path) as tmp:
                    result = source.render(scene, tmp)
                record.image_key, worked = key, True
                record.image_source, record.image_credit = result.source, result.credit
                self.store.save(manifest)
                console.print(
                    f"  scene {index}: {result.source}, {time.perf_counter() - scene_started:.1f}s"
                )
        if worked:
            manifest.step_seconds["images"] = round(time.perf_counter() - started, 1)
        self.store.save(manifest)

    def _video_key(self, manifest: RunManifest, script: Script) -> str:
        s = self.settings
        parts: list[object] = [s.video_width, s.video_height, s.video_fps, s.font_path.name]
        for scene, record in zip(script.scenes, manifest.scene_records, strict=True):
            parts += [scene.narration, record.audio_key, record.image_key]
        return fingerprint(*parts)

    def _video_step(self, manifest: RunManifest, folder: Path, script: Script) -> None:
        path = folder / "final.mp4"
        key = self._video_key(manifest, script)
        with step("Video"):
            if path.exists() and manifest.video_key == key:
                console.print(f"  cached: {manifest.video_seconds or 0:.1f}s")
                return
            assets = [
                SceneAssets(
                    scene.narration,
                    scene_file(folder, index, ".png"),
                    scene_file(folder, index, ".wav"),
                )
                for index, scene in enumerate(script.scenes, start=1)
            ]
            started = time.perf_counter()
            with atomic_path(path) as tmp:
                seconds = self._render(assets, tmp, self.settings, self._progress)
            manifest.video_key, manifest.video_seconds = key, round(seconds, 2)
            manifest.step_seconds["video"] = round(time.perf_counter() - started, 1)
            self.store.save(manifest)


def _is_current(path: Path, stored_key: str | None, key: str) -> bool:
    """True if `path` can be reused: it exists and was made from the same inputs.

    A file with no stored fingerprint (for example after `run.json` was lost) is trusted
    and adopted rather than regenerated, since regenerating images costs money.
    """
    return path.exists() and stored_key in (None, key)

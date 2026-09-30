# ShortForge

AI pipeline: topic → script → voiceover → images → captioned vertical Short → YouTube upload.
Portfolio project, must run on CPU only (Windows, Python 3.11, venv in `.venv`).
Full spec: see `docs/BRIEF.md`.

## Stack (fixed decisions)
- Script: Groq (GPT-OSS 120B, replaced Llama 3.3 70B after Groq retired it in Aug 2026), strict structured outputs, validated with Pydantic
- TTS: pluggable via `TTS_PROVIDER`. `kokoro` (local, needs espeak-ng). ElevenLabs deferred to future work
- Images: `IMAGE_SOURCE=generated` (default): Cloudflare Workers AI, FLUX.2 klein 4B, native 720x1280, realistic photo style. `hybrid`: Pexels stock photo per scene with FLUX fallback (needs PEXELS_API_KEY). Replicate deferred
- Captions: Pillow-rendered PNGs (no MoviePy TextClip / ImageMagick)
- Video: MoviePy 2.x + ffmpeg, 720x1280, 24fps
- Upload: YouTube Data API v3, OAuth desktop, uploads as private
- Stretch: FastMCP tool `create_short(topic, upload=False)`

## Key rules
- Per-scene TTS: scene duration = its audio duration. No Whisper.
- Every step caches to `output/<run_id>/` and reuses a file only while its input fingerprint in `run.json` matches.
- Secrets only in `.env`. Never commit `.env`, `client_secret.json`, `token.json`.
- Type hints, docstrings, small modules, pinned dependencies.
- No em dashes in README or docs.

## Commands
- Activate env: `.\.venv\Scripts\Activate.ps1`
- Run: `shortforge "topic" [--upload] [--resume RUN_ID] [--scenes N]`
- Tests: `pytest`
- Lint: `ruff check .` and `ruff format --check .`

## Working process
- Discuss and confirm the plan before writing code.
- Build one phase at a time, stop after each so I can test.
- After each phase, give the exact test command and expected output.
- After each phase, update the private development log (Claude Doc, not in repo).

## Progress
- [x] Setup: git, venv (3.11), espeak-ng, ffmpeg
- [x] Phase 1: config, models, requirements, provider interfaces
- [x] Phase 2: script generation
- [x] Phase 3: TTS
- [x] Phase 4: images
- [x] Phase 5: captions + video
- [x] Phase 6: pipeline, caching, CLI
- [x] Phase 7: YouTube upload
- [x] Phase 8: README, tests, cleanup
- [ ] Phase 9: MCP server (stretch)
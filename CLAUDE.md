# ShortForge

AI pipeline: topic → script → voiceover → images → captioned vertical Short → YouTube upload.
Portfolio project, must run on CPU only (Windows, Python 3.11, venv in `.venv`).
Full spec: see `docs/BRIEF.md`.

## Stack (fixed decisions)
- Script: Groq (GPT-OSS 120B, replaced Llama 3.3 70B after Groq retired it in Aug 2026), strict structured outputs, validated with Pydantic
- TTS: pluggable via `TTS_PROVIDER`. `kokoro` (local, needs espeak-ng). ElevenLabs deferred to future work
- Images: pluggable via `IMAGE_PROVIDER`. Default `cloudflare` (Workers AI, FLUX.1-schnell, free tier). Optional `replicate`
- Captions: Pillow-rendered PNGs (no MoviePy TextClip / ImageMagick)
- Video: MoviePy 2.x + ffmpeg, 720x1280, 24fps
- Upload: YouTube Data API v3, OAuth desktop, uploads as private
- Stretch: FastMCP tool `create_short(topic, upload=False)`

## Key rules
- Per-scene TTS: scene duration = its audio duration. No Whisper.
- Every step caches to `output/<run_id>/` and skips if output exists.
- Secrets only in `.env`. Never commit `.env`, `client_secret.json`, `token.json`.
- Type hints, docstrings, small modules, pinned dependencies.
- No em dashes in README or docs.

## Commands
- Activate env: `.\.venv\Scripts\Activate.ps1`
- Run: `python -m shortforge.cli "topic"`
- Tests: `pytest`

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
- [ ] Phase 4: images
- [ ] Phase 5: captions + video
- [ ] Phase 6: pipeline, caching, CLI
- [ ] Phase 7: YouTube upload
- [ ] Phase 8: README, tests, cleanup
- [ ] Phase 9: MCP server (stretch)
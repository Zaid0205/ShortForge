# PROJECT BRIEF: ShortForge, an AI Short-Form Video Pipeline

## Context
You are a senior AI engineer delivering a portfolio-grade project for a demanding, high-value client. Treat this as paid production work: clean architecture, reliable behavior, clear documentation, nothing hacky or half-finished. The finished repo will be reviewed by a CTO hiring for an AI & automation internship, so code quality, README quality and a working demo matter as much as features.

- Channel niche: "AI tools and tech concepts, explained in 45 seconds"
- Developer OS: Windows (PowerShell, VS Code)
- Python: 3.11.9, virtual environment in `.venv`
- Already installed: git, ffmpeg, espeak-ng
- Already configured: `.env` with API keys, `client_secret.json` for YouTube OAuth (both gitignored)

## Goal
A Python pipeline that takes a topic and produces a finished, captioned vertical Short (720x1280, 30 to 50 seconds), then optionally uploads it to YouTube. Must run on CPU only, using hosted APIs for heavy models. Target: completed in one working day. Runs at zero cost on free tiers by default.

Pipeline: topic → script (LLM) → voiceover per scene (TTS) → image per scene → captioned video with motion → YouTube upload → (stretch) MCP tool.

## Fixed Technical Decisions (do not change without asking)
| Step | Tool | Notes |
|---|---|---|
| Script | Groq API, GPT-OSS 120B (`openai/gpt-oss-120b`; Llama 3.3 70B was retired by Groq on 2026-08-16) | Strict structured output (JSON schema): title, description, tags, 5 to 7 scenes, each with `narration` and `image_prompt`. Validate with Pydantic. Retry on invalid JSON. |
| Voice | Pluggable via `TTS_PROVIDER` | Default `kokoro`: Kokoro-82M, open source, runs locally on CPU, uses espeak-ng. ElevenLabs was deferred to future work; the common interface keeps a new provider to one file. Generate audio PER SCENE and record each duration. |
| Images | Pluggable via `IMAGE_PROVIDER` | Default `cloudflare`: Cloudflare Workers AI, FLUX.1-schnell (free daily tier, needs `CLOUDFLARE_ACCOUNT_ID` and `CLOUDFLARE_API_TOKEN`). Replicate was deferred to future work; the common interface keeps a new provider to one file. 9:16 output (crop or pad if the provider returns another size). Consistent visual style suffix appended to every prompt. |
| Captions | Pillow-rendered PNG overlays | Do NOT use MoviePy TextClip (ImageMagick dependency). Bold, high-contrast, centered lower-third, 3 to 5 words per chunk. |
| Video | MoviePy 2.x + ffmpeg | 720x1280, 24fps. Slow Ken Burns zoom per image. Scene duration = that scene's audio duration. |
| Upload | YouTube Data API v3, OAuth desktop flow | Uploads default to PRIVATE (unverified API projects are restricted to private). Token cached locally in `token.json`. |
| Agent layer (stretch) | FastMCP | Expose `create_short(topic: str, upload: bool = False)` returning the output path or URL. |

## Key Design Requirements
1. **Per-scene TTS timing.** Each scene's image duration equals its audio length. Caption chunks are timed proportionally to their character length within the scene. No Whisper.
2. **Pluggable providers.** TTS and image generation each sit behind a small abstract interface with a factory that reads the provider name from config. Adding a provider must not touch pipeline code.
3. **Step-level caching.** Each run writes to `output/<run_id>/` (script.json, scene_XX.wav, scene_XX.png, final.mp4). Every step skips work if its output already exists, so a failed render never repeats paid or rate-limited calls. Support `--resume <run_id>`.
4. **Resilience.** Retries with exponential backoff on API calls, clear error messages, no silent failures. Handle free-tier rate limits gracefully.
5. **Config.** All keys and settings via `.env` plus a typed config module. Provide `.env.example`. Never hardcode secrets. `.env`, `token.json` and `client_secret.json` must stay gitignored.
6. **Logging.** Readable progress output per step (use `rich`), with timings.
7. **CLI.** `shortforge "topic" [--upload] [--resume RUN_ID] [--scenes N]` using Typer.

## Repo Structure
```
shortforge/
├── src/shortforge/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py            # Pydantic schemas (Script, Scene)
│   ├── script.py
│   ├── tts/                 # base.py, kokoro.py, __init__.py (factory)
│   ├── images/              # base.py, cloudflare.py, __init__.py (factory)
│   ├── captions.py
│   ├── video.py
│   ├── upload.py
│   ├── pipeline.py
│   ├── cli.py
│   └── mcp_server.py        # stretch
├── tests/                   # caption chunking/timing, schema validation, provider factory
├── docs/BRIEF.md
├── output/                  # gitignored
├── assets/fonts/            # one open-license bold font
├── CLAUDE.md
├── .env.example
├── .gitignore
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Code Standards
- Type hints everywhere, docstrings on public functions, small single-purpose modules.
- Each module independently testable (e.g. `python -m shortforge.tts` runs a quick self-check).
- No dead code, no commented-out blocks, no placeholder TODOs in the final version.
- Pin dependency versions. Everything must work on Windows with Python 3.11.

## README Requirements (a deliverable, not an afterthought)
- One-line pitch plus embedded demo GIF or link to a live Short
- Architecture diagram (Mermaid)
- Tech stack table and WHY each choice was made (open source vs hosted, CPU constraint, free tiers)
- Setup: API keys, espeak-ng and ffmpeg install on Windows, Google OAuth steps
- Usage examples (CLI and MCP)
- Design decisions section: per-scene timing, pluggable providers, caching, private-upload limitation
- Cost per video (zero on default free tiers)
- Future work (scheduling, multiple platforms, background music, voice selection)

## Working Process (important)
- Do NOT jump straight to code. First restate the plan, flag any risks or better alternatives, and ask me any open questions. Wait for my confirmation.
- Then build in phases, and stop after each phase so I can run and verify it:
  1. Setup: config, models, requirements, .env.example, provider interfaces
  2. Script generation (Groq) plus validation
  3. TTS per scene (Kokoro)
  4. Image generation (Cloudflare)
  5. Captions plus video assembly
  6. Pipeline orchestration, caching, CLI
  7. YouTube upload
  8. README, tests, cleanup
  9. Stretch: MCP server
- After each phase, give me the exact PowerShell command to test it and what output to expect.
- After each phase, update the Progress checklist in CLAUDE.md.

## Definition of Done
- `shortforge "some topic"` produces a watchable, correctly timed, captioned Short on a CPU-only Windows machine.
- `--upload` pushes it to YouTube as private and prints the URL.
- Public GitHub repo with polished README, demo, and 2 to 3 sample Shorts live on the channel.
- Resume bullet ready:
  "Built an end-to-end AI short-video pipeline combining open-source models (FLUX, Kokoro TTS) with LLM scripting and pluggable providers to auto-generate captioned vertical videos and publish via YouTube Data API; exposed as an MCP tool for agent-driven content creation."

## Writing Style
In README and any written content, do not use em dashes. Keep the writing natural and direct, not AI-sounding.

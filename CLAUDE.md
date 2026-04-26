# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Start backend (from repo root):**
```powershell
cd backend
uvicorn app:app --reload --port 8000
```

**Start frontend (from repo root):**
```powershell
streamlit run frontend/ui.py
```

**Kill stale backend processes on Windows before restarting:**
```powershell
Get-Process python | Stop-Process -Force
```

**Install dependencies:**
```powershell
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Required system dependency — must be on PATH:**
```powershell
winget install Gyan.FFmpeg
```

**Pending SQL migrations (run in Supabase SQL editor):**
```sql
ALTER TABLE videos ADD COLUMN transcript text;
```

## Architecture

Two-process app: FastAPI backend (`backend/app.py`, port 8000) + Streamlit frontend (`frontend/ui.py`, port 8501). All backend logic runs server-side; the frontend is a thin HTTP client calling `http://127.0.0.1:8000`.

### Data Pipeline (triggered by `POST /generate`)

```
YouTube URL
  → extractor.py: yt-dlp download → AssemblyAI speaker-diarized transcription
  → nlp.py: spaCy tokenization (length guard: reject < 20 tokens)
  → quiz_generator.py: Gemini 2.5 Flash → 10 MCQs
  → Supabase: videos + quizzes + questions tables
```

The pipeline also runs vector + clustering steps designed but not yet wired into the router:
- `vector_pipeline.py`: chunks transcript → HuggingFace `all-MiniLM-L6-v2` (384-dim) → `chunks` table (scoped by `vid_id`)
- `clustering.py`: DBSCAN on embeddings → Gemini cluster labels → `topic_label` stored back in `chunks` → RAG retrieval via Python cosine similarity

### Adaptive Quiz Engine (`Quiz_Engine.py`, `banditStrategy.py`)

Not yet wired to the API. Used via `run_quiz.py` for dev/test.

- `AdaptiveQuizSystem`: coordinates BKT + FSRS + UCB for per-topic mastery tracking and scheduling
- `QuizBandit`: Thompson Sampling (Beta distribution) over 4 quiz strategies
- **Key coupling:** `self.topics` in `AdaptiveQuizSystem` is keyed by UUID from the `topics` table, not `topic_label` strings. To bridge to `questions` table: `self.topics[topic_id].name` → `questions WHERE topic_label = name`

### Planned Router Split (spec: `docs/superpowers/specs/2026-04-23-full-pipeline-integration-design.md`)

`app.py` should shrink to ~20 lines once these routers are created:
- `backend/routers/ingestion.py` — `POST /generate` (hybrid async with `BackgroundTasks`), `GET /status/{job_id}`
- `backend/routers/quiz.py` — `GET /quiz`, `GET /quiz/{quiz_id}`
- `backend/routers/session.py` — `/session/start`, `/session/next`, `/session/answer`, `/session/end`

## Supabase Schema Key Points

- All server-side writes require `SUPABASE_SERVICE_ROLE_KEY` (bypasses RLS). `app.py` falls back to `SUPABASE_KEY` if not set.
- `clustering.py` uses `SUPABASE_ANON_KEY` — read-only operations only.
- `Quiz_Engine.py` reads `SUPABASE_KEY` directly — does not use service role.
- Table `chunks` has `vid_id UUID` (not the old `transcript_chunks` global table). All vector/clustering operations must filter by `vid_id`.

## Environment Variables (`.env` at repo root)

```
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=   # required for writes
SUPABASE_KEY=                # anon key (Quiz_Engine + clustering reads)
SUPABASE_ANON_KEY=           # alias used by clustering.py
GEMINI_API_KEY=
ASSEMBLYAI_API_KEY=
```

## Windows-Specific Notes

- Emoji in print statements requires UTF-8 stdout. `run_quiz.py` handles this with `sys.stdout.reconfigure(encoding="utf-8")` guarded by an encoding check.
- Use PowerShell `Stop-Process` to kill stale uvicorn processes — `pkill` does not work on Windows.
- Path separators: backend code uses forward slashes for file paths (cross-platform safe with Python's `os` module).

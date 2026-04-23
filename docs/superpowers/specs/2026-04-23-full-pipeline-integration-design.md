# Full Pipeline Integration Design
**Date:** 2026-04-23  
**Project:** Q-AI-ML-3 (AI Quiz Generator)  
**Approach:** API-first, router-based split, hybrid async ingestion, stateful sessions

---

## 1. File Structure

```
backend/
├── app.py                    # thin mount: env validation, supabase init, include_router calls
├── routers/
│   ├── ingestion.py          # POST /generate, GET /status/{job_id}
│   ├── quiz.py               # GET /quiz, GET /quiz/{quiz_id}
│   └── session.py            # POST /session/start, GET /session/next,
│                             # POST /session/answer, POST /session/end
├── extractor.py              # unchanged
├── nlp.py                    # unchanged
├── quiz_generator.py         # unchanged
├── Quiz_Engine.py            # unchanged
├── vector_pipeline.py        # update: write to chunks table (not transcript_chunks)
├── clustering.py             # unchanged
├── banditStrategy.py         # unchanged
└── run_quiz.py               # unchanged (dev/test use only)

frontend/
└── ui.py                     # update: add poll screen, quiz loop screen, summary screen
```

`app.py` shrinks to ~20 lines. All business logic lives in routers.

---

## 2. Ingestion Pipeline (`routers/ingestion.py`)

### In-memory job tracker
```python
job_status: dict[str, dict] = {}
# {job_id: {status: str, question_count: int, error: str}}
# status values: "transcribed" | "processing" | "done" | "failed"
```
No Supabase table needed. Lost on server restart — acceptable for dev/portfolio.

### `POST /generate` — body: `{url, user_id}`

**Sync block (returns immediately after transcription):**
1. Insert `videos` record → `video_id`
2. Generate `job_id = str(uuid4())`; set `job_status[job_id] = {status: "transcribed"}`
3. `download_audio(url)` → MP3
4. `transcribe(audio_file)` → speaker-diarized JSON
5. Save transcript to `videos` table
6. Parse plain text: join all `utt["text"]` from transcript JSON
7. `tokenize_text(plain_text)` — reject with 422 if < 20 tokens
8. Kick off `BackgroundTask(run_pipeline, video_id, job_id, plain_text)`
9. Return `{video_id, job_id, status: "transcribed", transcript}`

**Background task `run_pipeline(video_id, job_id, plain_text)`:**
1. Set `job_status[job_id] = {status: "processing"}`
2. `vector_pipeline.run_vector_pipeline(plain_text, video_id)` → writes to `chunks` table (scoped by `vid_id`)
3. `clustering.run_clustering_pipeline()` → DBSCAN → cluster labels stored in `chunks`
4. Fetch distinct `topic_labels` from `chunks` where `vid_id = video_id`
5. Create quiz record in `quizzes` table → `quiz_id`
6. For each `topic_label`:
   - `rag_retrieve(topic_label, top_k=5)` → context chunks
   - `generate_quiz(rag_context, api_key)` → 10 MCQs
   - Store each question with `quiz_id`, `topic_label` in `questions` table
7. Set `job_status[job_id] = {status: "done", question_count: total_questions}`
8. On any exception: set `{status: "failed", error: str(e)}`

### `GET /status/{job_id}`
Returns `job_status[job_id]` or 404 if not found.

---

## 3. Schema Alignment

### `vector_pipeline.py` — write to `chunks` not `transcript_chunks`
- Current: `store_in_supabase(chunks, embeddings)` writes to `transcript_chunks` (global, no `vid_id`)
- Fix: update to write to `chunks` table with `vid_id`, `chunk_index` fields
- `run_vector_pipeline(transcript, video_id)` signature gets `video_id` parameter

### `clustering.py` — scope to `chunks` table, accept `video_id`

- Current: `run_clustering_pipeline()` and `rag_retrieve()` both target `transcript_chunks` (global)
- Fix: update both to target `chunks` table, filtered by `vid_id`
- `run_clustering_pipeline(video_id)` — only clusters chunks for this video
- `rag_retrieve(query, video_id, top_k=5)` — pgvector search scoped to this video's chunks

### `questions` table — inject `topic_label` on insert
- `quiz_generator.generate_quiz()` is unchanged — returns questions without topic
- `routers/ingestion.py` injects `topic_label` (the cluster label) when storing each question
- `difficulty` and `type` columns remain null for now

---

## 4. Session Endpoints (`routers/session.py`)

### In-memory session store
```python
active_sessions: dict[str, dict] = {}
# {session_id: {user_id, quiz_id, adaptive_system, bandit, questions_answered, strategy_id}}
```
Learning state (BKT/FSRS) persists to Supabase via `AdaptiveQuizSystem`. Session metadata is in-memory only.

### `POST /session/start` — body: `{user_id, quiz_id}`
1. Fetch distinct `topic_labels` from `questions` where `quiz_id = quiz_id`
2. Look up `username` from `users` table using `user_id` (fallback: use `user_id` string directly)
3. `initialize_quiz_system(username, topic_labels, supabase_url, supabase_key)` → `AdaptiveQuizSystem`
4. Create `QuizBandit()` instance
5. `session_id = str(uuid4())`
6. Store in `active_sessions[session_id]`
7. Return `{session_id, topic_count: len(topic_labels)}`

### `GET /session/next?session_id=`
1. Look up session → 404 if not found
2. `bandit.select_strategy()` → `(strategy_id, strategy_name)` — stored in session
3. `adaptive.select_next_topic()` → UCB picks topic (FSRS `is_due()` filtered)
4. If no topic due: return `{done: true}` (all mastered or none due)
5. Fetch one unanswered question from `questions` where `topic_label = topic` and not in session's answered set
6. Return `{question_id, question_text, options, topic_label, strategy_name}`

### `POST /session/answer` — body: `{session_id, question_id, selected_option, quality_score (1-5)}`
1. Look up session → 404 if not found
2. Fetch `correct_answer` from `questions` table
3. `is_correct = correct_answer == selected_option`
4. `adaptive.submit_answer(topic_id, is_correct, quality_score)` → updates `user_topics` + writes to `review_history`
5. Insert into `answer_history` (user_id, question_id, score, quality, grade)
6. Add `question_id` to session's answered set
7. Return `{is_correct, correct_answer, mastery_before, mastery_after, next_review_minutes}`

### `POST /session/end` — body: `{session_id}`
1. Look up session → 404 if not found
2. `summary = adaptive.get_session_summary()` → per-topic mastery gains, duration
3. Calculate `reward = summary["accuracy"]` (0.0–1.0)
4. `bandit.update(strategy_id, reward)` → Beta distribution update
5. Delete session from `active_sessions`
6. Return full summary: `{topics_covered, accuracy, mastery_gains, strategy_performance}`

---

## 5. Quiz Router (`routers/quiz.py`)

Minimal changes — moves existing `/quiz` and `/answer` logic from `app.py`:

- `GET /quiz` — latest quiz with questions (existing logic, unchanged)
- `GET /quiz/{quiz_id}` — specific quiz questions
- `POST /answer` — legacy single-question answer check (keep for backward compat)

---

## 6. Streamlit Updates (`frontend/ui.py`)

Three new screens added to the existing tab structure:

**Generate screen (updated):**
- POST `/generate` → store `job_id` in session state
- Poll `GET /status/{job_id}` every 3 seconds
- Show progress bar: "Transcribing..." → "Processing..." → "Quiz ready!"
- On `status=done`: show "Start Quiz" button with `quiz_id`

**Quiz screen (new):**
- "Start Quiz" → POST `/session/start` → store `session_id`
- Loop: GET `/session/next` → display question + 4 options
- On answer select: POST `/session/answer` → show is_correct + mastery update inline
- Show live mastery progress bar per topic
- On `done: true` from `/session/next`: auto-navigate to summary

**Summary screen (new):**
- POST `/session/end` → display summary
- Per-topic mastery before/after
- Next review times (FSRS schedule)
- Strategy used + reward earned

---

## 7. Data Flow Summary

```
POST /generate (YouTube URL)
  │
  ├─ SYNC ──► transcribe ──► return {video_id, job_id}
  │
  └─ BACKGROUND ──► vector_pipeline (chunks table, scoped by vid_id)
                        ──► clustering (DBSCAN + Gemini cluster labels)
                        ──► per-cluster: rag_retrieve → generate_quiz
                        ──► store questions (with topic_label) in questions table
                        ──► job_status = "done"

Streamlit polls GET /status/{job_id} ──► "Quiz ready"

POST /session/start ──► AdaptiveQuizSystem + QuizBandit ──► session_id
GET  /session/next  ──► bandit.select_strategy + UCB.select_topic ──► question
POST /session/answer──► BKT + FSRS ──► user_topics + review_history + answer_history
POST /session/end   ──► bandit.update(reward) ──► summary
```

---

## 8. What Is Not Changed

- `extractor.py` — no changes
- `nlp.py` — no changes  
- `quiz_generator.py` — no changes
- `Quiz_Engine.py` — no changes
- `clustering.py` — updated: `run_clustering_pipeline(video_id)` and `rag_retrieve(query, video_id)` scoped to `chunks` table
- `banditStrategy.py` — no changes
- `run_quiz.py` — no changes (remains as dev/test tool)

Only files modified: `app.py` (shrinks), `vector_pipeline.py` (chunk table target), `frontend/ui.py` (new screens).  
New files created: `routers/ingestion.py`, `routers/quiz.py`, `routers/session.py`.

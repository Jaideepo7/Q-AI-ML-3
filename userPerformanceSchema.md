# User Performance Data Schema Design

## Overview

This schema tracks user learning progress across three dimensions:
1. **Topic Mastery** (BKT probabilities)
2. **Review Scheduling** (SM-2 intervals)
3. **Strategy Performance** (Thompson Sampling bandit state)

**Storage**: SQLite database (`user_performance.db`) for MVP, with potential migration to PostgreSQL for production.

---

## Entity Relationship Diagram

```
┌──────────────┐
│    Users     │
└──────┬───────┘
       │
       │ 1:N
       ↓
┌──────────────────┐      ┌─────────────────┐
│  Topic Mastery   │      │ Bandit Strategy │
│  (BKT + SM-2)    │      │   Performance   │
└────────┬─────────┘      └─────────────────┘
         │                         ↑
         │ 1:N                     │
         ↓                         │ N:1
┌──────────────────┐               │
│ Question History │───────────────┘
└────────┬─────────┘
         │
         │ N:1
         ↓
┌──────────────────┐
│  Quiz Sessions   │
└──────────────────┘
```

---

## Table Schemas

### 1. Users Table
**Purpose**: Store basic user profile information

```sql
CREATE TABLE users (
    user_id TEXT PRIMARY KEY,              -- UUID or auth provider ID
    username TEXT UNIQUE,                  -- Display name
    email TEXT UNIQUE,                     -- For notifications
    total_quizzes_taken INTEGER DEFAULT 0, -- Lifetime quiz count
    total_questions_answered INTEGER DEFAULT 0,
    overall_avg_score REAL DEFAULT 0.0,    -- Global average (0.0-1.0)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    settings_json TEXT                     -- JSON blob for user preferences
);

CREATE INDEX idx_users_last_active ON users(last_active);
```

**Sample Row**:
```json
{
  "user_id": "usr_a1b2c3d4",
  "username": "samvrith_n",
  "email": "sam@example.com",
  "total_quizzes_taken": 23,
  "total_questions_answered": 187,
  "overall_avg_score": 0.78,
  "created_at": "2026-03-15T10:30:00",
  "last_active": "2026-04-01T14:22:00",
  "settings_json": "{\"difficulty_preference\": \"adaptive\", \"daily_goal\": 5}"
}
```

---

### 2. Topic Mastery Table
**Purpose**: Track BKT knowledge state and SM-2 scheduling per topic

```sql
CREATE TABLE topic_mastery (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,                 -- e.g., "mitosis", "photosynthesis"
    video_id TEXT,                          -- Which video this topic came from
    
    -- BKT Parameters
    mastery_probability REAL DEFAULT 0.3,   -- P(L) - current knowledge estimate
    p_learn REAL DEFAULT 0.1,               -- P(T) - learning rate
    p_slip REAL DEFAULT 0.1,                -- P(S) - slip probability
    p_guess REAL DEFAULT 0.25,              -- P(G) - guess probability
    
    -- Performance Tracking
    questions_answered INTEGER DEFAULT 0,
    questions_correct INTEGER DEFAULT 0,
    current_streak INTEGER DEFAULT 0,       -- Consecutive correct answers
    
    -- SM-2 Scheduling
    easiness_factor REAL DEFAULT 2.5,       -- EF (1.3 to 2.5)
    interval_days INTEGER DEFAULT 1,        -- Current review interval
    repetition_number INTEGER DEFAULT 0,    -- How many times reviewed
    next_review_due TIMESTAMP,              -- When to review next
    last_reviewed TIMESTAMP,                -- Last quiz on this topic
    
    -- Metadata
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    UNIQUE(user_id, topic_id, video_id)
);

CREATE INDEX idx_topic_mastery_user ON topic_mastery(user_id);
CREATE INDEX idx_topic_mastery_review_due ON topic_mastery(next_review_due);
CREATE INDEX idx_topic_mastery_probability ON topic_mastery(mastery_probability);
```

**Sample Row**:
```json
{
  "id": 42,
  "user_id": "usr_a1b2c3d4",
  "topic_id": "cell_division",
  "video_id": "vid_biology_intro",
  "mastery_probability": 0.73,
  "p_learn": 0.1,
  "p_slip": 0.1,
  "p_guess": 0.25,
  "questions_answered": 8,
  "questions_correct": 6,
  "current_streak": 2,
  "easiness_factor": 2.3,
  "interval_days": 6,
  "repetition_number": 3,
  "next_review_due": "2026-04-07T00:00:00",
  "last_reviewed": "2026-04-01T14:22:00",
  "created_at": "2026-03-20T09:15:00",
  "updated_at": "2026-04-01T14:22:00"
}
```

---

### 3. Question History Table
**Purpose**: Log every question asked and answered

```sql
CREATE TABLE question_history (
    question_id TEXT PRIMARY KEY,           -- UUID
    user_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    video_id TEXT,
    session_id TEXT NOT NULL,               -- Links to quiz session
    
    -- Question Details
    question_text TEXT NOT NULL,
    question_type TEXT,                     -- 'mcq', 'short_answer', 'true_false'
    difficulty_level TEXT,                  -- 'easy', 'medium', 'hard'
    correct_answer TEXT NOT NULL,
    
    -- User Response
    user_answer TEXT,
    is_correct BOOLEAN NOT NULL,
    confidence_level INTEGER,               -- 1-5 self-rating (optional)
    response_time_seconds INTEGER,          -- Time to answer
    
    -- Context
    context_chunk_id TEXT,                  -- ChromaDB chunk ID this question came from
    strategy_used TEXT,                     -- Which bandit strategy selected this
    
    -- Metadata
    asked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES quiz_sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX idx_question_history_user ON question_history(user_id);
CREATE INDEX idx_question_history_topic ON question_history(topic_id);
CREATE INDEX idx_question_history_session ON question_history(session_id);
CREATE INDEX idx_question_history_timestamp ON question_history(asked_at);
```

**Sample Row**:
```json
{
  "question_id": "q_x7y8z9",
  "user_id": "usr_a1b2c3d4",
  "topic_id": "cell_division",
  "video_id": "vid_biology_intro",
  "session_id": "sess_2026_04_01_142200",
  "question_text": "What are the four phases of mitosis?",
  "question_type": "short_answer",
  "difficulty_level": "medium",
  "correct_answer": "Prophase, Metaphase, Anaphase, Telophase",
  "user_answer": "Prophase, Metaphase, Anaphase, Telophase",
  "is_correct": true,
  "confidence_level": 4,
  "response_time_seconds": 23,
  "context_chunk_id": "chunk_vid_bio_001_p5",
  "strategy_used": "focus_weak_topics",
  "asked_at": "2026-04-01T14:22:15"
}
```

---

### 4. Bandit Strategy Performance Table
**Purpose**: Track Thompson Sampling bandit state

```sql
CREATE TABLE bandit_strategy_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,                           -- NULL for global bandit
    strategy_id TEXT NOT NULL,              -- 'focus_weak_topics', 'mixed_difficulty', etc.
    
    -- Beta Distribution Parameters
    alpha REAL DEFAULT 1.0,                 -- Successes + prior
    beta REAL DEFAULT 1.0,                  -- Failures + prior
    
    -- Usage Statistics
    total_pulls INTEGER DEFAULT 0,          -- Times this strategy was selected
    cumulative_reward REAL DEFAULT 0.0,     -- Sum of all rewards
    
    -- Metadata
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    UNIQUE(user_id, strategy_id)
);

CREATE INDEX idx_bandit_user ON bandit_strategy_performance(user_id);
CREATE INDEX idx_bandit_strategy ON bandit_strategy_performance(strategy_id);
```

**Sample Rows**:
```json
[
  {
    "id": 1,
    "user_id": "usr_a1b2c3d4",
    "strategy_id": "focus_weak_topics",
    "alpha": 18.7,
    "beta": 6.3,
    "total_pulls": 25,
    "cumulative_reward": 18.7,
    "last_updated": "2026-04-01T14:22:00"
  },
  {
    "id": 2,
    "user_id": "usr_a1b2c3d4",
    "strategy_id": "mixed_difficulty",
    "alpha": 12.1,
    "beta": 8.9,
    "total_pulls": 21,
    "cumulative_reward": 12.1,
    "last_updated": "2026-04-01T13:45:00"
  }
]
```

---

### 5. Quiz Sessions Table
**Purpose**: Track individual quiz attempts

```sql
CREATE TABLE quiz_sessions (
    session_id TEXT PRIMARY KEY,            -- UUID
    user_id TEXT NOT NULL,
    video_id TEXT,
    
    -- Session Details
    strategy_used TEXT NOT NULL,            -- Which bandit arm was pulled
    quiz_type TEXT,                         -- 'initial', 'review', 'adaptive'
    
    -- Timing
    quiz_started TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    quiz_completed TIMESTAMP,
    duration_seconds INTEGER,               -- Total time spent
    
    -- Performance
    total_questions INTEGER DEFAULT 0,
    questions_correct INTEGER DEFAULT 0,
    score REAL DEFAULT 0.0,                 -- 0.0 to 1.0
    
    -- Topics Covered
    topics_json TEXT,                       -- JSON array of topic_ids
    
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE INDEX idx_quiz_sessions_user ON quiz_sessions(user_id);
CREATE INDEX idx_quiz_sessions_started ON quiz_sessions(quiz_started);
```

**Sample Row**:
```json
{
  "session_id": "sess_2026_04_01_142200",
  "user_id": "usr_a1b2c3d4",
  "video_id": "vid_biology_intro",
  "strategy_used": "focus_weak_topics",
  "quiz_type": "adaptive",
  "quiz_started": "2026-04-01T14:22:00",
  "quiz_completed": "2026-04-01T14:35:00",
  "duration_seconds": 780,
  "total_questions": 8,
  "questions_correct": 6,
  "score": 0.75,
  "topics_json": "[\"cell_division\", \"mitosis\", \"chromosomes\"]"
}
```

---

## Data Flow Diagrams

### 1. New Quiz Generation Flow
```
User clicks "Start Quiz"
    ↓
┌─────────────────────────────────────┐
│ 1. Load user's topic_mastery table │
│    - Filter: mastery_probability    │
│    - Filter: next_review_due        │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 2. Thompson Sampling selects        │
│    strategy from bandit_strategy    │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 3. Apply strategy logic:            │
│    - "focus_weak" → low P(L) topics │
│    - "review" → due topics          │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 4. Generate questions from ChromaDB │
│    Create new quiz_session record   │
└─────────────────────────────────────┘
```

### 2. Answer Submission & Update Flow
```
User submits answer
    ↓
┌─────────────────────────────────────┐
│ 1. Insert into question_history     │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 2. Update topic_mastery:            │
│    - BKT: Recalculate P(L)          │
│    - SM-2: Adjust EF and interval   │
│    - Update next_review_due         │
└─────────────┬───────────────────────┘
              ↓
┌─────────────────────────────────────┐
│ 3. If quiz complete:                │
│    - Update quiz_session            │
│    - Calculate session score        │
│    - Update bandit_strategy (α/β)   │
└─────────────────────────────────────┘
```

---

## Sample Queries

### Get Topics Due for Review
```sql
SELECT 
    topic_id,
    mastery_probability,
    next_review_due,
    interval_days
FROM topic_mastery
WHERE user_id = 'usr_a1b2c3d4'
  AND next_review_due <= datetime('now')
  AND mastery_probability < 0.9
ORDER BY mastery_probability ASC
LIMIT 5;
```

### Get Weak Topics (for "focus_weak" strategy)
```sql
SELECT 
    topic_id,
    mastery_probability,
    questions_answered
FROM topic_mastery
WHERE user_id = 'usr_a1b2c3d4'
  AND mastery_probability < 0.6
  AND questions_answered >= 2  -- Require minimum data
ORDER BY mastery_probability ASC
LIMIT 3;
```

### User Performance Dashboard
```sql
SELECT 
    u.username,
    u.overall_avg_score,
    COUNT(DISTINCT qs.session_id) as total_sessions,
    AVG(qs.score) as avg_session_score,
    COUNT(DISTINCT tm.topic_id) as topics_studied,
    AVG(tm.mastery_probability) as avg_topic_mastery
FROM users u
LEFT JOIN quiz_sessions qs ON u.user_id = qs.user_id
LEFT JOIN topic_mastery tm ON u.user_id = tm.user_id
WHERE u.user_id = 'usr_a1b2c3d4'
GROUP BY u.user_id;
```

### Bandit Strategy Performance Report
```sql
SELECT 
    strategy_id,
    total_pulls,
    cumulative_reward,
    ROUND(cumulative_reward / total_pulls, 3) as avg_reward,
    ROUND(alpha / (alpha + beta), 3) as expected_reward
FROM bandit_strategy_performance
WHERE user_id = 'usr_a1b2c3d4'
ORDER BY expected_reward DESC;
```

---

## Database Initialization Script

```sql
-- user_performance_init.sql
-- Run this to create the database schema

PRAGMA foreign_keys = ON;

-- Users table
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    username TEXT UNIQUE,
    email TEXT UNIQUE,
    total_quizzes_taken INTEGER DEFAULT 0,
    total_questions_answered INTEGER DEFAULT 0,
    overall_avg_score REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    settings_json TEXT
);

-- Topic mastery table
CREATE TABLE IF NOT EXISTS topic_mastery (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    video_id TEXT,
    mastery_probability REAL DEFAULT 0.3,
    p_learn REAL DEFAULT 0.1,
    p_slip REAL DEFAULT 0.1,
    p_guess REAL DEFAULT 0.25,
    questions_answered INTEGER DEFAULT 0,
    questions_correct INTEGER DEFAULT 0,
    current_streak INTEGER DEFAULT 0,
    easiness_factor REAL DEFAULT 2.5,
    interval_days INTEGER DEFAULT 1,
    repetition_number INTEGER DEFAULT 0,
    next_review_due TIMESTAMP,
    last_reviewed TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    UNIQUE(user_id, topic_id, video_id)
);

-- Quiz sessions table
CREATE TABLE IF NOT EXISTS quiz_sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    video_id TEXT,
    strategy_used TEXT NOT NULL,
    quiz_type TEXT,
    quiz_started TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    quiz_completed TIMESTAMP,
    duration_seconds INTEGER,
    total_questions INTEGER DEFAULT 0,
    questions_correct INTEGER DEFAULT 0,
    score REAL DEFAULT 0.0,
    topics_json TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- Question history table
CREATE TABLE IF NOT EXISTS question_history (
    question_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    video_id TEXT,
    session_id TEXT NOT NULL,
    question_text TEXT NOT NULL,
    question_type TEXT,
    difficulty_level TEXT,
    correct_answer TEXT NOT NULL,
    user_answer TEXT,
    is_correct BOOLEAN NOT NULL,
    confidence_level INTEGER,
    response_time_seconds INTEGER,
    context_chunk_id TEXT,
    strategy_used TEXT,
    asked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (session_id) REFERENCES quiz_sessions(session_id) ON DELETE CASCADE
);

-- Bandit strategy performance table
CREATE TABLE IF NOT EXISTS bandit_strategy_performance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT,
    strategy_id TEXT NOT NULL,
    alpha REAL DEFAULT 1.0,
    beta REAL DEFAULT 1.0,
    total_pulls INTEGER DEFAULT 0,
    cumulative_reward REAL DEFAULT 0.0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    UNIQUE(user_id, strategy_id)
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_users_last_active ON users(last_active);
CREATE INDEX IF NOT EXISTS idx_topic_mastery_user ON topic_mastery(user_id);
CREATE INDEX IF NOT EXISTS idx_topic_mastery_review_due ON topic_mastery(next_review_due);
CREATE INDEX IF NOT EXISTS idx_topic_mastery_probability ON topic_mastery(mastery_probability);
CREATE INDEX IF NOT EXISTS idx_quiz_sessions_user ON quiz_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_quiz_sessions_started ON quiz_sessions(quiz_started);
CREATE INDEX IF NOT EXISTS idx_question_history_user ON question_history(user_id);
CREATE INDEX IF NOT EXISTS idx_question_history_topic ON question_history(topic_id);
CREATE INDEX IF NOT EXISTS idx_question_history_session ON question_history(session_id);
CREATE INDEX IF NOT EXISTS idx_question_history_timestamp ON question_history(asked_at);
CREATE INDEX IF NOT EXISTS idx_bandit_user ON bandit_strategy_performance(user_id);
CREATE INDEX IF NOT EXISTS idx_bandit_strategy ON bandit_strategy_performance(strategy_id);
```

---

## Integration with ChromaDB

**Metadata Storage Strategy**:
Store minimal tracking data in ChromaDB metadata fields to link questions back to performance data.

```python
# Example: When storing chunks in ChromaDB
collection.add(
    documents=[chunk_text],
    metadatas=[{
        "video_id": "vid_biology_intro",
        "topic_id": "cell_division",
        "chunk_index": 5,
        "difficulty_estimate": "medium"  # From NLP analysis
    }],
    ids=[chunk_id]
)

# When generating questions, retrieve and log
results = collection.query(
    query_texts=["Questions about mitosis"],
    n_results=3
)

# Log in question_history table
for result in results:
    context_chunk_id = result['id']
    # ... create question, store with context_chunk_id
```

---

## Backup & Export Strategy

### Daily Backup
```python
import sqlite3
import datetime

def backup_database():
    conn = sqlite3.connect('user_performance.db')
    backup_name = f"backup_{datetime.date.today()}.db"
    backup = sqlite3.connect(backup_name)
    conn.backup(backup)
    backup.close()
    conn.close()
```

### JSON Export for Analytics
```python
def export_user_data(user_id):
    """Export all user data as JSON for analysis or migration."""
    conn = sqlite3.connect('user_performance.db')
    
    data = {
        "user": query_to_dict("SELECT * FROM users WHERE user_id = ?", user_id),
        "topics": query_to_list("SELECT * FROM topic_mastery WHERE user_id = ?", user_id),
        "sessions": query_to_list("SELECT * FROM quiz_sessions WHERE user_id = ?", user_id),
        "questions": query_to_list("SELECT * FROM question_history WHERE user_id = ?", user_id),
        "bandit": query_to_list("SELECT * FROM bandit_strategy_performance WHERE user_id = ?", user_id)
    }
    
    return data
```

---

## Privacy & GDPR Compliance

### Data Retention Policy
- Question history: Keep for 1 year, then anonymize (remove user_id, keep aggregate stats)
- Session data: Keep for 6 months
- Topic mastery: Keep indefinitely (user progress)
- User profile: Keep until account deletion

### User Data Deletion
```sql
-- When user requests account deletion, cascade will handle most cleanup
DELETE FROM users WHERE user_id = 'usr_to_delete';

-- Anonymize historical data for analytics
UPDATE question_history 
SET user_id = 'anonymous'
WHERE user_id = 'usr_to_delete';
```

---

## Future Extensions

### Potential Schema Additions:
1. **Study Groups Table**: Track shared quizzes and group performance
2. **Achievement Badges**: Gamification milestones
3. **Daily Streaks**: Track consecutive days of learning
4. **Topic Prerequisites**: Graph of topic dependencies
5. **Video Metadata**: Store video embeddings, transcripts separately

### Migration to PostgreSQL:
When scaling beyond SQLite:
- Change `AUTOINCREMENT` to `SERIAL`
- Add `JSONB` columns for structured metadata
- Use `ARRAY` types for topics_json
- Add full-text search indexes

---

**End of Schema Design**

This schema is ready for implementation in Week 1. Start with SQLite for simplicity, then migrate to Postgres when needed.

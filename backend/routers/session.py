from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from uuid import uuid4
from datetime import datetime
import os
from dotenv import load_dotenv, find_dotenv
from supabase import create_client, Client

load_dotenv(find_dotenv())

from Quiz_Engine import (
    BKTEngine, FSRSEngine, UCBTopicSelector,
    BKTParameters, FSRSCard, Rating, Topic,
)
from banditStrategy import QuizBandit

router = APIRouter(prefix="/session", tags=["session"])

# In-memory session store — lost on server restart, acceptable for dev/portfolio
active_sessions: dict = {}

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = (
    os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
    or os.environ.get("SUPABASE_KEY")
)

_bkt = BKTEngine()
_fsrs = FSRSEngine()
_ucb = UCBTopicSelector()


def _client() -> Client:
    return create_client(SUPABASE_URL, SUPABASE_KEY)


class SessionStart(BaseModel):
    user_id: str
    quiz_id: str


class SessionAnswer(BaseModel):
    session_id: str
    question_id: str
    selected_option: str
    quality_score: int = 5  # backend overrides to 1 on incorrect answers


class SessionEnd(BaseModel):
    session_id: str

class SingleAnswer(BaseModel):
    question_id: str
    selected_option: str

class SessionSubmitAll(BaseModel):
    session_id: str
    answers: list[SingleAnswer]


@router.post("/start")
async def session_start(data: SessionStart):
    client = _client()

    # Fetch distinct topic_labels for this quiz (null → "General")
    res = (
        client.table("questions")
        .select("topic_label")
        .eq("quiz_id", data.quiz_id)
        .execute()
    )
    raw_labels = {r.get("topic_label") or "General" for r in (res.data or [])}
    topic_labels = list(raw_labels) or ["General"]

    # Build Topic objects in-memory — no topics/user_topics DB tables needed
    topics: dict[str, Topic] = {
        label: Topic(
            id=label,
            name=label,
            description=None,
            bkt=BKTParameters(),
            fsrs=FSRSCard(),
            times_selected=0,
        )
        for label in topic_labels
    }

    session_id = str(uuid4())
    active_sessions[session_id] = {
        "user_id": data.user_id,
        "quiz_id": data.quiz_id,
        "topics": topics,
        "bandit": QuizBandit(),
        "answered": set(),
        "strategy_id": None,
        "current_topic_id": None,
        "total_questions": 0,
        "correct_count": 0,
        "session_start": datetime.now(),
    }

    return {
        "session_id": session_id,
        "topic_count": len(topic_labels),
        "topics": topic_labels,
    }


@router.get("/next")
async def session_next(session_id: str):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    topics: dict[str, Topic] = session["topics"]
    bandit: QuizBandit = session["bandit"]

    # Thompson Sampling picks strategy for this question
    strategy_id, strategy_name = bandit.select_strategy()
    session["strategy_id"] = strategy_id

    # UCB picks which topic to study next.
    # Within a session we ignore FSRS due-time (that's for cross-session scheduling)
    # by passing a far-future timestamp so every topic is always "due".
    from datetime import timedelta
    far_future = datetime.now() + timedelta(days=365)
    topic_id = _ucb.select_topic(
        topics, session["total_questions"], _fsrs, far_future
    )
    if not topic_id:
        return {"done": True}

    topics[topic_id].times_selected += 1
    session["total_questions"] += 1
    session["current_topic_id"] = topic_id
    topic_name = topics[topic_id].name

    # Fetch unanswered questions for this topic
    client = _client()
    q = (
        client.table("questions")
        .select("id, question_text, options, correct_answer")
        .eq("quiz_id", session["quiz_id"])
    )
    if topic_name != "General":
        q = q.eq("topic_label", topic_name)
    res = q.execute()

    available = [row for row in (res.data or []) if row["id"] not in session["answered"]]

    # All questions for this topic exhausted — force-mastery and try once more
    if not available:
        topics[topic_id].bkt.p_mastery = 1.0
        topic_id = _ucb.select_topic(
            topics, session["total_questions"], _fsrs, far_future
        )
        if not topic_id:
            return {"done": True}
        topics[topic_id].times_selected += 1
        session["total_questions"] += 1
        session["current_topic_id"] = topic_id
        topic_name = topics[topic_id].name
        q2 = (
            client.table("questions")
            .select("id, question_text, options, correct_answer")
            .eq("quiz_id", session["quiz_id"])
        )
        if topic_name != "General":
            q2 = q2.eq("topic_label", topic_name)
        res2 = q2.execute()
        available = [row for row in (res2.data or []) if row["id"] not in session["answered"]]
        if not available:
            return {"done": True}

    question = available[0]
    return {
        "done": False,
        "question_id": question["id"],
        "question_text": question["question_text"],
        "options": question["options"],
        "topic_label": topic_name,
        "strategy_name": strategy_name,
        "mastery": round(topics[topic_id].bkt.p_mastery * 100, 1),
    }


@router.post("/answer")
async def session_answer(data: SessionAnswer):
    session = active_sessions.get(data.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    client = _client()
    res = (
        client.table("questions")
        .select("correct_answer")
        .eq("id", data.question_id)
        .single()
        .execute()
    )
    if not res.data:
        raise HTTPException(status_code=404, detail="Question not found.")

    correct_answer = res.data["correct_answer"]
    is_correct = correct_answer == data.selected_option

    # Update BKT + FSRS
    topic_id: str = session["current_topic_id"]
    topic: Topic = session["topics"][topic_id]
    mastery_before = topic.bkt.p_mastery

    quality = data.quality_score if is_correct else 1
    _bkt.update(topic.bkt, is_correct)
    rating = Rating.from_quality_score(quality)
    topic.fsrs = _fsrs.schedule(topic.fsrs, rating, datetime.now())
    mastery_after = topic.bkt.p_mastery

    if is_correct:
        session["correct_count"] += 1

    # Write to answer_history (non-critical)
    try:
        client.table("answer_history").insert({
            "user_id": session["user_id"],
            "question_id": data.question_id,
            "score": 1.0 if is_correct else 0.0,
            "quality": quality,
            "grade": 1 if is_correct else 0,
        }).execute()
    except Exception:
        pass

    session["answered"].add(data.question_id)

    return {
        "is_correct": is_correct,
        "correct_answer": correct_answer,
        "mastery_before": round(mastery_before * 100, 1),
        "mastery_after": round(mastery_after * 100, 1),
        "next_review_minutes": round(topic.fsrs.next_review_minutes, 1),
    }


@router.post("/end")
async def session_end(data: SessionEnd):
    session = active_sessions.get(data.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    topics: dict[str, Topic] = session["topics"]
    bandit: QuizBandit = session["bandit"]
    total = session["total_questions"]
    correct = session["correct_count"]
    accuracy = correct / total if total > 0 else 0.0
    duration = (datetime.now() - session["session_start"]).total_seconds() / 60

    strategy_id = session.get("strategy_id")
    if strategy_id is not None:
        bandit.update(strategy_id, accuracy)

    topic_summary = [
        {
            "topic": t.name,
            "mastery": round(t.bkt.p_mastery * 100, 1),
            "mastered": _bkt.is_mastered(t.bkt),
            "next_review_minutes": round(t.fsrs.next_review_minutes, 1),
        }
        for t in topics.values()
    ]

    del active_sessions[data.session_id]

    return {
        "questions_answered": total,
        "correct": correct,
        "accuracy": round(accuracy * 100, 1),
        "duration_minutes": round(duration, 1),
        "topics": topic_summary,
    }


@router.post("/submit_all")
async def session_submit_all(data: SessionSubmitAll):
    """Submit all answers at once — BKT/FSRS still update per answer sequentially."""
    session = active_sessions.get(data.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    client = _client()
    topics: dict[str, Topic] = session["topics"]
    results = []

    for ans in data.answers:
        res = (
            client.table("questions")
            .select("correct_answer, topic_label")
            .eq("id", ans.question_id)
            .single()
            .execute()
        )
        if not res.data:
            continue

        correct_answer = res.data["correct_answer"]
        topic_label = res.data.get("topic_label") or "General"
        is_correct = correct_answer == ans.selected_option

        topic = topics.get(topic_label) or topics.get("General")
        if not topic:
            continue

        quality = 5 if is_correct else 1
        _bkt.update(topic.bkt, is_correct)
        rating = Rating.from_quality_score(quality)
        topic.fsrs = _fsrs.schedule(topic.fsrs, rating, datetime.now())

        if is_correct:
            session["correct_count"] += 1
        session["total_questions"] += 1

        try:
            client.table("answer_history").insert({
                "user_id": session["user_id"],
                "question_id": ans.question_id,
                "score": 1.0 if is_correct else 0.0,
                "quality": quality,
                "grade": 1 if is_correct else 0,
            }).execute()
        except Exception:
            pass

        results.append({
            "question_id": ans.question_id,
            "is_correct": is_correct,
            "correct_answer": correct_answer,
            "mastery_after": round(topic.bkt.p_mastery * 100, 1),
        })

    # Build summary
    total = session["total_questions"]
    correct = session["correct_count"]
    accuracy = correct / total if total > 0 else 0.0
    duration = (datetime.now() - session["session_start"]).total_seconds() / 60

    strategy_id = session.get("strategy_id")
    if strategy_id is not None:
        session["bandit"].update(strategy_id, accuracy)

    topic_summary = [
        {
            "topic": t.name,
            "mastery": round(t.bkt.p_mastery * 100, 1),
            "mastered": _bkt.is_mastered(t.bkt),
            "next_review_minutes": round(t.fsrs.next_review_minutes, 1),
        }
        for t in topics.values()
    ]

    del active_sessions[data.session_id]

    return {
        "results": results,
        "questions_answered": total,
        "correct": correct,
        "accuracy": round(accuracy * 100, 1),
        "duration_minutes": round(duration, 1),
        "topics": topic_summary,
    }

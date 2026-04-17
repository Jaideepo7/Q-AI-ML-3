# Fast API backend for Q-AI Project
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from typing import List
app = FastAPI(title = "Q-AI Project API")

# --- Pydantic Models ---
class QuizQuestion(BaseModel):
    id: int
    question: str
    options: List[str]
    answer: str

class QuizResponse(BaseModel):
    video_name: str
    questions: List[QuizQuestion]

class videoURL(BaseModel):
    url: HttpUrl

class AnswerSubmission(BaseModel):
    user_id: str
    question_id: int
    selected_option: str

# --- API Endpoints ---

@app.get("/")
async def root():
    return {"message": "Welcome to the Q-AI Project API!"}

@app.post("/generate")
async def generate_from_url(data: videoURL):
    # log the URL and confirm receipt
    print(f"Received URL for processing: {data.url}")
    
    return {
        "status": "success",
        "received_url": data.url,
        "message": "URL validated. Ready for transcription pipeline."
    }

@app.get("/quiz", response_model = QuizResponse)
async def get_quiz():
    # Sample quiz data for testing
    sample_quiz = QuizResponse(
        video_name="Sample Video",
        questions=[
            QuizQuestion(
                id=1,
                question="What is the capital of France?",
                options=["Paris", "London", "Berlin", "Madrid"],
                answer="Paris"
            ),
            QuizQuestion(
                id=2,
                question="What is 2 + 2?",
                options=["3", "4", "5", "6"],
                answer="4"
            )
        ]
    )
    return sample_quiz

@app.post("/answer")
async def submit_answer(submission: AnswerSubmission):
    # Placeholder for answer processing logic
    print(f"Received answer submission: {submission}")
    
    # Simulate checking the answer (in a real implementation, this would involve more complex logic)
    correct_answers = {
        1: "Paris",
        2: "4"
    }
    
    is_correct = correct_answers.get(submission.question_id) == submission.selected_option
    
    return {
        "user_id": submission.user_id,
        "question_id": submission.question_id,
        "selected_option": submission.selected_option,
        "is_correct": is_correct,
        "message": "Answer received and processed."
    }

@app.get("/progress")
async def check_progress(user_id: str):
    # Placeholder for BKT/IRT logic to determine progress
    return {
        "user_id": user_id, 
        "progress": "75%", 
        "message": "User is making good progress!"}
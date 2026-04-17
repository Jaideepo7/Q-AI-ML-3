# Fast API backend for Q-AI Project
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl
from typing import List
import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)

app = FastAPI(title = "Q-AI Project API")

# --- Pydantic Models ---
class QuizQuestion(BaseModel):
    id: str # Change from int to str to support UUIDs
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
    question_id: str
    selected_option: str

# --- API Endpoints ---

@app.get("/")
async def root():
    return {"message": "Welcome to the Q-AI Project API!"}

@app.post("/generate")
async def generate_from_url(data: videoURL):
    # For now, we use a hardcoded user_id until you have Auth set up
    temp_user_id = "00000000-0000-0000-0000-000000000000" 
    
    try:
        # Step 1: Create Video Entry
        video_entry = supabase.table("videos").insert({
            "user_id": temp_user_id,
            "youtube_url": str(data.url)
        }).execute()
        
        video_id = video_entry.data[0]['id']

        # Step 2: Create a Quiz Shell linked to this video
        quiz_entry = supabase.table("quizzes").insert({
            "vid_id": video_id
        }).execute()

        return {
            "status": "success",
            "video_id": video_id,
            "quiz_id": quiz_entry.data[0]['id'],
            "message": "Database records initialized."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/quiz", response_model=QuizResponse)
async def get_quiz():
    try:
        # 1. Get the latest quiz and its linked video info
        # Using .select("*, videos(*)") is a Supabase "Join"
        quiz_res = supabase.table("quizzes").select("id, videos(youtube_url)").order("created_at", desc=True).limit(1).execute()
        
        if quiz_res.data:
            quiz_id = quiz_res.data[0]['id']
            video_name = quiz_res.data[0]['videos']['youtube_url'] # Using URL as name for now

            # 2. Get all questions linked to THIS quiz_id
            questions_res = supabase.table("questions").select("*").eq("quiz_id", quiz_id).execute()
            
            if questions_res.data:
                formatted_questions = []
                for q in questions_res.data:
                    formatted_questions.append(
                        QuizQuestion(
                            id=q['id'], # This will now be a UUID string
                            question=q['question_text'],
                            options=q['options'], # This is your JSONB column
                            answer=q['correct_answer']
                        )
                    )
                return QuizResponse(video_name=video_name, questions=formatted_questions)
                
    except Exception as e:
        print(f"DB Error: {e}")
        
    # Fallback Sample Data if DB is empty or fails
    return QuizResponse(
        video_name="Sample Video",
        questions=[QuizQuestion(id="1", question="Capital of France?", options=["Paris", "London"], answer="Paris")]
    )

@app.post("/answer")
async def submit_answer(submission: AnswerSubmission):
    try:
        # 1. Fetch the correct answer from the 'questions' table
        question_res = supabase.table("questions").select("correct_answer").eq("id", submission.question_id).single().execute()
        
        if not question_res.data:
            return {"status": "error", "message": "Question not found"}
            
        correct_answer = question_res.data['correct_answer']
        is_correct = correct_answer == submission.selected_option
        score = 1.0 if is_correct else 0.0

        # 2. LOG to answer_history table
        supabase.table("answer_history").insert({
            "user_id": submission.user_id,
            "question_id": submission.question_id,
            "score": score,
            "quality": 5 if is_correct else 1, # Placeholder for FSRS/BKT logic
            "grade": 1 if is_correct else 0
        }).execute()

        return {
            "is_correct": is_correct,
            "message": "Answer recorded in history."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
    
@app.get("/progress")
async def check_progress(user_id: str):
    # Placeholder for BKT/IRT logic to determine progress
    return {
        "user_id": user_id, 
        "progress": "75%", 
        "message": "User is making good progress!"}
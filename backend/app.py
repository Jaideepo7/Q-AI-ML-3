# Fast API backend for Q-AI Project
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, EmailStr, field_validator
from typing import List, Optional
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from extractor import download_audio, transcribe

# Load environment variables
load_dotenv()

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
if not url or not key:
    raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")

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
    user_id: Optional[str] = None

class AnswerSubmission(BaseModel):
    user_id: str
    question_id: str
    selected_option: str

class UserAuth(BaseModel):
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters.")
        return v

# --- API Endpoints ---

@app.get("/")
async def root():
    return {"message": "Welcome to the Q-AI Project API!"}

@app.post("/generate")
async def generate_from_url(data: videoURL):
    user_id = data.user_id or "00000000-0000-0000-0000-000000000000"
    video_url = str(data.url)

    try:
        # Step 1: Create video record
        video_entry = supabase.table("videos").insert({
            "user_id": user_id,
            "youtube_url": video_url
        }).execute()

        if not video_entry.data:
            raise HTTPException(status_code=500, detail="Failed to create video record.")

        video_id = video_entry.data[0]["id"]

        # Step 2: Download audio and transcribe
        audio_file = download_audio(video_url)
        transcript_json = transcribe(audio_file)

        # Step 3: Save transcript to video record
        supabase.table("videos").update({
            "transcript": transcript_json
        }).eq("id", video_id).execute()

        # Step 4: Create quiz shell linked to this video
        quiz_entry = supabase.table("quizzes").insert({
            "vid_id": video_id
        }).execute()

        return {
            "status": "success",
            "video_id": video_id,
            "quiz_id": quiz_entry.data[0]["id"],
            "transcript": transcript_json,
            "message": "Transcript complete. Ready to generate quiz."
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@app.post("/signup")
async def signup(auth: UserAuth):
    try:
        # This triggers the Supabase Auth system
        # The Trigger we just created in SQL will handle the public.users table!
        res = supabase.auth.sign_up({
            "email": auth.email,
            "password": auth.password
        })
        
        if not res.user:
            return {"status": "error", "message": "Signup failed. Please check your email or try a different password."}
        
        return {
            "status": "success", 
            "message": "User created. You can now log in!",
            "user_id": res.user.id
        }
    except Exception as e:
        return {"status": "error", "message": f"Signup error: {str(e)}"}

@app.post("/login")
async def login(auth: UserAuth):
    try:
        res = supabase.auth.sign_in_with_password({
            "email": auth.email,
            "password": auth.password
        })

        if not getattr(res, "session", None) or not getattr(res, "user", None):
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        return {
            "status": "success",
            "session": {
                "access_token": res.session.access_token,
                "user_id": res.user.id
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))
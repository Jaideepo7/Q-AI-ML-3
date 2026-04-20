# Q-AI: Video to Quiz Generator

An AI-powered application that takes a YouTube URL, transcribes its audio, and automatically generates an interactive quiz — then evaluates your answers and gives personalized feedback.

---

## How to Run

### Prerequisites

- Python 3.10+
- `ffmpeg` installed and on your PATH (required by yt-dlp for audio extraction)

### 1. Clone and install dependencies

```bash
git clone https://github.com/Jaideeep7/Q-AI-ML-3.git
cd Q-AI-ML-3
pip install -r requirements.txt
```

### 2. Set up environment variables

Create a `.env` file in the project root:

```env
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
ASSEMBLYAI_API_KEY=your_assemblyai_key
GEMINI_API_KEY=your_gemini_key
```

### 3. Run the backend

```bash
cd backend
uvicorn app:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

### 4. Run the frontend (new terminal)

```bash
cd frontend
streamlit run ui.py
```

The app will open automatically at `http://localhost:8501`.

---

## What It Does

1. **Sign up / Log in** via the Streamlit UI (backed by Supabase Auth)
2. **Paste a YouTube URL** and click Generate Quiz
3. The backend **downloads the audio** with `yt-dlp` and **transcribes** it via AssemblyAI (speaker-diarized, multilingual)
4. The transcript is **stored in Supabase** and displayed in the UI
5. A **quiz is generated** from the transcript using the Gemini API
6. **Answers are evaluated** and feedback is returned

---

## Tech Stack

| Layer | Technology |
| --- | --- |
| **Language** | Python 3.10+ |
| **Frontend** | Streamlit |
| **Backend / API** | FastAPI + Uvicorn |
| **Auth & Database** | Supabase |
| **Audio Extraction** | yt-dlp + ffmpeg |
| **Speech-to-Text** | AssemblyAI |
| **Quiz Generation** | Google Gemini API |

---

## Project Structure

```text
Q-AI-ML-3/
├── backend/
│   ├── app.py            # FastAPI endpoints
│   ├── extractor.py      # Audio download + AssemblyAI transcription
│   ├── quiz_generator.py # Gemini quiz generation
│   ├── embedder.py       # Embedding pipeline
│   └── transcriber.py    # (reserved)
├── frontend/
│   └── ui.py             # Streamlit UI
├── requirements.txt
└── .env                  # Not committed — see setup above
```

# AI Quiz Generator from Videos
 
An AI-powered application that takes a video file as input, transcribes its audio and visual content, and automatically generates an interactive quiz — then evaluates your answers and gives personalized feedback.
 
---
 
## What It Does
 
1. **Ingests a video file** uploaded by the user
2. **Extracts and transcribes** both audio (speech-to-text) and visual content from the video
3. **Chunks and embeds** the transcribed data into a vector database for semantic retrieval
4. **Generates a quiz** based on the video's content using an LLM
5. **Evaluates your answers** and provides targeted feedback on what to review
 
---
 
## Minimum Viable Product (MVP)
 
The MVP focuses on delivering a working end-to-end pipeline with a simple UI:
 
- Upload a video file via a Streamlit interface
- Extract and transcribe audio from the video using AssemblyAI
- Chunk and store transcriptions in ChromaDB
- Generate a 5–10 question quiz (MCQ or short answer) using the Gemini API
- Display quiz questions in the UI and accept user answers
- Return basic feedback on correct/incorrect answers
 

---
##  Possible Expansions / Stretch Goals
 
- **YouTube URL Support** — Allow users to paste a YouTube link instead of uploading a file locally, using `yt-dlp` or `pytube` to download and process the video automatically.
- **Adaptive Questioning** — Track which questions a user gets wrong and use a custom ML scoring layer to generate more questions targeting those weak areas, creating a personalized study loop.
- **Multi-language Support** — Leverage AssemblyAI's multilingual transcription models to support videos in languages other than English.
- **React Frontend** — Replace Streamlit with a full React frontend connected to the Flask REST API for a more polished, production-ready user experience.
- **Video Content Transcription** — Extend beyond audio by fully integrating visual transcription (e.g., slides, on-screen text) to enrich quiz context.
- **Progress Tracking** — Save user quiz history and scores across sessions to visualize improvement over time.
- All LLM outputs are converted to structured JSON for reliable downstream processing.

---
 
## Tech Stack
 
| Layer | Technology |
|---|---|
| **Language** | Python 3.10 |
| **Frontend** | Streamlit (React if time permits) |
| **Backend / API** | Flask |
| **Audio Extraction** | MoviePy, Pydub |
| **Speech-to-Text** | AssemblyAI *(multilingual support)* |
| **Video Transcription** | OpenAI Whisper / deepgram.ai / Riverside.ai |
| **LLM Orchestration** | LangChain |
| **Vector Database** | ChromaDB or FAISS |
| **Quiz Generation & Eval** | Google Gemini API |
 
---
 
## Rough Project Timeline
 
### Week 1 — Setup & Audio Pipeline
- Set up project structure, dependencies, and API keys
- Implement video ingestion with MoviePy
- Extract and chunk audio with Pydub (recursive chunking for large files)
- Integrate AssemblyAI for speech-to-text transcription
- Output raw transcription as structured JSON
 
### Week 2 — Video Transcription & Embeddings
- Integrate video content transcription library
- Combine audio and video transcripts into a unified document
- Semantically chunk combined transcript using LangChain
- Embed chunks and store in ChromaDB or FAISS
 
### Week 3 — Quiz Generation & Evaluation
- Connect Gemini API to generate quiz questions from retrieved chunks
- Build answer evaluation logic with LangChain structured outputs
- Return personalized feedback (what was wrong, what to review)
 
### Week 4 — Frontend & Polish
- Build Streamlit UI: video upload → quiz display → results page
- Wire Flask endpoints to the pipeline
- End-to-end testing and bug fixes
- (Stretch) Adaptive questioning based on past wrong answers
 
---


 
 


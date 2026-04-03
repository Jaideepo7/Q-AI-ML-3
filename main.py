import requests
import time
import os
import yt_dlp
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API key for AssemblyAI authentication
API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

# Headers for API requests, including authorization
headers = {"authorization": API_KEY}

def download_audio(url, output_path="downloads/"):
    os.makedirs(output_path, exist_ok=True)
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": f"{output_path}%(title)s.%(ext)s",
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
        }],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return f"{output_path}{info['title']}.mp3"

def transcribe(audio_file):
    # Step 1: Upload
    print("Uploading audio...")
    with open(audio_file, "rb") as f:
        upload_res = requests.post("https://api.assemblyai.com/v2/upload", headers=headers, data=f)
    audio_url = upload_res.json()["upload_url"]
    print("Uploaded!")

    # Step 2: Request transcription
    print("Requesting transcript...")
    transcript_res = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        headers=headers,
        json={
            "audio_url": audio_url, 
            "speech_models": ["universal-3-pro"],
            "speaker_labels": True,
            }
    )
    transcript_id = transcript_res.json()["id"]

    # Step 3: Poll until done
    while True:
        result = requests.get(f"https://api.assemblyai.com/v2/transcript/{transcript_id}", headers=headers)
        status = result.json()["status"]
        print(f"Status: {status}")
        if status == "completed":
            print("\nTranscript:\n")
            for utterance in result.json()["utterances"]:
                print(f"Speaker {utterance['speaker']}: {utterance['text']}")
            break
        elif status == "error":
            print("Error:", result.json()["error"])
            break
        time.sleep(5)

# --- Run ---
url = input("Enter audio URL: ")
audio_file = download_audio(url)
transcribe(audio_file)
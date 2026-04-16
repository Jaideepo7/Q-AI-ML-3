import requests
import time
import os
import yt_dlp
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
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

def get_significant_speakers(utterances, threshold=0.10):
    speaker_totals = {}
    for u in utterances:
        duration = u["end"] - u["start"]
        speaker_totals[u["speaker"]] = speaker_totals.get(u["speaker"], 0) + duration

    total_time = sum(speaker_totals.values())

    print("\n--- Speaker breakdown ---")
    significant = set()
    for speaker, ms in sorted(speaker_totals.items(), key=lambda x: -x[1]):
        percent = ms / total_time
        flag = " ✓ kept" if percent >= threshold else " ✗ filtered"
        print(f"  Speaker {speaker}: {ms / 1000:.1f}s ({percent:.1%}){flag}")
        if percent >= threshold:
            significant.add(speaker)

    return significant

def transcribe(audio_file):
    # Step 1: Upload
    print("Uploading audio...")
    with open(audio_file, "rb") as f:
        upload_res = requests.post(
            "https://api.assemblyai.com/v2/upload", headers=headers, data=f)
    audio_url = upload_res.json()["upload_url"]
    print("Uploaded!")

    print("Requesting transcript...")
    transcript_res = requests.post(
        "https://api.assemblyai.com/v2/transcript",
        headers=headers,
        json={
            "audio_url": audio_url,
            "speech_models": ["universal-3-pro", "universal-2"],
            "language_detection": True,
            "speaker_labels": True,
            "speaker_options": {
                "min_speakers_expected": 1,
                "max_speakers_expected": 5
            },
            "speech_understanding": {
                "request": {
                    "translation": {
                        "target_languages": ["en"],
                        "match_original_utterance": True,
                        "formal": True
                    }
                }
            }
        }
    )
    transcript_id = transcript_res.json()["id"]

    while True:
        result = requests.get(
            f"https://api.assemblyai.com/v2/transcript/{transcript_id}", headers=headers)
        result_json = result.json()
        status = result_json["status"]
        percent = result_json.get("percent_complete", 0)

        print(f"Status: {status} - {percent}% complete", end="\r")

        if status == "completed":
            utterances = result_json.get("utterances") or []

            keep = get_significant_speakers(utterances)

            transcript = []
            for u in utterances:
                if u["speaker"] in keep:
                    translated = u.get("translated_texts", {}).get("en", u["text"])
                    transcript.append(f"Speaker {u['speaker']}: {translated}")
            
            if os.path.exists(audio_file):
                os.remove(audio_file)
                print(f"Deleted : {audio_file}")
            
            return "\n".join(transcript)

        elif status == "error":
            print("Error:", result_json["error"])
            break

        time.sleep(5)
    

url = input("Enter audio URL: ")
audio_file = download_audio(url)
transcribe(audio_file)
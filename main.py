import requests
import time

# API key for AssemblyAI authentication
API_KEY = "0472e76115b5459e9c1ac306d5963673"
# Path to the audio file to be transcribed
AUDIO_FILE = "Assembly AI Speech to Text API (Part 1) - Extract Speakers and Transcription [Z7EBdmkB8kE].mp3"
# Headers for API requests, including authorization
headers = {"authorization": API_KEY}

# Step 1: Upload the audio file to AssemblyAI
print("Uploading audio...")
with open(AUDIO_FILE, "rb") as f:
    upload_res = requests.post("https://api.assemblyai.com/v2/upload", headers=headers, data=f)
audio_url = upload_res.json()["upload_url"]
print("Uploaded!")

# Step 2: Request transcription of the uploaded audio
print("Requesting transcript...")
transcript_res = requests.post(
    "https://api.assemblyai.com/v2/transcript",
    headers=headers,
    json={"audio_url": audio_url, "speech_models": ["universal-3-pro"]}
)
transcript_id = transcript_res.json()["id"]

# Step 3: Poll the API until transcription is complete
while True:
    result = requests.get(f"https://api.assemblyai.com/v2/transcript/{transcript_id}", headers=headers)
    status = result.json()["status"]
    print(f"Status: {status}")
    if status == "completed":
        print("\nTranscript:\n")
        print(result.json()["text"])
        break
    elif status == "error":
        print("Error:", result.json()["error"])
        break
    time.sleep(5)  # Wait 5 seconds before polling again
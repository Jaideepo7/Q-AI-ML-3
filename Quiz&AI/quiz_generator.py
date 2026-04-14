"""
quiz_generator.py
-----------------
Sends a video transcript to the Gemini API and returns 10 MCQs
as a structured list of dicts.

Dependencies:
    pip install google-genai dotenv

Usage:
    from quiz_generator import generate_quiz
    questions = generate_quiz(transcript_text, api_key="YOUR_KEY")
"""

import json
import re
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_MODEL = "gemini-2.5-flash"

SYSTEM_INSTRUCTION = """
You are an expert quiz creator. Read the given video transcript and produce
high-quality multiple-choice questions that test genuine understanding.
Always respond with valid JSON only — no markdown fences, no extra prose.
""".strip()

QUIZ_PROMPT_TEMPLATE = """
Below is a transcript from a video. Read it carefully, then generate exactly 10
multiple-choice questions that test a viewer's understanding of the key concepts,
facts, and ideas presented.

TRANSCRIPT:
\"\"\"
{transcript}
\"\"\"

REQUIREMENTS:
- Produce exactly 10 questions.
- Each question must have exactly 4 answer options labeled A, B, C, D.
- Only one option is correct.
- Questions should vary in difficulty (mix of recall, comprehension, and application).
- Base every question strictly on the transcript — no outside knowledge.

OUTPUT FORMAT (strict JSON, no markdown):
{{
  "quiz": [
    {{
      "id": 1,
      "question": "<question text>",
      "options": {{
        "A": "<option A>",
        "B": "<option B>",
        "C": "<option C>",
        "D": "<option D>"
      }},
      "correct_answer": "<A | B | C | D>",
      "explanation": "<one sentence explaining why the correct answer is right>"
    }}
  ]
}}
""".strip()


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def generate_quiz(transcript: str, api_key: str, model: str = GEMINI_MODEL) -> list[dict]:
    client = genai.Client(api_key=api_key)

    prompt = QUIZ_PROMPT_TEMPLATE.format(transcript=transcript.strip())

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.4,
            top_p=0.95,
            max_output_tokens=4096,
        ),
    )

    raw_text = response.text.strip()
    cleaned = _strip_markdown_fences(raw_text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Gemini returned invalid JSON.\n"
            f"Parse error: {e}\n"
            f"Raw response (first 500 chars):\n{raw_text[:500]}"
        ) from e

    questions = data.get("quiz", [])
    if not questions:
        raise ValueError("Gemini response parsed but contained no 'quiz' key.")

    return questions


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _strip_markdown_fences(text: str) -> str:
    match = re.match(r"^```(?:json)?\s*([\s\S]*?)\s*```$", text.strip())
    return match.group(1) if match else text


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    load_dotenv()
    my_transcript = """
    Urban gardening has quietly transformed rooftops, balconies, and abandoned lots into pockets of green life within dense cities. What once seemed like an impractical hobby has become a meaningful way for people to reconnect with nature, even in environments dominated by concrete and glass. Small containers filled with herbs, vegetables, and flowers now line apartment windows, proving that space is less important than intention.
One of the most appealing aspects of urban gardening is its accessibility. You don’t need acres of land or expensive equipment—just a few pots, some soil, and a bit of patience. Over time, gardeners begin to understand the subtle rhythms of plant growth, from how sunlight shifts throughout the day to how different plants respond to watering habits. This process builds not only skill but also a sense of mindfulness that is often missing in fast-paced city life.
Beyond personal satisfaction, urban gardening contributes to broader environmental benefits. Plants help improve air quality, reduce urban heat, and even support pollinators like bees and butterflies. In some neighborhoods, community gardens have become gathering spaces where people share knowledge, food, and culture. These gardens blur the line between individual effort and collective impact.
Despite its benefits, urban gardening does come with challenges. Limited space, inconsistent sunlight, and unpredictable weather can all affect plant growth. However, many gardeners see these constraints as part of the experience, encouraging creativity and experimentation. Vertical gardens, hydroponic systems, and recycled materials are just a few examples of how people adapt to urban conditions.
In the end, urban gardening is more than just growing plants—it’s about cultivating resilience, creativity, and connection in places where they might otherwise be overlooked.
    """

    questions = generate_quiz(transcript=my_transcript, api_key=os.getenv("GEMINI_API_KEY"))
    with open("quiz_output.json", "w") as f:
        json.dump({"quiz": questions}, f, indent=2)

    print("Quiz saved to quiz_output.json")
"""
quiz_generator.py
-----------------
Sends a video transcript to the Gemini API and returns 10 MCQs
as a structured list of dicts.

spaCy NER extracts key entities from the transcript (people, places,
concepts, dates, etc.) and passes them to Gemini so questions are
more targeted and grounded in the transcript's important terms.

Dependencies:
    pip install google-genai spacy python-dotenv
    python -m spacy download en_core_web_sm

Usage:
    from quiz_generator import generate_quiz
    questions = generate_quiz(transcript_text, api_key="YOUR_KEY")
"""

import json
import re
import os
import spacy
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

KEY ENTITIES EXTRACTED FROM THE TRANSCRIPT:
The following named entities were automatically identified in the transcript.
Use these to guide your questions — prioritise asking about these specific
people, places, concepts, dates, and terms where relevant.
{entities_block}

REQUIREMENTS:
- Produce exactly 10 questions.
- Each question must have exactly 4 answer options labeled A, B, C, D.
- Only one option is correct.
- Questions should vary in difficulty (mix of recall, comprehension, and application).
- Base every question strictly on the transcript — no outside knowledge.
- Where possible, incorporate the key entities listed above into your questions.

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
# spaCy NER
# ---------------------------------------------------------------------------

LABEL_MAP = {
    "PERSON":      "Person",
    "ORG":         "Organization",
    "GPE":         "Place",
    "LOC":         "Location",
    "DATE":        "Date",
    "TIME":        "Time",
    "EVENT":       "Event",
    "WORK_OF_ART": "Work of Art",
    "LAW":         "Law",
    "LANGUAGE":    "Language",
    "PRODUCT":     "Product",
    "NORP":        "Nationality/Group",
    "FAC":         "Facility",
    "MONEY":       "Money",
    "PERCENT":     "Percentage",
    "QUANTITY":    "Quantity",
    "ORDINAL":     "Ordinal",
    "CARDINAL":    "Number",
}

def extract_entities(transcript: str) -> dict:
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise OSError(
            "spaCy model not found. Run: python -m spacy download en_core_web_sm"
        )

    doc = nlp(transcript)
    entities = {}

    for ent in doc.ents:
        category = LABEL_MAP.get(ent.label_, ent.label_)
        entities.setdefault(category, set()).add(ent.text.strip())

    STOPWORDS = {
        "the day", "the end", "the line", "a bit", "a sense", "that space",
        "its benefits", "its accessibility", "this process", "these gardens",
        "these constraints", "individual effort", "collective impact",
        "personal satisfaction", "broader environmental benefits",
        "not only skill", "just a few pots", "just a few examples",
        "just growing plants", "land or expensive equipment",
        "the most appealing aspects", "the subtle rhythms",
        "how sunlight shifts", "how different plants",
    }

    noun_chunks = set()
    for chunk in doc.noun_chunks:
        text = chunk.text.strip()
        words = text.split()
        if 2 <= len(words) <= 3 and text.lower() not in STOPWORDS:
            noun_chunks.add(text)

    if noun_chunks:
        entities["Key Concept"] = sorted(noun_chunks)

    return {cat: sorted(ents) for cat, ents in entities.items()}


def format_entities_block(entities: dict) -> str:
    if not entities:
        return "  (No named entities detected)"
    lines = []
    for category, terms in entities.items():
        lines.append(f"  - {category}: {', '.join(terms)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def generate_quiz(transcript: str, api_key: str, model: str = GEMINI_MODEL) -> list:
    """
    Generate 10 MCQs from a video transcript using spaCy NER + Gemini API.

    Args:
        transcript: The full text transcript of the video.
        api_key:    Your Google Gemini API key.
        model:      Gemini model name (default: gemini-2.5-flash).

    Returns:
        A list of 10 question dicts, each with keys:
            id, question, options (dict A-D), correct_answer, explanation
    """
    entities = extract_entities(transcript)
    entities_block = format_entities_block(entities)
    print(f"[NER] Extracted entities:\n{entities_block}\n")

    client = genai.Client(api_key=api_key)
    prompt = QUIZ_PROMPT_TEMPLATE.format(
        transcript=transcript.strip(),
        entities_block=entities_block,
    )

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
            f"Gemini returned invalid JSON.\nParse error: {e}\nRaw response:\n{raw_text[:500]}"
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
    Urban gardening has quietly transformed rooftops, balconies, and abandoned lots into
    pockets of green life within dense cities. What once seemed like an impractical hobby
    has become a meaningful way for people to reconnect with nature, even in environments
    dominated by concrete and glass.
    """

    questions = generate_quiz(transcript=my_transcript, api_key=os.getenv("GEMINI_API_KEY"))

    with open("quiz_output.json", "w") as f:
        json.dump({"quiz": questions}, f, indent=2)

    print("Quiz saved to quiz_output.json")

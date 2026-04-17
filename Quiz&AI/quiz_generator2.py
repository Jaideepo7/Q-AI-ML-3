import json
import re
import os
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from google import genai
from google.genai import types
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_MODEL = "gemini-2.5-flash"

ABSTRACTIVE_SYSTEM_INSTRUCTION = """
You are an expert summarizer. You will receive key excerpts from a video transcript.
Synthesize them into a single, coherent, information-dense summary paragraph.
Preserve all important facts, names, concepts, and relationships.
Do not add any information not present in the excerpts.
Respond with plain text only — no bullet points, no headers, no markdown.
""".strip()

ABSTRACTIVE_PROMPT_TEMPLATE = """
Below are the most important excerpts from a video transcript, selected by relevance ranking.
Synthesize them into one coherent summary that captures all the key ideas, facts, and concepts.

EXCERPTS:
\"\"\"
{excerpts}
\"\"\"

Write a dense, informative summary paragraph:
""".strip()

QUIZ_SYSTEM_INSTRUCTION = """
You are an expert quiz creator. Read the given video transcript summary and produce
high-quality multiple-choice questions that test genuine understanding.
Always respond with valid JSON only — no markdown fences, no extra prose.
""".strip()

QUIZ_PROMPT_TEMPLATE = """
Below is a summary of a video transcript. Generate exactly 10 multiple-choice questions
that test a viewer's understanding of the key concepts, facts, and ideas presented.

TRANSCRIPT SUMMARY:
\"\"\"
{transcript}
\"\"\"

KEY ENTITIES EXTRACTED FROM THE TRANSCRIPT:
The following named entities were automatically identified. Use these to guide your
questions — prioritise asking about these specific people, places, concepts, and terms.
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
# Determine top N sentences based on video duration
# ---------------------------------------------------------------------------

def get_top_n(duration_minutes: float) -> int:
    """
    Return number of sentences to extract based on video length:
      < 30 min  → 5
      30-60 min → 8
      60-120 min → interpolated between 8 and 15
      > 120 min → 15
    """
    if duration_minutes < 30:
        return 5
    elif duration_minutes <= 60:
        return 8
    elif duration_minutes <= 120:
        ratio = (duration_minutes - 60) / 60
        return round(8 + ratio * 7)
    else:
        return 15


# ---------------------------------------------------------------------------
# Extractive summarization via TF-IDF + cosine similarity
# ---------------------------------------------------------------------------

def extractive_summarize(transcript: str, top_n: int) -> list:
    """
    Extract the top_n most representative sentences using
    TF-IDF vectorization + cosine similarity to the document vector.
    Returns sentences in their original order.
    """
    sentences = _split_sentences(transcript)

    if len(sentences) <= top_n:
        return sentences

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(sentences)
    except ValueError:
        return sentences[:top_n]

    # Document vector = mean of all sentence vectors
    doc_vector = np.asarray(tfidf_matrix.mean(axis=0))

    # Score each sentence by cosine similarity to the document
    scores = []
    for i in range(len(sentences)):
        score = cosine_similarity(tfidf_matrix[i], doc_vector)[0][0]
        scores.append((score, i))

    # Keep top_n sentences, restore original order
    top_indices = sorted(
        [idx for _, idx in sorted(scores, reverse=True)[:top_n]]
    )

    return [sentences[i] for i in top_indices]


def _split_sentences(text: str) -> list:
    """Split transcript into clean sentences."""
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 20]


# ---------------------------------------------------------------------------
# Abstractive summarization via Gemini
# ---------------------------------------------------------------------------

def abstractive_summarize(top_sentences: list, api_key: str) -> str:
    """
    Send top extracted sentences to Gemini and get back a
    coherent abstractive summary paragraph.
    """
    excerpts = "\n\n".join(f"- {s}" for s in top_sentences)
    prompt = ABSTRACTIVE_PROMPT_TEMPLATE.format(excerpts=excerpts)

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=ABSTRACTIVE_SYSTEM_INSTRUCTION,
            temperature=0.3,
            max_output_tokens=1024,
        ),
    )
    return response.text.strip()


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

CONCEPT_STOPWORDS = {
    "the day", "the end", "the line", "a bit", "a sense", "that space",
    "its benefits", "its accessibility", "this process", "these gardens",
    "these constraints", "individual effort", "collective impact",
    "personal satisfaction", "broader environmental benefits",
    "not only skill", "just a few pots", "just a few examples",
    "just growing plants", "land or expensive equipment",
    "the most appealing aspects", "the subtle rhythms",
    "how sunlight shifts", "how different plants",
}

def extract_entities(text: str) -> dict:
    """Run spaCy NER + noun chunk extraction on text."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise OSError(
            "spaCy model not found. Run: python -m spacy download en_core_web_sm"
        )

    doc = nlp(text)
    entities = {}

    # Named entities
    for ent in doc.ents:
        category = LABEL_MAP.get(ent.label_, ent.label_)
        entities.setdefault(category, set()).add(ent.text.strip())

    # Key noun chunks (2-3 words, filtered)
    noun_chunks = set()
    for chunk in doc.noun_chunks:
        t = chunk.text.strip()
        words = t.split()
        if 2 <= len(words) <= 3 and t.lower() not in CONCEPT_STOPWORDS:
            noun_chunks.add(t)
    if noun_chunks:
        entities["Key Concept"] = sorted(noun_chunks)

    return {cat: sorted(ents) for cat, ents in entities.items()}


def format_entities_block(entities: dict) -> str:
    if not entities:
        return "  (No named entities detected)"
    return "\n".join(
        f"  - {cat}: {', '.join(terms)}" for cat, terms in entities.items()
    )


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def generate_quiz(
    transcript: str,
    api_key: str,
    duration_minutes: float = 20.0,
    model: str = GEMINI_MODEL,
) -> list:
    """
    Full pipeline: summarize transcript → extract entities → generate quiz.

    Args:
        transcript:        Full transcript text from the video.
        api_key:           Your Google Gemini API key.
        duration_minutes:  Length of the video in minutes (default: 20).
        model:             Gemini model name.

    Returns:
        A list of 10 question dicts with keys:
            id, question, options (dict A-D), correct_answer, explanation
    """
    # --- Step 1: Extractive summarization ---
    top_n = get_top_n(duration_minutes)
    print(f"[SUMMARIZER] Duration: {duration_minutes} min → extracting top {top_n} sentences")
    top_sentences = extractive_summarize(transcript, top_n)
    print(f"[SUMMARIZER] Extracted {len(top_sentences)} key sentences")

    # --- Step 2: Abstractive summarization ---
    print("[SUMMARIZER] Sending to Gemini for abstractive summarization...")
    summary = abstractive_summarize(top_sentences, api_key)
    print(f"[SUMMARIZER] Summary:\n{summary}\n")

    # --- Step 3: NER on the summary ---
    entities = extract_entities(summary)
    entities_block = format_entities_block(entities)
    print(f"[NER] Extracted entities:\n{entities_block}\n")

    # --- Step 4: Generate quiz from summary + entities ---
    client = genai.Client(api_key=api_key)
    prompt = QUIZ_PROMPT_TEMPLATE.format(
        transcript=summary,
        entities_block=entities_block,
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=QUIZ_SYSTEM_INSTRUCTION,
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
    Urban gardening has quietly transformed rooftops, balconies, and abandoned lots into 
    pockets of green life within dense cities. What once seemed like an impractical hobby 
    has become a meaningful way for people to reconnect with nature, even in environments 
    dominated by concrete and glass. Small containers filled with herbs, vegetables, and 
    flowers now line apartment windows, proving that space is less important than intention.
    One of the most appealing aspects of urban gardening is its accessibility. You don't 
    need acres of land or expensive equipment — just a few pots, some soil, and a bit of 
    patience. Over time, gardeners begin to understand the subtle rhythms of plant growth, 
    from how sunlight shifts throughout the day to how different plants respond to watering 
    habits. This process builds not only skill but also a sense of mindfulness that is often 
    missing in fast-paced city life. Beyond personal satisfaction, urban gardening contributes 
    to broader environmental benefits. Plants help improve air quality, reduce urban heat, and 
    even support pollinators like bees and butterflies. In some neighborhoods, community gardens 
    have become gathering spaces where people share knowledge, food, and culture. These gardens 
    blur the line between individual effort and collective impact. Despite its benefits, urban 
    gardening does come with challenges. Limited space, inconsistent sunlight, and unpredictable 
    weather can all affect plant growth. However, many gardeners see these constraints as part 
    of the experience, encouraging creativity and experimentation. Vertical gardens, hydroponic 
    systems, and recycled materials are just a few examples of how people adapt to urban 
    conditions. In the end, urban gardening is more than just growing plants — it's about 
    cultivating resilience, creativity, and connection in places where they might otherwise 
    be overlooked.
    """

    questions = generate_quiz(
        transcript=my_transcript,
        api_key=os.getenv("GEMINI_API_KEY"),
        duration_minutes=45,
    )

    with open("quiz_output.json", "w") as f:
        json.dump({"quiz": questions}, f, indent=2)

    print("Quiz saved to quiz_output.json")
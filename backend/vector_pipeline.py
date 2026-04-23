from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List
import os
import json
import base64
import numpy as np
from dotenv import load_dotenv
from supabase import Client, create_client


load_dotenv()


# ---------------------------------------------------------------------------
# Supabase connection config
# ---------------------------------------------------------------------------

SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY") or os.getenv("NEXT_PUBLIC_SUPABASE_PUBLISHABLE_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("NEXT_PUBLIC_SUPABASE_URL")


def _derive_supabase_url_from_key(api_key: str) -> str:
    try:
        payload_b64 = api_key.split(".")[1]
        payload_b64 += "=" * (-len(payload_b64) % 4)
        payload_json = base64.urlsafe_b64decode(payload_b64.encode("utf-8")).decode("utf-8")
        project_ref = json.loads(payload_json).get("ref")
        if not project_ref:
            raise ValueError("JWT payload does not contain project ref.")
        return f"https://{project_ref}.supabase.co"
    except Exception as exc:
        raise ValueError(
            "Could not derive SUPABASE_URL from SUPABASE_ANON_KEY. Set SUPABASE_URL explicitly."
        ) from exc


def get_supabase_client() -> Client:
    if not SUPABASE_ANON_KEY:
        raise ValueError(
            "Missing Supabase config. Set SUPABASE_ANON_KEY in your .env file."
        )

    resolved_url = SUPABASE_URL or _derive_supabase_url_from_key(SUPABASE_ANON_KEY)
    return create_client(resolved_url, SUPABASE_ANON_KEY)


def _embedding_to_vector_literal(embedding: List[float]) -> str:
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


def _parse_vector(value) -> np.ndarray:
    if isinstance(value, list):
        return np.array(value, dtype=float)
    if isinstance(value, str):
        return np.fromstring(value.strip("[]"), sep=",", dtype=float)
    return np.array([], dtype=float)


# ---------------------------------------------------------------------------
# Load HuggingFace sentence-transformer model once at module level
# ---------------------------------------------------------------------------

# all-MiniLM-L6-v2: lightweight, fast, 384-dim embeddings — good default
EMBEDDING_MODEL = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# ---------------------------------------------------------------------------
# Part 1 — Dynamic Chunking based on video/transcript length
# ---------------------------------------------------------------------------

def get_chunk_params(transcript: str, video_duration_seconds: int) -> dict:
    minutes = video_duration_seconds / 60

    if minutes < 5:
        return {"chunk_size": 300,  "chunk_overlap": 30}
    elif minutes < 20:
        return {"chunk_size": 500,  "chunk_overlap": 75}
    elif minutes < 60:
        return {"chunk_size": 800,  "chunk_overlap": 120}
    else:
        return {"chunk_size": 1200, "chunk_overlap": 200}


def chunk_transcript(transcript: str, video_duration_seconds: int) -> List[str]:
    if not transcript or not transcript.strip():
        return []

    params = get_chunk_params(transcript, video_duration_seconds)
    print(f"[Chunking] Video: {video_duration_seconds//60}min → "
          f"chunk_size={params['chunk_size']}, overlap={params['chunk_overlap']}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=params["chunk_size"],
        chunk_overlap=params["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(transcript)
    print(f"[Chunking] Produced {len(chunks)} chunks")
    return chunks


# ---------------------------------------------------------------------------
# Part 2 — Embed chunks using HuggingFace sentence-transformers
# ---------------------------------------------------------------------------

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    if not chunks:
        return []

    print(f"[Embedding] Embedding {len(chunks)} chunks...")
    embeddings = EMBEDDING_MODEL.embed_documents(chunks)
    print(f"[Embedding] Done. Each vector has {len(embeddings[0])} dimensions.")
    return embeddings


# ---------------------------------------------------------------------------
# Part 3 — Store chunks + embeddings in Supabase (Postgres + pgvector)
# ---------------------------------------------------------------------------

def store_in_supabase(
    chunks: List[str],
    embeddings: List[List[float]],
    video_id: str,
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length.")

    client = get_supabase_client()
    print(f"[Storage] Inserting {len(chunks)} rows into Supabase (chunks table)...")

    payload = [
        {
            "vid_id": video_id,
            "content": chunk,
            "chunk_index": i,
            "embedding": _embedding_to_vector_literal(embedding),
        }
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

    client.table("chunks").insert(payload).execute()
    print("[Storage] Done. All chunks stored in Supabase.")


# ---------------------------------------------------------------------------
# Part 4 — Semantic search (used later by quiz generator to retrieve context)
# ---------------------------------------------------------------------------

def search_similar_chunks(query: str, video_id: str, top_k: int = 5) -> List[str]:
    query_embedding = np.array(EMBEDDING_MODEL.embed_query(query), dtype=float)
    query_norm = np.linalg.norm(query_embedding)
    if query_norm == 0:
        return []

    client = get_supabase_client()
    response = (
        client.table("chunks")
        .select("content, embedding")
        .eq("vid_id", video_id)
        .execute()
    )
    rows = response.data or []

    scored = []
    for row in rows:
        content = row.get("content")
        vector = _parse_vector(row.get("embedding"))
        if vector.size == 0 or vector.shape[0] != query_embedding.shape[0]:
            continue
        denom = np.linalg.norm(vector) * query_norm
        if denom == 0:
            continue
        score = float(np.dot(query_embedding, vector) / denom)
        scored.append((score, content))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [content for _, content in scored[:top_k]]


# ---------------------------------------------------------------------------
# Main pipeline — call this from app.py with a transcript + video duration
# ---------------------------------------------------------------------------

def run_vector_pipeline(
    transcript: str,
    video_id: str,
    video_duration_seconds: int = 600,
) -> None:
    chunks     = chunk_transcript(transcript, video_duration_seconds)
    embeddings = embed_chunks(chunks)
    store_in_supabase(chunks, embeddings, video_id)


# ---------------------------------------------------------------------------
# Smoke test  (run: python vector_pipeline.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sample_transcript = """
    Machine learning is a subset of artificial intelligence. It enables systems
    to learn from data without being explicitly programmed. Supervised learning
    uses labeled data to train models. Unsupervised learning finds hidden
    patterns in unlabeled data. Reinforcement learning trains agents through
    rewards and penalties. Neural networks are inspired by the human brain.
    Deep learning uses many layers of neurons to learn complex representations.
    """ * 10  # repeat to simulate a longer transcript

    # Simulate a 12-minute video
    video_duration = 12 * 60

    print("=== CHUNKING ===")
    chunks = chunk_transcript(sample_transcript, video_duration)
    print(f"First chunk: {repr(chunks[0])}\n")

    print("=== EMBEDDING ===")
    embeddings = embed_chunks(chunks[:3])  # just embed first 3 for speed
    print(f"Embedding shape: {len(embeddings)} x {len(embeddings[0])}\n")

    print("=== STORING ===")
    store_in_supabase(chunks, embed_chunks(chunks))

    print("=== SEARCH ===")
    results = search_similar_chunks("supervised learning", top_k=3)
    for i, r in enumerate(results):
        print(f"[Result {i}] {r}\n")

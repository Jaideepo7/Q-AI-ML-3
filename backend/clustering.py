"""
clustering_pipeline.py
=======================
Performs DBSCAN clustering on stored embeddings, labels each cluster
using Gemini, stores labels back in Supabase, and provides a RAG
retrieval function using Supabase's match_chunks function.

Pipeline:
  1. Fetch all stored embeddings from Supabase
  2. Run DBSCAN with 3 different eps values, pick the best one
  3. Label each cluster with 2-3 words using Gemini
  4. Store cluster_id + cluster_label back in Supabase
  5. RAG retrieval via match_chunks Supabase function
  6. Entry point to connect audio/video → transcription → vector pipeline

Dependencies:
  pip install scikit-learn google-generativeai supabase python-dotenv numpy
"""

import os
import time
from urllib import response
import numpy as np
from dotenv import load_dotenv
from supabase import create_client, Client
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize
from typing import List, Dict, Tuple
from google import genai
# Import our existing pipeline to connect audio/video
from vector_pipeline import run_vector_pipeline


load_dotenv()

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

def get_supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_ANON_KEY")
    if not url or not key:
        raise ValueError("Missing SUPABASE_URL or SUPABASE_ANON_KEY in .env")
    return create_client(url, key)


def get_gemini_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY in .env")
    client = genai.Client(api_key=api_key)
    return client


# ---------------------------------------------------------------------------
# Part 1 — Fetch stored embeddings from Supabase
# ---------------------------------------------------------------------------

def fetch_embeddings(video_id: str) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Fetch all chunks for a specific video from the chunks table.

    Returns:
        ids       : list of row UUIDs
        contents  : list of chunk text strings
        embeddings: numpy array of shape (n_chunks, 384)
    """
    client = get_supabase_client()
    response = (
        client.table("chunks")
        .select("id, content, embedding")
        .eq("vid_id", video_id)
        .execute()
    )
    rows = response.data or []

    if not rows:
        raise ValueError(f"No chunks found for video_id={video_id}. Run vector_pipeline first.")

    ids, contents, embeddings = [], [], []

    for row in rows:
        ids.append(row["id"])
        contents.append(row["content"])

        vec = row["embedding"]
        if isinstance(vec, str):
            vec = np.fromstring(vec.strip("[]"), sep=",", dtype=float)
        else:
            vec = np.array(vec, dtype=float)
        embeddings.append(vec)

    print(f"[Fetch] Loaded {len(ids)} chunks for video_id={video_id}.")
    return ids, contents, np.array(embeddings)


# ---------------------------------------------------------------------------
# Part 2 — DBSCAN clustering with 3 eps values
# ---------------------------------------------------------------------------

def run_dbscan_experiments(embeddings: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Test DBSCAN with 3 different eps values and pick the best one.

    'Best' = most meaningful clusters (fewest noise points, most clusters,
    not everything collapsed into one giant cluster).

    eps controls the neighborhood radius:
      - Too small → everything is noise (-1)
      - Too large  → everything merges into one cluster

    Args:
        embeddings: numpy array of shape (n_chunks, 384)

    Returns:
        best_labels : cluster label array (-1 = noise)
        best_eps    : the eps value that was chosen
    """
    # Normalize embeddings to unit length for cosine-like distances
    normed = normalize(embeddings)

    eps_candidates = [0.2, 0.4, 0.6]
    results = {}

    print("\n[DBSCAN] Testing 3 eps values...")
    print(f"{'eps':>6} | {'clusters':>8} | {'noise':>6} | {'score':>8}")
    print("-" * 40)

    for eps in eps_candidates:
        db = DBSCAN(eps=eps, min_samples=2, metric="cosine")
        labels = db.fit_predict(normed)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise    = list(labels).count(-1)
        total      = len(labels)

        # Score: reward more clusters, penalize noise heavily
        # We want clusters but not at the cost of everything being noise
        score = n_clusters - (n_noise / total) * 2

        results[eps] = {"labels": labels, "n_clusters": n_clusters,
                        "n_noise": n_noise, "score": score}

        print(f"{eps:>6} | {n_clusters:>8} | {n_noise:>6} | {score:>8.3f}")

    # Pick the eps with the highest score
    best_eps = max(results, key=lambda e: results[e]["score"])
    best     = results[best_eps]
    print(f"\n[DBSCAN] Best eps={best_eps} → "
          f"{best['n_clusters']} clusters, {best['n_noise']} noise points\n")

    return best["labels"], best_eps


# ---------------------------------------------------------------------------
# Part 3 — Label each cluster with 2-3 words using Gemini
# ---------------------------------------------------------------------------

def label_clusters_with_gemini(
    contents: List[str],
    labels:   np.ndarray,
) -> Dict[int, str]:
    """
    For each unique cluster, collect its chunk texts and ask Gemini
    to summarize the cluster in 2-3 words.

    Noise points (label = -1) are labeled "Uncategorized".

    Args:
        contents : list of chunk text strings
        labels   : DBSCAN cluster labels per chunk

    Returns:
        Dict mapping cluster_id → 2-3 word label string
    """
    model = get_gemini_model()
    cluster_ids = sorted(set(labels))
    cluster_labels = {}

    for cluster_id in cluster_ids:
        if cluster_id == -1:
            cluster_labels[-1] = "Uncategorized"
            continue

        # Gather all chunks belonging to this cluster
        cluster_chunks = [
            contents[i]
            for i, label in enumerate(labels)
            if label == cluster_id
        ]

        # Combine into one context block (truncated for API efficiency)
        combined = " ".join(cluster_chunks)[:1500]

        prompt = f"""You are labeling a topic cluster from a video transcript.
        
Here are some transcript excerpts from the same topic cluster:
\"\"\"{combined}\"\"\"

Respond with ONLY a 2-3 word label that best describes the main topic of these excerpts.
Examples of good labels: "Neural Networks", "Data Preprocessing", "Model Evaluation"
Do NOT include punctuation or explanation — just the label."""

        try:
            response = model.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            label_text = response.text.strip()
            cluster_labels[cluster_id] = label_text
            print(f"[Gemini] Cluster {cluster_id} → '{label_text}'")
            time.sleep(0.5)  # avoid hitting rate limits
        except Exception as e:
            print(f"[Gemini] Error labeling cluster {cluster_id}: {e}")
            cluster_labels[cluster_id] = f"Cluster {cluster_id}"

    return cluster_labels


# ---------------------------------------------------------------------------
# Part 4 — Store cluster_id and cluster_label back in Supabase
# ---------------------------------------------------------------------------

def store_cluster_labels(
    ids:            List[str],
    labels:         np.ndarray,
    cluster_labels: Dict[int, str],
) -> None:
    """
    Update each row in chunks with its topic_label derived from DBSCAN cluster.

    Args:
        ids            : list of Supabase row UUIDs
        labels         : DBSCAN labels per chunk
        cluster_labels : dict mapping cluster_id → label string
    """
    client = get_supabase_client()
    print(f"\n[Storage] Updating {len(ids)} rows with topic labels...")

    for row_id, cluster_id in zip(ids, labels):
        label_text = cluster_labels.get(int(cluster_id), "Uncategorized")
        client.table("chunks").update({
            "topic_label": label_text,
        }).eq("id", row_id).execute()

    print("[Storage] Done. All topic labels stored in Supabase.")


# ---------------------------------------------------------------------------
# Part 5 — RAG retrieval using Supabase match_chunks function
# ---------------------------------------------------------------------------

def rag_retrieve(query: str, video_id: str, top_k: int = 5) -> List[Dict]:
    """
    Embed a query and retrieve the most semantically similar chunks for a
    specific video using Python cosine similarity (no RPC required).

    Args:
        query    : Topic or question string, e.g. "What is supervised learning?"
        video_id : UUID of the video to scope the search to.
        top_k    : Number of chunks to retrieve. Default 5.

    Returns:
        List of dicts: [{"id", "content", "topic_label", "similarity"}, ...]
    """
    from vector_pipeline import EMBEDDING_MODEL, _parse_vector

    print(f"\n[RAG] Retrieving top {top_k} chunks for query: '{query}'")
    query_vec = np.array(EMBEDDING_MODEL.embed_query(query), dtype=float)
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return []

    client = get_supabase_client()
    response = (
        client.table("chunks")
        .select("id, content, topic_label, embedding")
        .eq("vid_id", video_id)
        .execute()
    )
    rows = response.data or []

    scored = []
    for row in rows:
        vec = _parse_vector(row.get("embedding"))
        if vec.size == 0 or vec.shape[0] != query_vec.shape[0]:
            continue
        denom = np.linalg.norm(vec) * query_norm
        if denom == 0:
            continue
        similarity = float(np.dot(query_vec, vec) / denom)
        scored.append({
            "id":          row["id"],
            "content":     row["content"],
            "topic_label": row.get("topic_label", "Uncategorized"),
            "similarity":  similarity,
        })

    scored.sort(key=lambda r: r["similarity"], reverse=True)
    results = scored[:top_k]

    print(f"[RAG] Retrieved {len(results)} chunks.")
    for i, r in enumerate(results):
        print(f"  [{i}] (sim={r['similarity']:.3f}) [{r['topic_label']}] "
              f"{r['content'][:80]}...")

    return results


def run_clustering_pipeline(video_id: str) -> None:
    """
    Fetch embeddings for a video, cluster with DBSCAN, label with Gemini,
    and store topic_labels back in chunks table.
    Call this after run_vector_pipeline().
    """
    ids, contents, embeddings = fetch_embeddings(video_id)
    labels, best_eps          = run_dbscan_experiments(embeddings)
    cluster_labels            = label_clusters_with_gemini(contents, labels)
    store_cluster_labels(ids, labels, cluster_labels)


# ---------------------------------------------------------------------------
# Smoke test  (run: python clustering_pipeline.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== CLUSTERING PIPELINE ===\n")

    # Step 1-4: cluster existing embeddings already in Supabase
    run_clustering_pipeline()

    # Step 5: test RAG retrieval
    print("\n=== RAG RETRIEVAL TEST ===")
    results = rag_retrieve("supervised learning", top_k=3)

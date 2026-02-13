import json
import math
import subprocess
import shlex
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()

# Choose a sentence-transformers model available locally or will auto-download.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # fast and effective; change if you prefer
OLLAMA_MODEL = 'llama-3.1-8b-instant'

print("Loading embedding model:", EMBEDDING_MODEL_NAME)
embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
print("Embedding model loaded.")

def get_embedding(text: str) -> np.ndarray:
    if text is None:
        text = ""
    # SentenceTransformer returns numpy arrays
    emb = embed_model.encode([text], show_progress_bar=False, normalize_embeddings=True)
    return np.asarray(emb[0])

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None:
        return 0.0
    # embeddings normalized in get_embedding; but compute robustly
    return float(np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12), -1.0, 1.0))

def sim_to_01(sim: float) -> float:
    # map cosine similarity (-1..1) to 0..1
    return max(0.0, min(1.0, (sim + 1.0) / 2.0))


def step_text(step: Dict[str, Any]) -> str:
    """
    Extract a canonical text representation for a ToT step.
    Assumes step dict has 'input' and 'output' fields (based on your inference.py).
    """
    parts = []
    if step.get("input"):
        parts.append(str(step["input"]).strip())
    if step.get("output"):
        parts.append(str(step["output"]).strip())
    return " ".join(parts).strip()

def compute_goal_alignment(prompt: str, final_subtasks: List[Dict[str, Any]], use_last_only: bool = True) -> float:
    """
    Compute semantic similarity between the prompt (goal) and the ToT final answer.
    If use_last_only: only consider the last step output, else concatenate all outputs.
    Returns normalized score in [0,1].
    """
    if not final_subtasks:
        return 0.0
    if use_last_only:
        final_text = step_text(final_subtasks[-1])
    else:
        final_text = " ".join([step_text(s) for s in final_subtasks])
    if not final_text:
        return 0.0
    sim = cosine(get_embedding(final_text), get_embedding(prompt))
    return sim_to_01(sim)

def compute_stepwise_coherence(final_subtasks: List[Dict[str, Any]]) -> float:
    """
    Average adjacent-step similarity (normalized to 0..1).
    If only one step, returns 1.0 (perfect local coherence).
    """
    texts = [step_text(s) for s in final_subtasks]
    embs = [get_embedding(t) for t in texts if t]
    if len(embs) <= 1:
        return 1.0
    sims = []
    for i in range(len(embs) - 1):
        sims.append(sim_to_01(cosine(embs[i], embs[i+1])))
    return float(sum(sims) / len(sims))

def call_verifier_coverage(prompt: str, final_subtasks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Uses ollama LLM to ask for missing concepts / expected concept count.
    The LLM is instructed to output strict JSON:
    {
      "missing": ["concept1", "concept2", ...],
      "expected_concepts": 4,
      "notes": "short text"
    }
    """
    # Build a compact representation of steps
    steps_text = "\n".join([f"Step {i+1}: {step_text(s)}" for i, s in enumerate(final_subtasks)])
    verifier_prompt = f"""
You are a verifier assistant. Given the original prompt and the chain-of-thought steps, output a strict JSON object with fields:
 - missing: a list of missing sub-concepts or checks that should be present for a complete solution (may be empty list).
 - expected_concepts: integer estimate for the number of major sub-concepts that should be covered.
 - notes: one-line diagnostic advice.

Respond ONLY with JSON (no extra commentary).

Original prompt:
\"\"\"{prompt}\"\"\"

Chain-of-thought steps:
\"\"\"{steps_text}\"\"\"

Produce the JSON now.
"""
    llm = ChatGroq(model=OLLAMA_MODEL, api_key=os.environ.get("GROQ_API_KEY"))
    response = llm.invoke(verifier_prompt)
    raw = response.content
    # Try to extract JSON substring (some LLMs may wrap). We'll attempt to find the first '{' and last '}'.
    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        json_text = raw[start:end]
        parsed = json.loads(json_text)
        # sanitise fields
        parsed.setdefault("missing", [])
        parsed.setdefault("expected_concepts", max(1, len(final_subtasks)))
        parsed.setdefault("notes", "")
        return parsed
    except Exception as e:
        # Fallback heuristic: no LLM JSON available, return heuristics
        print("Warning: verifier did not return clean JSON. Falling back to heuristic coverage. Error:", e)
        # Simple heuristic: expected = len(steps), missing = []
        return {"missing": [], "expected_concepts": max(1, len(final_subtasks)), "notes": "fallback heuristic used"}


def compute_granularity(final_subtasks: List[Dict[str, Any]], ideal_len: int = 25, sigma: float = 12.0) -> float:
    """
    Score each step by how close its token count (word count) is to ideal_len using a Gaussian mapping.
    Returns average score in [0,1].
    """
    lengths = []
    for s in final_subtasks:
        l = len(step_text(s).split())
        lengths.append(l)
    if not lengths:
        return 1.0
    scores = [math.exp(-((l - ideal_len) ** 2) / (2 * (sigma ** 2))) for l in lengths]
    # normalize scores to [0,1] (they already are because exp -> (0,1])
    return float(sum(scores) / len(scores))

def compute_redundancy(final_subtasks: List[Dict[str, Any]], cluster_threshold: float = 0.85) -> float:
    """
    Greedy clustering by cosine similarity threshold. Returns "uniqueness" fraction (higher = less redundant).
    """
    texts = [step_text(s) for s in final_subtasks]
    if not texts:
        return 1.0
    embs = np.vstack([get_embedding(t) for t in texts])
    # Use AgglomerativeClustering with distance threshold equivalent to (1 - cluster_threshold)
    try:
        # sklearn AgglomerativeClustering with distance_threshold requires n_clusters=None
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0 - cluster_threshold, linkage="average").fit(embs)
        labels = clustering.labels_
        num_clusters = len(set(labels))
    except Exception:
        # fallback: greedy grouping
        centroids = []
        labels = []
        for e in embs:
            placed = False
            for i, c in enumerate(centroids):
                if cosine(e, c) >= cluster_threshold:
                    # update centroid (mean)
                    centroids[i] = (centroids[i] * labels.count(i) + e) / (labels.count(i) + 1)
                    labels.append(i)
                    placed = True
                    break
            if not placed:
                centroids.append(e)
                labels.append(len(centroids) - 1)
        num_clusters = len(centroids)
    unique_frac = num_clusters / max(1, len(texts))
    # uniqueness score: closer to 1 is better (less redundancy)
    return float(unique_frac)


def combined_reasoning_score(out: Dict[str, Any], prompt: str, weights: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    Compute component scores and combined reasoning score.
    out: dictionary returned by run_inference_pipeline (expects out['final_subtasks'] list)
    prompt: original prompt text
    """
    final_subtasks = out.get("final_subtasks", [])
    # components
    alignment = compute_goal_alignment(prompt, final_subtasks, use_last_only=True)
    coherence = compute_stepwise_coherence(final_subtasks)
    granularity = compute_granularity(final_subtasks)
    redundancy = compute_redundancy(final_subtasks)
    # coverage uses LLM verifier
    verifier_info = call_verifier_coverage(prompt, final_subtasks)
    missing = verifier_info.get("missing", [])
    expected = max(1, int(verifier_info.get("expected_concepts", max(1, len(final_subtasks)))))
    coverage = max(0.0, 1.0 - (len(missing) / expected))
    # default weights (sum to 1)
    if weights is None:
        weights = {
            "alignment": 0.30,
            "coherence": 0.25,
            "coverage": 0.25,
            "granularity": 0.10,
            "redundancy": 0.10
        }
    # ensure normalization
    s = sum(weights.values())
    if s == 0:
        raise ValueError("weights must sum to > 0")
    for k in weights:
        weights[k] = weights[k] / s
    reasoning_score = (
        weights["alignment"] * alignment
        + weights["coherence"] * coherence
        + weights["coverage"] * coverage
        + weights["granularity"] * granularity
        + weights["redundancy"] * redundancy
    )
    # diagnostics
    diagnostics = {
        "stepwise_similarities": None,  # compute adjacent sims for debugging
        "missing_concepts": missing,
        "expected_concepts": expected,
        "average_step_length": None,
        "unique_clusters": None,
        "num_steps": len(final_subtasks),
        "verifier_notes": verifier_info.get("notes", "")
    }
    # populate additional diagnostics
    # adjacent sims
    texts = [step_text(s) for s in final_subtasks]
    embs = [get_embedding(t) for t in texts if t]
    adj_sims = [sim_to_01(cosine(embs[i], embs[i+1])) for i in range(len(embs)-1)] if len(embs) > 1 else []
    diagnostics["stepwise_similarities"] = adj_sims
    diagnostics["average_step_length"] = float(sum(len(t.split()) for t in texts) / max(1, len(texts)))
    # clusters uniqueness
    diagnostics["unique_clusters"] = float(compute_redundancy(final_subtasks))
    # verdict
    if reasoning_score >= 0.75:
        verdict = "Reasoning is coherent and well-aligned with goal."
    elif reasoning_score >= 0.5:
        verdict = "Reasoning is partially coherent; some gaps or inconsistencies."
    else:
        verdict = "Reasoning shows substantial drift or incompleteness."
    return {
        "reasoning_score": float(reasoning_score),
        "components": {
            "goal_alignment": float(alignment),
            "stepwise_coherence": float(coherence),
            "coverage": float(coverage),
            "granularity": float(granularity),
            "redundancy": float(redundancy)
        },
        "diagnostics": diagnostics,
        "verdict": verdict
    }
"""
faiss_index.py

FAISS index for fast offer retrieval.
Builds an index over offer embeddings and provides ANN search.
"""

import os
import json
import numpy as np
import faiss

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INDEX_PATH = os.path.join(BACKEND_DIR, "faiss_offers.index")
META_PATH = os.path.join(BACKEND_DIR, "faiss_offers_meta.json")

_index: faiss.Index | None = None
_offer_ids: list[str] = []


def build_index(offer_ids: list[str], embeddings: np.ndarray):
    """Build FAISS index from offer embeddings.
    
    Args:
        offer_ids: list of offer IDs (order must match embeddings rows)
        embeddings: numpy array of shape (n_offers, dim), L2-normalized
    """
    global _index, _offer_ids

    dim = embeddings.shape[1]
    n_offers = embeddings.shape[0]

    _index = faiss.IndexFlatIP(dim)
    _index.add(embeddings.astype(np.float32))

    _offer_ids = list(offer_ids)

    faiss.write_index(_index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump({"offer_ids": _offer_ids, "dim": dim, "n_offers": n_offers}, f)

    print(f"FAISS index built: {n_offers} offers, {dim}d, saved to {INDEX_PATH}")


def load_index():
    """Load FAISS index from disk."""
    global _index, _offer_ids

    if not os.path.exists(INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found at {INDEX_PATH}. Run build_index first.")

    _index = faiss.read_index(INDEX_PATH)
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    _offer_ids = meta["offer_ids"]

    print(f"FAISS index loaded: {len(_offer_ids)} offers")


def search(query_embedding: np.ndarray, top_k: int = 50) -> list[tuple[str, float]]:
    """Search for similar offers.
    
    Args:
        query_embedding: numpy array of shape (dim,), L2-normalized
        top_k: number of results to return
    
    Returns:
        list of (offer_id, score) tuples, sorted by score descending
    """
    global _index, _offer_ids

    if _index is None:
        load_index()

    query = query_embedding.reshape(1, -1).astype(np.float32)
    scores, indices = _index.search(query, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(_offer_ids):
            continue
        results.append((_offer_ids[idx], float(score)))

    return results

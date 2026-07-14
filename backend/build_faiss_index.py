"""
build_faiss_index.py

Build FAISS index from ChromaDB offer embeddings.
Usage: cd backend && python build_faiss_index.py
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.chromadb_client import get_offers_collection
from app.services.faiss_index import build_index


def main():
    offers = get_offers_collection()
    count = offers.count()
    print(f"ChromaDB offers: {count} vectors")

    if count == 0:
        print("ERROR: No offers in ChromaDB. Run seed_data.py first.")
        return

    all_data = offers.get(include=["embeddings", "metadatas"])
    ids = [m["id"] for m in all_data["metadatas"]]
    embeddings = np.array(all_data["embeddings"], dtype=np.float32)
    print(f"Loaded {len(ids)} offers, dim={embeddings.shape[1]}")

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_normed = embeddings / norms

    build_index(ids, embeddings_normed)
    print("Done!")


if __name__ == "__main__":
    main()

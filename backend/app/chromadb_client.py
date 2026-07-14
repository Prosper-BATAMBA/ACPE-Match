from __future__ import annotations

import chromadb

from .config import CHROMADB_PERSIST_DIR

_client: chromadb.ClientAPI | None = None

CANDIDATES_COLLECTION = "candidate_embeddings"
OFFERS_COLLECTION = "offer_embeddings"


def get_chroma_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMADB_PERSIST_DIR)
    return _client


def get_candidates_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=CANDIDATES_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def get_offers_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=OFFERS_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

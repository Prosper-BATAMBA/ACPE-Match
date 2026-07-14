from __future__ import annotations

from typing import List

from sentence_transformers import SentenceTransformer

from ..config import SENTENCE_TRANSFORMER_MODEL

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    return _model


def encode(text: str) -> List[float]:
    model = get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def encode_batch(texts: List[str]) -> List[List[float]]:
    model = get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()

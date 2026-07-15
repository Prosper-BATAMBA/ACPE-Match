from __future__ import annotations

import os
from typing import List

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from ..config import SENTENCE_TRANSFORMER_MODEL

_model = None
_device: str | None = None


def _get_device() -> str:
    global _device
    if _device is None:
        import torch
        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL, device=_get_device())
    return _model


def encode(text: str) -> List[float]:
    model = get_model()
    embedding = model.encode(text, normalize_embeddings=True, device=_get_device())
    return embedding.tolist()


def encode_batch(texts: List[str]) -> List[List[float]]:
    model = get_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=64,
        device=_get_device(),
        convert_to_numpy=True,
    )
    return embeddings.tolist()

from __future__ import annotations

import os
from typing import List

os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

from ..config import SENTENCE_TRANSFORMER_MODEL

_model = None
_device: str | None = None

# BGE-M3 truncation: ~3000 chars (~500-600 tokens) avoids silent degradation
_MAX_CHARS = 3000


def _truncate_text(text: str) -> str:
    """Truncate text to avoid BGE-M3 performance degradation on long inputs."""
    if len(text) <= _MAX_CHARS:
        return text
    return text[:_MAX_CHARS]


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
    embedding = model.encode(_truncate_text(text), normalize_embeddings=True, device=_get_device())
    return embedding.tolist()


def encode_batch(texts: List[str]) -> List[List[float]]:
    model = get_model()
    truncated = [_truncate_text(t) for t in texts]
    embeddings = model.encode(
        truncated,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=64,
        device=_get_device(),
        convert_to_numpy=True,
    )
    return embeddings.tolist()

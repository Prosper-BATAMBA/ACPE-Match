from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

_session = None
_tokenizer = None

ONNX_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "cross_encoder_onnx"
)
ONNX_DIR = os.path.normpath(ONNX_DIR)


def _load_model():
    global _session, _tokenizer
    if _session is not None:
        return

    import onnxruntime as ort
    from transformers import AutoTokenizer

    onnx_path = os.path.join(ONNX_DIR, "model.onnx")
    print(f"  [CrossEncoder ONNX] Chargement depuis {ONNX_DIR}...")

    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 4
    sess_options.inter_op_num_threads = 2

    _session = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])
    _tokenizer = AutoTokenizer.from_pretrained(ONNX_DIR, local_files_only=True)
    print("  [CrossEncoder ONNX] Charge.")


def rerank(
    query_text: str,
    offer_texts: List[str],
    top_k: int = 20,
) -> List[Tuple[int, float]]:
    _load_model()

    if not offer_texts:
        return []

    pairs = [[query_text, ot] for ot in offer_texts]

    all_scores = []
    batch_size = 64
    for i in range(0, len(pairs), batch_size):
        batch = pairs[i : i + batch_size]
        encoded = _tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="np",
        )
        inputs = {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
        }
        if "token_type_ids" in encoded:
            inputs["token_type_ids"] = encoded["token_type_ids"].astype(np.int64)

        outputs = _session.run(None, inputs)
        scores = outputs[0].squeeze(-1).astype(np.float32)
        all_scores.extend(scores.tolist())

    all_scores = np.array(all_scores)
    top_indices = np.argsort(all_scores)[::-1][:top_k]

    return [(int(idx), float(all_scores[idx])) for idx in top_indices]

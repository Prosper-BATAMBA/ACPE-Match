"""
ranker_service.py

CatBoost Ranker for re-ranking offers after FAISS retrieval.
"""

import os
import json
import numpy as np
from catboost import CatBoost, Pool

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_PATH = os.path.join(BACKEND_DIR, "catboost_ranker.cbm")
CONFIG_PATH = os.path.join(BACKEND_DIR, "ranker_config.json")

_model = None
_feature_names = None


def load_model():
    global _model, _feature_names
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"CatBoost model not found at {MODEL_PATH}. Run train_ranker.py first.")
        _model = CatBoost()
        _model.load_model(MODEL_PATH)
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            config = json.load(f)
        _feature_names = config["feature_names"]
    return _model


def get_feature_names():
    global _feature_names
    if _feature_names is None:
        load_model()
    return _feature_names


def rerank(feature_matrix: np.ndarray) -> np.ndarray:
    """Re-rank offers using CatBoost Ranker.
    
    Args:
        feature_matrix: numpy array of shape (n_offers, n_features)
    
    Returns:
        numpy array of scores (higher = better)
    """
    model = load_model()
    pool = Pool(data=feature_matrix.astype(np.float32))
    scores = model.predict(pool)
    return scores

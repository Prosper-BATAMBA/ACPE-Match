"""
train_ranker.py

Train CatBoost Ranker (YetiRank) for offer re-ranking.
Uses FAISS top-50 retrieval + 50 features.

Usage:
    cd backend
    python train_ranker.py
"""

import sys
import os
import json
import random
import time
import math

import numpy as np
import pandas as pd
import sqlite3
import chromadb
import faiss
from catboost import CatBoost, Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
HACKATON_DIR = os.path.dirname(BACKEND_DIR)
DATA_DIR = os.path.join(HACKATON_DIR, "matching_engine", "matching_engine", "data")

N_CANDIDATES = 5000
FAISS_TOP_K = 200
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Import unified feature extraction
from app.services.feature_extractor import (
    extract_features,
    get_graphs,
)


def main():
    print("=" * 60)
    print("  TRAINING CATBOOST RANKER (YetiRank)")
    print("=" * 60)

    print("\n[1/5] Chargement des donnees...")
    gt_df = pd.read_excel(os.path.join(DATA_DIR, "raw", "Appariement_Demandeurs_Offres.xlsx"), engine="openpyxl")
    gt_df = gt_df.drop_duplicates(subset="id_demandeur", keep="first")

    conn = sqlite3.connect(os.path.join(BACKEND_DIR, "acpe.db"))
    conn.row_factory = sqlite3.Row
    cand_rows = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM candidates").fetchall()}
    offer_rows = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM job_offers").fetchall()}

    chroma = chromadb.PersistentClient(path=os.path.join(BACKEND_DIR, "chroma_data"))
    col_cand = chroma.get_collection("candidate_embeddings")
    col_off = chroma.get_collection("offer_embeddings")
    chroma_cand_ids = set(col_cand.get()["ids"])
    chroma_off_ids = set(col_off.get()["ids"])

    print(f"  Candidats ChromaDB: {len(chroma_cand_ids)}")
    print(f"  Offres ChromaDB: {len(chroma_off_ids)}")

    print("\n[2/5] Construction de l'index FAISS...")
    all_off_data = col_off.get(include=["embeddings"])
    valid_offer_ids = all_off_data["ids"]
    offer_embeddings = np.array(all_off_data["embeddings"], dtype=np.float32)
    dim = offer_embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(offer_embeddings)
    index.add(offer_embeddings)
    print(f"  FAISS index: {len(valid_offer_ids)} offres, {dim}d")

    offer_id_to_emb_idx = {oid: i for i, oid in enumerate(valid_offer_ids)}

    job_graph, secteur_graph, speciality_graph = get_graphs()

    from matching_engine import SkillNormalizer, SpecialtyNormalizer
    sn = SkillNormalizer()
    spec_norm = SpecialtyNormalizer()

    print(f"\n[3/5] Selection de {N_CANDIDATES} candidats et generation des features...")
    selected = []
    for _, row in gt_df.iterrows():
        cid = str(row["id_demandeur"]).strip()
        if cid not in chroma_cand_ids or cid not in cand_rows:
            continue
        offers = [str(row[f"id_offre{i}"]).strip() for i in [1, 2, 3]]
        valid_offers = [o for o in offers if o in chroma_off_ids and o in offer_rows]
        if len(valid_offers) >= 2:
            selected.append((cid, valid_offers))

    random.shuffle(selected)
    selected = selected[:N_CANDIDATES]
    print(f"  {len(selected)} candidats selectionnes")

    print("  Pre-loading candidate embeddings...")
    selected_cids = [cid for cid, _ in selected]
    all_cand_data = col_cand.get(ids=selected_cids, include=["embeddings"])
    cand_emb_map = {eid: np.array(emb, dtype=np.float32)
                    for eid, emb in zip(all_cand_data["ids"], all_cand_data["embeddings"])}
    print(f"  {len(cand_emb_map)} candidate embeddings loaded")

    all_features = []
    all_labels = []
    all_query_ids = []

    offer_skills_cache = {}
    cand_spec_cache = {}

    t0 = time.time()
    for idx, (cid, gt_offers) in enumerate(selected):
        if (idx + 1) % 100 == 0:
            print(f"  ... {idx+1}/{len(selected)}")

        cand = cand_rows[cid]
        cand_data = dict(cand)

        cand_embedding = cand_emb_map.get(cid)
        if cand_embedding is None:
            continue

        cand_embedding = cand_embedding.reshape(1, -1)
        faiss.normalize_L2(cand_embedding)

        scores, indices = index.search(cand_embedding, min(FAISS_TOP_K, len(valid_offer_ids)))
        top_offer_ids = [valid_offer_ids[i] for i in indices[0] if i >= 0]

        if cid not in cand_spec_cache:
            cand_spec_cache[cid] = spec_norm.normalize(cand_data.get("specialite") or "") or {}
        cand_skills = sn.extract_from_text(cand_data.get("profile_text") or "")
        cand_data["_spec_result"] = cand_spec_cache[cid]

        for oid in top_offer_ids:
            if oid not in offer_rows:
                continue
            offer = offer_rows[oid]
            offer_data = dict(offer)

            cos_sim = float(np.dot(cand_embedding[0], offer_embeddings[offer_id_to_emb_idx[oid]]))

            if oid not in offer_skills_cache:
                offer_skills_cache[oid] = sn.extract_from_text(offer_data.get("competences_recherchees") or "") or sn.extract_from_text(offer_data.get("description") or "")
            offer_skills = offer_skills_cache[oid]

            feat = extract_features(
                cand_data, offer_data, cos_sim,
                job_graph=job_graph, secteur_graph=secteur_graph, speciality_graph=speciality_graph,
                cand_skills=cand_skills, offer_skills=offer_skills,
            )
            all_features.append(feat)
            all_labels.append(1 if oid in gt_offers else 0)
            all_query_ids.append(cid)

    elapsed = time.time() - t0
    print(f"  Features generees: {len(all_features)} paires ({elapsed:.1f}s)")
    print(f"  Positifs: {sum(all_labels)}, Negatifs: {len(all_labels) - sum(all_labels)}")

    if sum(all_labels) == 0:
        print("  ERREUR: Aucun positif dans les donnees!")
        return

    print(f"\n[4/5] Entrainement CatBoost Ranker...")
    feature_names = list(all_features[0].keys())
    X = np.array([[f[fn] for fn in feature_names] for f in all_features], dtype=np.float32)
    y = np.array(all_labels)
    qids = np.array(all_query_ids)

    unique_cands = list(set(all_query_ids))
    random.shuffle(unique_cands)
    split_idx = int(len(unique_cands) * 0.8)
    train_cand_set = set(unique_cands[:split_idx])
    test_cand_set = set(unique_cands[split_idx:])

    train_mask = np.array([cid in train_cand_set for cid in qids])
    test_mask = np.array([cid in test_cand_set for cid in qids])

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    train_qids = qids[train_mask]
    test_qids = qids[test_mask]

    train_group_sizes = []
    current_qid = None
    current_count = 0
    for qid in train_qids:
        if qid != current_qid:
            if current_qid is not None:
                train_group_sizes.append(current_count)
            current_qid = qid
            current_count = 1
        else:
            current_count += 1
    if current_qid is not None:
        train_group_sizes.append(current_count)

    test_group_sizes = []
    current_qid = None
    current_count = 0
    for qid in test_qids:
        if qid != current_qid:
            if current_qid is not None:
                test_group_sizes.append(current_count)
            current_qid = qid
            current_count = 1
        else:
            current_count += 1
    if current_qid is not None:
        test_group_sizes.append(current_count)

    print(f"  Train: {len(y_train)} paires, {len(train_group_sizes)} queries")
    print(f"  Test:  {len(y_test)} paires, {len(test_group_sizes)} queries")

    train_pool = Pool(
        data=X_train,
        label=y_train,
        group_id=train_qids,
    )
    test_pool = Pool(
        data=X_test,
        label=y_test,
        group_id=test_qids,
    )

    model = CatBoost({
        "iterations": 500,
        "learning_rate": 0.1,
        "depth": 6,
        "loss_function": "YetiRank",
        "eval_metric": "NDCG:top=5",
        "task_type": "CPU",
        "random_seed": RANDOM_SEED,
        "verbose": 100,
        "early_stopping_rounds": 50,
    })

    t0 = time.time()
    model.fit(train_pool, eval_set=test_pool)
    train_time = time.time() - t0
    print(f"  Entrainement: {train_time:.1f}s")

    print(f"\n[5/5] Evaluation du modele (per-query)...")

    test_preds = model.predict(test_pool)

    from collections import defaultdict
    query_indices = defaultdict(list)
    for i, qid in enumerate(test_qids):
        query_indices[qid].append(i)

    hr = {k: [] for k in [1, 3, 5, 10]}
    ndcg = {k: [] for k in [1, 3, 5, 10]}

    for qid, indices in query_indices.items():
        preds = test_preds[indices]
        labels = y_test[indices]
        sorted_idx = np.argsort(preds)[::-1]
        sorted_labels = labels[sorted_idx]
        for k in [1, 3, 5, 10]:
            top_k = sorted_labels[:k]
            hr[k].append(1 if any(l == 1 for l in top_k) else 0)
            dcg = sum(l / math.log2(i + 2) for i, l in enumerate(top_k))
            ideal = sorted(labels, reverse=True)[:k]
            idcg = sum(l / math.log2(i + 2) for i, l in enumerate(ideal))
            ndcg[k].append(dcg / idcg if idcg > 0 else 0)

    print(f"\n  --- Hit Rate (test set, {len(query_indices)} queries) ---")
    for k in [1, 3, 5, 10]:
        print(f"  Hit Rate@{k}: {np.mean(hr[k]):.4f}")

    print(f"\n  --- NDCG (test set) ---")
    for k in [1, 3, 5, 10]:
        print(f"  NDCG@{k}: {np.mean(ndcg[k]):.4f}")

    print(f"\n  --- Top 10 feature importance ---")
    importances = model.get_feature_importance(data=train_pool)
    sorted_idx = np.argsort(importances)[::-1]
    for i in sorted_idx[:10]:
        print(f"  {feature_names[i]:35s} {importances[i]:.4f}")

    print(f"\n[SAUVEGARDE]")
    model_path = os.path.join(BACKEND_DIR, "catboost_ranker.cbm")
    model.save_model(model_path)
    print(f"  Modele: {model_path}")

    config = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "n_train_pairs": len(y_train),
        "n_test_pairs": len(y_test),
        "n_train_queries": len(train_group_sizes),
        "n_test_queries": len(test_group_sizes),
        "positive_ratio": float(sum(y_train) / len(y_train)),
        "loss_function": "YetiRank",
        "eval_metric": "NDCG:top=5",
    }
    config_path = os.path.join(BACKEND_DIR, "ranker_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"  Config: {config_path}")

    print(f"\n{'='*60}")
    print(f"  TERMINE!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

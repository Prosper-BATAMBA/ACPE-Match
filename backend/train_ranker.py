"""
train_ranker.py

Train CatBoost Ranker (YetiRank) for offer re-ranking.
Uses FAISS top-50 retrieval + 40 features.

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

EDUCATION_RANKS = {
    "NV_0_AUCUN": 0, "NV_1_PRIMARY": 1, "NV_2_COLLEGE": 2,
    "NV_3_PRO_N1": 3, "NV_4_BAC": 4, "NV_5_BAC_2": 5,
    "NV_6_BAC_3": 6, "NV_7_BAC_5": 7, "NV_8_DOCTORAT": 8,
}
TENSION_MAP = {"faible": 0, "moyenne": 1, "forte": 2}
SKILL_DOMAINS = {
    "finance": ["SKILL_FIN_COMPTA_OHADA", "SKILL_FIN_FISCALITE", "SKILL_FIN_COMPTA2", "SKILL_FIN_AUDIT", "SKILL_FIN_FINANCE", "SKILL_FIN_RECOUV_ASSUR"],
    "hse_maintenance": ["SKILL_HSE_CONFORMITE", "SKILL_HSE_PREVENTION", "SKILL_HSE_QUALITE2", "SKILL_HSE_MECA", "SKILL_HSE_ELEC2", "SKILL_HSE_PRODUCTION", "SKILL_MAINT_PREVENT", "SKILL_MAINT_ELEC"],
    "it_digital": ["SKILL_IT_OFFICE", "SKILL_IT_PYTHON", "SKILL_IT_BI_POWERBI", "SKILL_IT_SQL", "SKILL_IT_ML", "SKILL_IT_STATS", "SKILL_IT_WEBDEV", "SKILL_IT_DEVOPS", "SKILL_IT_RESEAUX", "SKILL_IT_TELECOM", "SKILL_IT_CYBERSEC", "SKILL_IT_SIG", "SKILL_IT_DIGITAL_MKT"],
    "commerce": ["SKILL_COM_NEGOCIATION", "SKILL_COM_ASSURANCE", "SKILL_COM_VENTE2", "SKILL_COM_RELATION_CLIENT", "SKILL_COM_MARKETING", "SKILL_COM_INSTITUTIONNELLE", "SKILL_COM_MEDIAS", "SKILL_COM_EVENEMENTIEL"],
    "logistique": ["SKILL_LOG_TRANSPORT", "SKILL_LOG_ENTREPOT", "SKILL_LOG_ACHATS", "SKILL_LOG_DOUANE", "SKILL_LOG_MANAGEMENT"],
    "sante": ["SKILL_SANTE_MEDICAL", "SKILL_SANTE_PHARMA", "SKILL_SANTE_RECHERCHE"],
    "btp_industrie": ["SKILL_BTP_GROS_OEUVRE", "SKILL_BTP_ENGINS", "SKILL_BTP_ETUDES", "SKILL_PETROLE_FORAGE", "SKILL_PETROLE_INGENIERIE", "SKILL_PETROLE_OFFSHORE"],
    "restauration": ["SKILL_RESTA_CUISINE", "SKILL_RESTA_HOTELLERIE", "SKILL_RESTA_SERVICES_PERSONNE"],
    "agriculture": ["SKILL_AGRI_PRODUCTION", "SKILL_AGRI_TRANSFORMATION", "SKILL_AGRI_FORESTERIE"],
    "droit_securite": ["SKILL_JURI_AFFAIRES", "SKILL_JURI_CONTENTIEUX", "SKILL_SECU_GARDIENNAGE", "SKILL_SECU_DEFENSE"],
    "marine_aero": ["SKILL_MARINE_MARCHANDE", "SKILL_AERO", "SKILL_MARINE_PORT"],
    "transversal": ["SKILL_SOFT_RELATIONNEL", "SKILL_SOFT_COMMUNICATION", "SKILL_SOFT_LEADERSHIP", "SKILL_RH_CORE", "SKILL_ADMIN_SECRETARIAT2", "SKILL_ADMIN_PILOTAGE", "SKILL_ADMIN_GESTION", "SKILL_DIR_GENERALE", "SKILL_DIR_MANAGEMENT_OP", "SKILL_DIR_PROJET"],
}


def load_knowledge_graphs():
    graphs_dir = os.path.join(DATA_DIR, "graphs")
    with open(os.path.join(graphs_dir, "job_knowledge_graph_v3.json"), encoding="utf-8") as f:
        job_graph = json.load(f)
    with open(os.path.join(graphs_dir, "secteur_knowledge_graph.json"), encoding="utf-8") as f:
        secteur_graph = json.load(f)
    with open(os.path.join(graphs_dir, "speciality_knowledge_graph.json"), encoding="utf-8") as f:
        speciality_graph = json.load(f)
    return job_graph, secteur_graph, speciality_graph


def get_education_rank(code):
    if not code:
        return 4
    return EDUCATION_RANKS.get(code, 4)


def _tokenize_french(text):
    if not text:
        return set()
    text = text.lower()
    for ch in "(),.:-;/":
        text = text.replace(ch, " ")
    return set(w for w in text.split() if len(w) > 2)


def _jaccard(set_a, set_b):
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def extract_features(cand, offer, job_graph, secteur_graph, speciality_graph):
    features = {}
    features["same_id_famille"] = int(bool(cand.get("id_famille")) and bool(offer.get("id_famille")) and cand["id_famille"] == offer["id_famille"])
    features["same_id_secteur"] = int(bool(cand.get("id_secteur")) and bool(offer.get("id_secteur")) and cand["id_secteur"] == offer["id_secteur"])
    features["same_departement"] = int(bool(cand.get("code_departement")) and bool(offer.get("code_departement")) and cand["code_departement"] == offer["code_departement"] and cand["code_departement"] != "INC")

    mobilite = (cand.get("mobilite") or "").lower()
    features["candidate_mobilite"] = 0 if "non" in mobilite else 1
    features["candidate_age"] = cand.get("age") or 30
    features["candidate_niveau_rang"] = get_education_rank(cand.get("code_niveau_etude"))
    features["education_gap"] = features["candidate_niveau_rang"] - 4
    features["candidate_has_famille"] = int(bool(cand.get("id_famille")))
    features["candidate_has_secteur"] = int(bool(cand.get("id_secteur")))
    features["offer_has_famille"] = int(bool(offer.get("id_famille")))
    features["offer_has_secteur"] = int(bool(offer.get("id_secteur")))

    cand_id_fam = cand.get("id_famille", "")
    offer_id_sect = offer.get("id_secteur", "")
    features["sector_proximity"] = 0
    if cand_id_fam and offer_id_sect and offer_id_sect in secteur_graph:
        sg = secteur_graph[offer_id_sect]
        associated_fams = sg.get("familles_specialite_associees", [])
        features["sector_proximity"] = int(cand_id_fam in associated_fams)

    offer_id_fam = offer.get("id_famille", "")
    features["family_proximity"] = 0
    if cand_id_fam and offer_id_fam:
        job_entry = job_graph.get("metiers", {}).get(cand_id_fam, {})
        if not job_entry:
            job_entry = job_graph.get("familles", {}).get(cand_id_fam, {})
        proches = job_entry.get("metiers_proches", [])
        features["family_proximity"] = int(offer_id_fam in proches)

    features["semantic_similarity"] = cand.get("_semantic_score", 0.5)
    features["n_candidate_skills"] = cand.get("_n_skills", 0)
    features["n_offer_skills"] = offer.get("_n_skills", 0)
    features["skill_gap_score"] = cand.get("_skill_gap", 0.5)
    features["has_description"] = int(bool(offer.get("description")))
    features["has_competences"] = int(bool(offer.get("competences_recherchees")))

    cand_tension = 0
    if cand.get("id_secteur") and cand["id_secteur"] in secteur_graph:
        cand_tension = TENSION_MAP.get(secteur_graph[cand["id_secteur"]].get("tension_marche", ""), 1)
    features["sector_tension"] = cand_tension
    features["candidate_profile_length"] = len(cand.get("profile_text") or "")
    features["offer_profile_length"] = len(offer.get("profile_text") or "")
    features["profile_length_ratio"] = features["candidate_profile_length"] / max(features["offer_profile_length"], 1)
    features["intitule_length"] = len(offer.get("intitule") or "")
    features["candidate_metier_length"] = len(cand.get("metier_vise") or "")
    features["offer_intitule_length"] = len(offer.get("intitule") or "")
    offer_skill_ids = set(s.get("id_skill", "") for s in offer.get("_extracted_skills", []))
    features["n_offer_skills_total"] = len(offer_skill_ids)
    features["offer_has_any_skill"] = 1 if offer_skill_ids else 0
    for domain, skill_ids in SKILL_DOMAINS.items():
        domain_skills = set(skill_ids) & offer_skill_ids
        features[f"offer_domain_{domain}_count"] = len(domain_skills)
        features[f"offer_domain_{domain}_has"] = 1 if domain_skills else 0
    features["candidate_gender"] = 1 if (cand.get("genre") or "").lower().startswith("h") else 0

    cand_spec_family = cand.get("_spec_result", {}).get("id_famille_affiliation", "")
    features["same_specialty_family"] = int(bool(cand_spec_family) and bool(offer_id_fam) and cand_spec_family == offer_id_fam)
    features["candidate_has_specialite"] = int(bool(cand_spec_family))

    contrat = (offer.get("type_contrat") or "").upper()
    features["offer_type_contrat_cdi"] = int("CDI" in contrat)
    features["offer_type_contrat_cdd"] = int("CDD" in contrat)
    features["offer_type_contrat_stage"] = int("STAGE" in contrat)

    features["candidate_qualification_length"] = len(cand.get("qualification") or "")

    cand_sec_proches = secteur_graph.get(cand.get("id_secteur", ""), {}).get("secteurs_proches", [])
    features["secteur_proximity_from_secteur_graph"] = int(offer_id_sect in cand_sec_proches)

    spec_graph_data = speciality_graph.get("graph", {})
    spec_entry = spec_graph_data.get(cand_spec_family, {})
    typical_level = spec_entry.get("niveau_etude_typique", "")
    features["family_education_gap"] = get_education_rank(cand.get("code_niveau_etude")) - get_education_rank(typical_level)

    features["offer_n_sectors_proches"] = len(secteur_graph.get(offer_id_sect, {}).get("secteurs_proches", []))
    features["cand_n_sectors_proches"] = len(cand_sec_proches)
    features["offer_n_competences_hors_ref"] = len(spec_entry.get("competences_inferred_hors_referentiel", []))

    cand_metier_tokens = _tokenize_french(cand.get("metier_vise") or "")
    offer_intitule_tokens = _tokenize_french(offer.get("intitule") or "")
    features["metier_intitule_jaccard"] = _jaccard(cand_metier_tokens, offer_intitule_tokens)
    features["metier_intitule_contains"] = int(bool(cand_metier_tokens and offer_intitule_tokens and cand_metier_tokens.issubset(offer_intitule_tokens)))
    features["cand_metier_vise_len"] = len(cand.get("metier_vise") or "")

    cand_sec_tokens = _tokenize_french(cand.get("secteur_demande") or "")
    offer_sec_tokens = _tokenize_french(offer.get("secteur") or "")
    features["secteur_demande_jaccard"] = _jaccard(cand_sec_tokens, offer_sec_tokens)

    features["same_id_sous_famille"] = int(bool(cand.get("id_sous_famille")) and bool(offer.get("id_sous_famille")) and cand["id_sous_famille"] == offer["id_sous_famille"])
    features["candidate_has_sous_famille"] = int(bool(cand.get("id_sous_famille")))
    features["offer_has_sous_famille"] = int(bool(offer.get("id_sous_famille")))

    return features


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

    job_graph, secteur_graph, speciality_graph = load_knowledge_graphs()

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
        cand_data["_n_skills"] = len(cand_skills)
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
            cand_data["_semantic_score"] = cos_sim
            offer_data["_n_skills"] = len(offer_skills)
            offer_data["_extracted_skills"] = offer_skills
            common = set(s.get("libelle_canonique") for s in cand_skills) & set(s.get("libelle_canonique") for s in offer_skills)
            cand_data["_skill_gap"] = 1 - (len(common) / max(len(offer_skills), 1)) if offer_skills else 0.5

            feat = extract_features(cand_data, offer_data, job_graph, secteur_graph, speciality_graph)
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

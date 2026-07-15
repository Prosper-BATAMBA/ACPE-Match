"""
full_evaluation_v2.py

Evaluation with FAISS retrieval + CatBoost Ranker pipeline.
Compares: FAISS-only vs bge-m3+FAISS+CatBoost.

Usage:
    cd backend
    python full_evaluation_v2.py
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


def compute_ndcg(relevance_scores, k):
    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(relevance_scores[:k]))
    ideal = sorted(relevance_scores, reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def main():
    print("=" * 70)
    print("  EVALUATION: FAISS + CATBOOST RANKER")
    print("=" * 70)

    print("\n[1/4] Chargement...")
    catboost_model = CatBoost()
    catboost_model.load_model(os.path.join(BACKEND_DIR, "catboost_ranker.cbm"))
    with open(os.path.join(BACKEND_DIR, "ranker_config.json"), encoding="utf-8") as f:
        ranker_config = json.load(f)
    feature_names = ranker_config["feature_names"]
    print(f"  CatBoost Ranker charge: {len(feature_names)} features")

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

    print("\n[2/4] Construction index FAISS...")
    offer_ids_list = list(chroma_off_ids)
    offer_embeddings = []
    valid_offer_ids = []
    for oid in offer_ids_list:
        try:
            res = col_off.get(ids=[oid], include=["embeddings"])
            emb = np.array(res["embeddings"][0])
            offer_embeddings.append(emb)
            valid_offer_ids.append(oid)
        except Exception:
            continue

    offer_embeddings = np.array(offer_embeddings, dtype=np.float32)
    dim = offer_embeddings.shape[1]
    faiss.normalize_L2(offer_embeddings)
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(offer_embeddings)
    print(f"  FAISS: {len(valid_offer_ids)} offres, {dim}d")

    job_graph, secteur_graph, speciality_graph = load_knowledge_graphs()

    from matching_engine import SkillNormalizer, SpecialtyNormalizer
    sn = SkillNormalizer()
    spec_norm = SpecialtyNormalizer()

    SCORE_THRESHOLD = 0.3

    print(f"\n[3/4] Evaluation de {N_CANDIDATES} candidats...")
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
    selected = selected[N_CANDIDATES:N_CANDIDATES * 2]
    print(f"  {len(selected)} candidats selectionnes")

    hr_catboost = {k: [] for k in [1, 3, 5, 10]}
    hr_faiss_only = {k: [] for k in [1, 3, 5, 10]}
    ndcg_catboost = {k: [] for k in [1, 3, 5, 10]}
    prec_catboost = {k: [] for k in [1, 3, 5, 10]}
    prec_faiss = {k: [] for k in [1, 3, 5, 10]}
    rec_catboost = {k: [] for k in [1, 3, 5, 10]}
    rec_faiss = {k: [] for k in [1, 3, 5, 10]}
    per_query_ap_catboost = []
    per_query_rr_catboost = []

    t0 = time.time()
    processed = 0
    skipped = 0

    offer_skills_cache = {}
    cand_spec_cache = {}
    offer_id_to_emb_idx = {oid: i for i, oid in enumerate(valid_offer_ids)}

    for idx, (cid, gt_offers) in enumerate(selected):
        if (idx + 1) % 20 == 0:
            print(f"  ... {idx+1}/{len(selected)} ({time.time()-t0:.1f}s)")

        cand = cand_rows[cid]
        cand_data = dict(cand)

        try:
            cand_emb_result = col_cand.get(ids=[cid], include=["embeddings"])
            cand_embedding = np.array(cand_emb_result["embeddings"][0], dtype=np.float32)
        except Exception:
            skipped += 1
            continue

        cand_emb_2d = cand_embedding.reshape(1, -1).copy()
        faiss.normalize_L2(cand_emb_2d)
        scores, indices = faiss_index.search(cand_emb_2d, min(FAISS_TOP_K, len(valid_offer_ids)))
        faiss_top_ids = [valid_offer_ids[i] for i in indices[0] if i >= 0]

        if not faiss_top_ids:
            skipped += 1
            continue

        if len(faiss_top_ids) < 5:
            skipped += 1
            continue

        has_gt_in_pool = any(o in faiss_top_ids for o in gt_offers)
        if not has_gt_in_pool:
            skipped += 1
            continue

        processed += 1

        if cid not in cand_spec_cache:
            cand_spec_cache[cid] = spec_norm.normalize(cand_data.get("specialite") or "") or {}
        cand_skills = sn.extract_from_text(cand_data.get("profile_text") or "")
        cand_data["_n_skills"] = len(cand_skills)
        cand_data["_spec_result"] = cand_spec_cache[cid]

        catboost_scores = []
        labels = []
        feat_rows = []

        for oid in faiss_top_ids:
            if oid not in offer_rows:
                continue
            offer = offer_rows[oid]
            offer_data = dict(offer)

            emb_idx = offer_id_to_emb_idx.get(oid)
            if emb_idx is None:
                continue
            cos_sim = float(np.dot(cand_embedding, offer_embeddings[emb_idx]))

            if oid not in offer_skills_cache:
                offer_skills_cache[oid] = sn.extract_from_text(offer_data.get("competences_recherchees") or "") or sn.extract_from_text(offer_data.get("description") or "")
            offer_skills = offer_skills_cache[oid]
            cand_data["_semantic_score"] = cos_sim
            offer_data["_n_skills"] = len(offer_skills)
            offer_data["_extracted_skills"] = offer_skills
            common = set(s.get("libelle_canonique") for s in cand_skills) & set(s.get("libelle_canonique") for s in offer_skills)
            cand_data["_skill_gap"] = 1 - (len(common) / max(len(offer_skills), 1)) if offer_skills else 0.5

            feat = extract_features(cand_data, offer_data, job_graph, secteur_graph, speciality_graph)
            feat_rows.append([feat[f] for f in feature_names])

            labels.append(1 if oid in gt_offers else 0)

        if feat_rows:
            feat_matrix = np.array(feat_rows, dtype=np.float32)
            cb_preds = catboost_model.predict(Pool(data=feat_matrix))
            catboost_scores = cb_preds.tolist()
        if sum(labels) == 0:
            skipped += 1
            continue

        catboost_scores = np.array(catboost_scores)
        labels = np.array(labels)

        sorted_idx_cb = np.argsort(catboost_scores)[::-1]
        valid_faiss_scores = scores[0][indices[0] >= 0]
        sorted_idx_faiss = np.argsort(valid_faiss_scores)[::-1]

        for k in [1, 3, 5, 10]:
            top_k_cb = sorted_idx_cb[:k]
            top_k_faiss = sorted_idx_faiss[:k]
            hr_catboost[k].append(1 if any(labels[i] == 1 for i in top_k_cb) else 0)
            hr_faiss_only[k].append(1 if any(labels[i] == 1 for i in top_k_faiss) else 0)
            rel_cb = [int(labels[i] == 1) for i in sorted_idx_cb[:k]]
            rel_faiss = [int(labels[i] == 1) for i in sorted_idx_faiss[:k]]
            ndcg_catboost[k].append(compute_ndcg(rel_cb, k))
            n_relevant = sum(labels)
            prec_catboost[k].append(sum(labels[i] for i in top_k_cb) / k)
            prec_faiss[k].append(sum(labels[i] for i in top_k_faiss) / k)
            rec_catboost[k].append(sum(labels[i] for i in top_k_cb) / n_relevant)
            rec_faiss[k].append(sum(labels[i] for i in top_k_faiss) / n_relevant)

        ap_cb = 0.0
        n_rel = sum(labels)
        n_rel_cb = 0
        for i, pos in enumerate(sorted_idx_cb):
            if labels[pos] == 1:
                n_rel_cb += 1
                ap_cb += n_rel_cb / (i + 1)
        ap_cb = ap_cb / n_rel if n_rel > 0 else 0.0
        per_query_ap_catboost.append(ap_cb)

        first_hit_cb = next((i for i, pos in enumerate(sorted_idx_cb) if labels[pos] == 1), len(sorted_idx_cb))
        per_query_rr_catboost.append(1.0 / (first_hit_cb + 1) if first_hit_cb < len(sorted_idx_cb) else 0.0)

    elapsed = time.time() - t0

    print(f"\n[4/4] Resultats ({elapsed:.1f}s, {processed} candidats, {skipped} ignores)")

    valid_ap = [ap for ap in per_query_ap_catboost if ap > 0]
    map_cb = np.mean(valid_ap) if valid_ap else 0
    mrr_cb = np.mean(per_query_rr_catboost) if per_query_rr_catboost else 0

    print()
    print("=" * 70)
    print("  RAPPORT D'EVALUATION - FAISS + CATBOOST RANKER")
    print("  {} candidats x FAISS top-{} -> CatBoost".format(processed, FAISS_TOP_K))
    print("=" * 70)

    print("\n" + "-" * 70)
    print("  HIT RATE@K")
    print("-" * 70)
    print(f"  {'K':<8} {'FAISS Only':>12} {'CatBoost':>12}")
    print(f"  {'-'*34}")
    for k in [1, 3, 5, 10]:
        hf = np.mean(hr_faiss_only[k]) if hr_faiss_only[k] else 0
        hc = np.mean(hr_catboost[k]) if hr_catboost[k] else 0
        print(f"  @{k:<7} {hf:>11.1%} {hc:>11.1%}")

    print("\n" + "-" * 70)
    print("  NDCG@K")
    print("-" * 70)
    print(f"  {'K':<8} {'CatBoost':>12}")
    print(f"  {'-'*22}")
    for k in [1, 3, 5, 10]:
        nc = np.mean(ndcg_catboost[k]) if ndcg_catboost[k] else 0
        print(f"  @{k:<7} {nc:>11.4f}")

    print("\n" + "-" * 70)
    print("  PRECISION@K")
    print("-" * 70)
    print(f"  {'K':<8} {'FAISS Only':>12} {'CatBoost':>12}")
    print(f"  {'-'*34}")
    for k in [1, 3, 5, 10]:
        pf = np.mean(prec_faiss[k]) if prec_faiss[k] else 0
        pc = np.mean(prec_catboost[k]) if prec_catboost[k] else 0
        print(f"  @{k:<7} {pf:>11.1%} {pc:>11.1%}")

    print("\n" + "-" * 70)
    print("  RECALL@K")
    print("-" * 70)
    print(f"  {'K':<8} {'FAISS Only':>12} {'CatBoost':>12}")
    print(f"  {'-'*34}")
    for k in [1, 3, 5, 10]:
        rf = np.mean(rec_faiss[k]) if rec_faiss[k] else 0
        rc = np.mean(rec_catboost[k]) if rec_catboost[k] else 0
        print(f"  @{k:<7} {rf:>11.1%} {rc:>11.1%}")

    print("\n" + "-" * 70)
    print("  MRR & MAP")
    print("-" * 70)
    print(f"  {'Metrique':<25} {'CatBoost':>12}")
    print(f"  {'-'*38}")
    print(f"  {'MRR':<25} {mrr_cb:>11.4f}")
    print(f"  {'MAP':<25} {map_cb:>11.4f}")

    print("\n" + "-" * 70)
    print("  RESUME")
    print("-" * 70)
    print(f"""
  Pipeline: bge-m3 -> FAISS top-{FAISS_TOP_K} -> CatBoost ({len(feature_names)} features)
  Candidats: {processed}, Pool: {FAISS_TOP_K} offres (FAISS retrieval)
  -----------------------------------------------------------
  Hit Rate@1:     {np.mean(hr_catboost[1]):.1%}
  Hit Rate@3:     {np.mean(hr_catboost[3]):.1%}
  Hit Rate@5:     {np.mean(hr_catboost[5]):.1%}
  Hit Rate@10:    {np.mean(hr_catboost[10]):.1%}
  NDCG@3:         {np.mean(ndcg_catboost[3]):.4f}
  NDCG@10:        {np.mean(ndcg_catboost[10]):.4f}
  Precision@3:    {np.mean(prec_catboost[3]):.1%}
  Precision@5:    {np.mean(prec_catboost[5]):.1%}
  Recall@3:       {np.mean(rec_catboost[3]):.1%}
  Recall@5:       {np.mean(rec_catboost[5]):.1%}
  MRR:            {mrr_cb:.4f}
  MAP:            {map_cb:.4f}
  -----------------------------------------------------------
""")

    print("=" * 70)
    print("  TERMINE!")
    print("=" * 70)


if __name__ == "__main__":
    main()

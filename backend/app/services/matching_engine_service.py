from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

import numpy as np
from sqlalchemy.orm import Session

from ..models.candidate import Candidate
from ..models.job_offer import JobOffer

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

EDUCATION_RANKS = {
    "NV_0_AUCUN": 0, "NV_1_PRIMARY": 1, "NV_2_COLLEGE": 2,
    "NV_3_PRO_N1": 3, "NV_4_BAC": 4, "NV_5_BAC_2": 5,
    "NV_6_BAC_3": 6, "NV_7_BAC_5": 7, "NV_8_DOCTORAT": 8,
}

_job_graph: dict = {}
_secteur_graph: dict = {}
_speciality_graph: dict = {}


def _load_graphs():
    global _job_graph, _secteur_graph, _speciality_graph
    if _job_graph:
        return
    base = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                        "matching_engine", "matching_engine", "data", "graphs")
    jpath = os.path.join(base, "job_knowledge_graph_v3.json")
    spath = os.path.join(base, "secteur_knowledge_graph.json")
    specpath = os.path.join(base, "speciality_knowledge_graph.json")
    if os.path.exists(jpath):
        with open(jpath, "r", encoding="utf-8") as f:
            _job_graph = json.load(f)
    if os.path.exists(spath):
        with open(spath, "r", encoding="utf-8") as f:
            _secteur_graph = json.load(f)
    if os.path.exists(specpath):
        with open(specpath, "r", encoding="utf-8") as f:
            _speciality_graph = json.load(f)


def _get_education_rank(code):
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


def _extract_features(cand: Dict, offer: Dict, semantic_score: float,
                      sn) -> Dict[str, float]:
    _load_graphs()
    features = {}

    features["same_id_famille"] = int(
        bool(cand.get("id_famille")) and bool(offer.get("id_famille"))
        and cand["id_famille"] == offer["id_famille"])
    features["same_id_secteur"] = int(
        bool(cand.get("id_secteur")) and bool(offer.get("id_secteur"))
        and cand["id_secteur"] == offer["id_secteur"])
    features["same_departement"] = int(
        bool(cand.get("code_departement")) and bool(offer.get("code_departement"))
        and cand["code_departement"] == offer["code_departement"]
        and cand["code_departement"] != "INC")

    mobilite = (cand.get("mobilite") or "").lower()
    features["candidate_mobilite"] = 0 if "non" in mobilite else 1
    features["candidate_age"] = cand.get("age") or 30
    features["candidate_niveau_rang"] = _get_education_rank(cand.get("code_niveau_etude"))
    features["education_gap"] = features["candidate_niveau_rang"] - 4
    features["candidate_has_famille"] = int(bool(cand.get("id_famille")))
    features["candidate_has_secteur"] = int(bool(cand.get("id_secteur")))
    features["offer_has_famille"] = int(bool(offer.get("id_famille")))
    features["offer_has_secteur"] = int(bool(offer.get("id_secteur")))

    cand_id_fam = cand.get("id_famille", "")
    offer_id_sect = offer.get("id_secteur", "")
    features["sector_proximity"] = 0
    if cand_id_fam and offer_id_sect and offer_id_sect in _secteur_graph:
        sg = _secteur_graph[offer_id_sect]
        associated_fams = sg.get("familles_specialite_associees", [])
        features["sector_proximity"] = int(cand_id_fam in associated_fams)

    offer_id_fam = offer.get("id_famille", "")
    features["family_proximity"] = 0
    if cand_id_fam and offer_id_fam:
        job_entry = _job_graph.get("metiers", {}).get(cand_id_fam, {})
        if not job_entry:
            job_entry = _job_graph.get("familles", {}).get(cand_id_fam, {})
        proches = job_entry.get("metiers_proches", [])
        features["family_proximity"] = int(offer_id_fam in proches)

    features["semantic_similarity"] = semantic_score

    cand_skills = sn.extract_from_text(cand.get("profile_text") or "")
    offer_skills_raw = sn.extract_from_text(offer.get("competences_recherchees") or "")
    if not offer_skills_raw:
        offer_skills_raw = sn.extract_from_text(offer.get("description") or "")

    features["n_candidate_skills"] = len(cand_skills)
    features["n_offer_skills"] = len(offer_skills_raw)

    cand_labels = {s.get("libelle_canonique", "").lower() for s in cand_skills if s.get("libelle_canonique")}
    offer_labels = {s.get("libelle_canonique", "").lower() for s in offer_skills_raw if s.get("libelle_canonique")}
    common = cand_labels & offer_labels
    features["skill_gap_score"] = 1 - (len(common) / max(len(offer_labels), 1)) if offer_labels else 0.5

    features["has_description"] = int(bool(offer.get("description")))
    features["has_competences"] = int(bool(offer.get("competences_recherchees")))

    cand_tension = 0
    if cand.get("id_secteur") and cand["id_secteur"] in _secteur_graph:
        tension_str = _secteur_graph[cand["id_secteur"]].get("tension_marche", "")
        tension_map = {"faible": 0, "moyenne": 1, "forte": 2}
        cand_tension = tension_map.get(tension_str, 1)
    features["sector_tension"] = cand_tension

    features["candidate_profile_length"] = len(cand.get("profile_text") or "")
    features["offer_profile_length"] = len(offer.get("profile_text") or "")
    features["profile_length_ratio"] = features["candidate_profile_length"] / max(features["offer_profile_length"], 1)
    features["intitule_length"] = len(offer.get("intitule") or "")
    features["candidate_metier_length"] = len(cand.get("metier_vise") or "")
    features["offer_intitule_length"] = len(offer.get("intitule") or "")

    offer_skill_ids = set(s.get("id_skill", "") for s in offer_skills_raw)
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

    cand_sec_proches = _secteur_graph.get(cand.get("id_secteur", ""), {}).get("secteurs_proches", [])
    features["secteur_proximity_from_secteur_graph"] = int(offer_id_sect in cand_sec_proches)

    spec_graph_data = _speciality_graph.get("graph", {})
    spec_entry = spec_graph_data.get(cand_spec_family, {})
    typical_level = spec_entry.get("niveau_etude_typique", "")
    features["family_education_gap"] = _get_education_rank(cand.get("code_niveau_etude")) - _get_education_rank(typical_level)

    features["offer_n_sectors_proches"] = len(_secteur_graph.get(offer_id_sect, {}).get("secteurs_proches", []))
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


def compute_skill_gap(
    candidate_skills: List[Dict[str, Any]],
    offer_skills: List[Dict[str, Any]],
) -> Dict[str, Any]:
    cand_labels = {s.get("libelle_canonique", "").lower() for s in candidate_skills if s.get("libelle_canonique")}
    offer_labels = {s.get("libelle_canonique", "").lower() for s in offer_skills if s.get("libelle_canonique")}

    missing = [s for s in offer_skills if s.get("libelle_canonique", "").lower() not in cand_labels]
    acquired = [s for s in candidate_skills if s.get("libelle_canonique", "").lower() in offer_labels]

    return {
        "acquired": [s.get("libelle_canonique") for s in acquired],
        "missing": [s.get("libelle_canonique") for s in missing],
        "gap_score": 1 - (len(missing) / max(len(offer_skills), 1)),
    }


def get_recommendations(
    db: Session,
    chroma_collection,
    candidate_id: str,
    k: int = 10,
    embedding_fn=None,
) -> List[Dict[str, Any]]:
    candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    if not candidate or not candidate.profile_text:
        return []

    from . import faiss_index as fi
    from . import ranker_service as rs
    from matching_engine import SkillNormalizer, SpecialtyNormalizer

    sn = SkillNormalizer()
    spec_norm = SpecialtyNormalizer()
    logging.debug(f"Début de get_recommendations pour candidate {candidate_id}")

    candidate_embedding = None

    try:
        from ..chromadb_client import get_candidates_collection
        cand_col = get_candidates_collection()
        stored = cand_col.get(ids=[candidate_id], include=["embeddings"])
        embeddings = stored.get("embeddings") if isinstance(stored, dict) else None
        if embeddings is not None and len(embeddings) > 0:
            candidate_embedding = np.array(embeddings[0], dtype=np.float32)
    except Exception as e:
        import traceback
        traceback.print_exc()
        logging.warning(f"ChromaDB lookup failed for {candidate_id}: {e}")

    if candidate_embedding is None:
        if embedding_fn is None:
            from .embedding_service import encode
            embedding_fn = encode
        candidate_embedding = np.array(embedding_fn(candidate.profile_text), dtype=np.float32)
        logging.debug(f"Embedding généré via encode() pour {candidate_id}")
    else:
        logging.debug(f"Embedding récupéré depuis ChromaDB pour {candidate_id}")

    candidate_embedding_normed = candidate_embedding / (np.linalg.norm(candidate_embedding) or 1.0)

    try:
        logging.debug(f"Recherche FAISS top-k={min(k * 20, 200)}")
        faiss_results = fi.search(candidate_embedding_normed, top_k=min(k * 20, 200))
        logging.debug(f"FAISS a retourné {len(faiss_results)} résultats")
    except FileNotFoundError:
        results = chroma_collection.query(
            query_embeddings=[candidate_embedding.tolist()],
            n_results=min(k * 3, 100),
            include=["documents", "distances"],
        )
        if not results["ids"][0]:
            return []
        faiss_results = [(oid, max(0, 1 - d)) for oid, d in zip(results["ids"][0], results["distances"][0])]

    cand_data = {
        "id_famille": candidate.id_famille,
        "id_sous_famille": candidate.id_sous_famille,
        "id_secteur": candidate.id_secteur,
        "code_departement": candidate.code_departement,
        "code_niveau_etude": candidate.code_niveau_etude,
        "age": candidate.age,
        "mobilite": candidate.mobilite,
        "profile_text": candidate.profile_text,
        "metier_vise": candidate.metier_vise,
        "genre": candidate.genre,
        "specialite": candidate.specialite,
        "qualification": candidate.qualification,
        "secteur_demande": candidate.secteur_demande,
    }
    cand_data["_spec_result"] = spec_norm.normalize(cand_data.get("specialite") or "") or {}

    try:
        feature_names = rs.get_feature_names()
    except FileNotFoundError:
        feature_names = None

    SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "0.3"))

    offer_ids_to_load = [oid for oid, _ in faiss_results if oid]
    offers_from_db = db.query(JobOffer).filter(JobOffer.id.in_(offer_ids_to_load)).all()
    offer_map = {o.id: o for o in offers_from_db}

    recommendations = []
    for offer_id, semantic_score in faiss_results:
        offer = offer_map.get(offer_id)
        if not offer:
            continue

        offer_data = {
            "id_famille": offer.id_famille,
            "id_sous_famille": offer.id_sous_famille,
            "id_secteur": offer.id_secteur,
            "code_departement": offer.code_departement,
            "profile_text": offer.profile_text or "",
            "intitule": offer.intitule,
            "description": offer.description,
            "competences_recherchees": offer.competences_recherchees,
            "type_contrat": offer.type_contrat,
            "secteur": offer.secteur,
        }

        if feature_names:
            feat = _extract_features(cand_data, offer_data, semantic_score, sn)
            feat_vec = np.array([[feat[f] for f in feature_names]], dtype=np.float32)
            try:
                score = float(rs.rerank(feat_vec)[0])
            except Exception:
                score = semantic_score
        else:
            score = semantic_score

        if score < SCORE_THRESHOLD:
            continue

        candidate_skills = sn.extract_from_text(candidate.profile_text)
        offer_skills = []
        if offer.competences_recherchees:
            offer_skills = sn.extract_from_text(offer.competences_recherchees)
        if offer.description and not offer_skills:
            offer_skills = sn.extract_from_text(offer.description)

        skill_gap = compute_skill_gap(candidate_skills, offer_skills)

        recommendations.append({
            "offer_id": offer_id,
            "intitule": offer.intitule,
            "entreprise": offer.entreprise,
            "score": round(score, 4),
            "skill_gap": skill_gap,
        })

    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations[:k]


def get_candidates_for_offer(
    db: Session,
    chroma_collection,
    offer_id: str,
    k: int = 10,
    embedding_fn=None,
) -> List[Dict[str, Any]]:
    offer = db.query(JobOffer).filter(JobOffer.id == offer_id).first()
    if not offer or not offer.profile_text:
        return []

    from . import ranker_service as rs
    from matching_engine import SkillNormalizer, SpecialtyNormalizer

    sn = SkillNormalizer()
    spec_norm = SpecialtyNormalizer()
    logging.debug(f"Debut de get_candidates_for_offer pour offer {offer_id}")

    offer_embedding = None
    try:
        from ..chromadb_client import get_offers_collection
        off_col = get_offers_collection()
        stored = off_col.get(ids=[offer_id], include=["embeddings"])
        embeddings = stored.get("embeddings") if isinstance(stored, dict) else None
        if embeddings is not None and len(embeddings) > 0:
            offer_embedding = np.array(embeddings[0], dtype=np.float32)
    except Exception as e:
        logging.warning(f"ChromaDB offer lookup failed for {offer_id}: {e}")

    if offer_embedding is None:
        if embedding_fn is None:
            from .embedding_service import encode
            embedding_fn = encode
        offer_embedding = np.array(embedding_fn(offer.profile_text), dtype=np.float32)
        logging.debug(f"Embedding offre genere via encode() pour {offer_id}")
    else:
        logging.debug(f"Embedding offre recupere depuis ChromaDB pour {offer_id}")

    offer_embedding_normed = offer_embedding / (np.linalg.norm(offer_embedding) or 1.0)

    from ..chromadb_client import get_candidates_collection
    cand_col = get_candidates_collection()
    retrieval = min(k * 20, 200)
    results = cand_col.query(
        query_embeddings=[offer_embedding_normed.tolist()],
        n_results=retrieval,
        include=["distances"],
    )
    cand_ids = results.get("ids", [[]])[0] if results.get("ids") else []
    distances = results.get("distances", [[]])[0] if results.get("distances") else []
    candidate_scores = [(cid, max(0.0, 1.0 - d)) for cid, d in zip(cand_ids, distances)]

    offer_data = {
        "id_famille": offer.id_famille,
        "id_sous_famille": offer.id_sous_famille,
        "id_secteur": offer.id_secteur,
        "code_departement": offer.code_departement,
        "profile_text": offer.profile_text or "",
        "intitule": offer.intitule,
        "description": offer.description,
        "competences_recherchees": offer.competences_recherchees,
        "type_contrat": offer.type_contrat,
        "secteur": offer.secteur,
    }

    candidate_ids_to_load = [cid for cid, _ in candidate_scores if cid]
    candidates_from_db = db.query(Candidate).filter(Candidate.id.in_(candidate_ids_to_load)).all()
    cand_map = {c.id: c for c in candidates_from_db}

    try:
        feature_names = rs.get_feature_names()
    except FileNotFoundError:
        feature_names = None

    SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "0.3"))

    recommendations = []
    for cand_id, semantic_score in candidate_scores:
        candidate = cand_map.get(cand_id)
        if not candidate or not candidate.profile_text:
            continue

        cand_data = {
            "id_famille": candidate.id_famille,
            "id_sous_famille": candidate.id_sous_famille,
            "id_secteur": candidate.id_secteur,
            "code_departement": candidate.code_departement,
            "code_niveau_etude": candidate.code_niveau_etude,
            "age": candidate.age,
            "mobilite": candidate.mobilite,
            "profile_text": candidate.profile_text,
            "metier_vise": candidate.metier_vise,
            "genre": candidate.genre,
            "specialite": candidate.specialite,
            "qualification": candidate.qualification,
            "secteur_demande": candidate.secteur_demande,
        }
        cand_data["_spec_result"] = spec_norm.normalize(cand_data.get("specialite") or "") or {}

        if feature_names:
            feat = _extract_features(cand_data, offer_data, semantic_score, sn)
            feat_vec = np.array([[feat[f] for f in feature_names]], dtype=np.float32)
            try:
                score = float(rs.rerank(feat_vec)[0])
            except Exception:
                score = semantic_score
        else:
            score = semantic_score

        if score < SCORE_THRESHOLD:
            continue

        candidate_skills = sn.extract_from_text(candidate.profile_text)
        offer_skills = []
        if offer.competences_recherchees:
            offer_skills = sn.extract_from_text(offer.competences_recherchees)
        if offer.description and not offer_skills:
            offer_skills = sn.extract_from_text(offer.description)

        skill_gap = compute_skill_gap(candidate_skills, offer_skills)

        recommendations.append({
            "candidate_id": cand_id,
            "score": round(score, 4),
            "skill_gap": skill_gap,
        })

    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations[:k]

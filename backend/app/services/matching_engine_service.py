from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
from sqlalchemy.orm import Session

from ..models.candidate import Candidate
from ..models.job_offer import JobOffer

SKILL_IDS = [
    "SKILL_FIN_COMPTA_OHADA", "SKILL_FIN_FISCALITE",
    "SKILL_HSE_CONFORMITE", "SKILL_HSE_PREVENTION",
    "SKILL_MAINT_PREVENT", "SKILL_MAINT_ELEC",
    "SKILL_COM_NEGOCIATION", "SKILL_COM_ASSURANCE",
    "SKILL_ADMIN_GESTION", "SKILL_IT_OFFICE",
    "SKILL_SOFT_RELATIONNEL", "SKILL_SOFT_COMMUNICATION",
    "SKILL_SOFT_LEADERSHIP",
]

EDUCATION_RANKS = {
    "NV_1_PRIMAIRE": 1, "NV_2_COLLEGE": 2, "NV_3_LYCEE": 3,
    "NV_4_BAC": 4, "NV_5_BAC2": 5, "NV_6_BAC3": 6,
    "NV_7_BAC4": 7, "NV_8_BAC5": 8,
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
    jpath = os.path.join(base, "job_knowledge_graph.json")
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
    features["skill_gap_score"] = 1 - (len(offer_labels - cand_labels) / max(len(offer_labels), 1))

    features["has_description"] = int(bool(offer.get("description")))
    features["has_competences"] = int(bool(offer.get("competences_recherchees")))

    cand_tension = 0
    if cand.get("id_secteur") and cand["id_secteur"] in _secteur_graph:
        tension_str = _secteur_graph[cand["id_secteur"]].get("tension_marche", "")
        tension_map = {"forte": 3, "moderee": 2, "faible": 1}
        cand_tension = tension_map.get(tension_str, 1)
    features["sector_tension"] = cand_tension

    features["candidate_profile_length"] = len(cand.get("profile_text") or "")
    features["offer_profile_length"] = len(offer.get("profile_text") or "")
    features["profile_length_ratio"] = features["candidate_profile_length"] / max(features["offer_profile_length"], 1)
    features["intitule_length"] = len(offer.get("intitule") or "")
    features["candidate_metier_length"] = len(cand.get("metier_vise") or "")
    features["offer_intitule_length"] = len(offer.get("intitule") or "")

    offer_skill_ids = set(s.get("id_skill", "") for s in offer_skills_raw)
    for skill_id in SKILL_IDS:
        features[f"offer_skill_{skill_id}"] = 1 if skill_id in offer_skill_ids else 0
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

    if embedding_fn is None:
        from .embedding_service import encode
        embedding_fn = encode

    candidate_embedding = np.array(embedding_fn(candidate.profile_text), dtype=np.float32)
    candidate_embedding_normed = candidate_embedding / (np.linalg.norm(candidate_embedding) or 1.0)

    from matching_engine import SkillNormalizer, SpecialtyNormalizer
    from . import faiss_index as fi
    from . import ranker_service as rs

    sn = SkillNormalizer()
    spec_norm = SpecialtyNormalizer()

    try:
        faiss_results = fi.search(candidate_embedding_normed, top_k=min(k * 20, 200))
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
        "secteur_demande": candidate.secteur_demande if hasattr(candidate, 'secteur_demande') else None,
    }
    cand_data["_spec_result"] = spec_norm.normalize(cand_data.get("specialite") or "") or {}

    try:
        feature_names = rs.get_feature_names()
    except FileNotFoundError:
        feature_names = None

    SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "0.3"))
    recommendations = []
    for offer_id, semantic_score in faiss_results:
        offer = db.query(JobOffer).filter(JobOffer.id == offer_id).first()
        if not offer:
            continue

        offer_data = {
            "id_famille": offer.id_famille,
            "id_secteur": offer.id_secteur,
            "code_departement": None,
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

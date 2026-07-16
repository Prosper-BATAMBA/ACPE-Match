from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Any, Dict, List

import numpy as np
from sqlalchemy.orm import Session

from ..models.candidate import Candidate
from ..models.job_offer import JobOffer

from .feature_extractor import (
    SKILL_DOMAINS,
    SKILL_ID_TO_DOMAIN,
    DOMAIN_TO_SECTOR,
    EDUCATION_RANKS,
    extract_features,
    tokenize_french,
    jaccard,
    get_education_rank,
    get_graphs,
)


# ---------------------------------------------------------------------------
# Cached normalizer singletons (avoids re-instantiation per request)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_skill_normalizer():
    from matching_engine import SkillNormalizer
    return SkillNormalizer()


@lru_cache(maxsize=1)
def _get_specialty_normalizer():
    from matching_engine import SpecialtyNormalizer
    return SpecialtyNormalizer()


@lru_cache(maxsize=1)
def _get_sector_normalizer():
    from matching_engine import SectorNormalizer
    return SectorNormalizer()


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

    sn = _get_skill_normalizer()
    spec_norm = _get_specialty_normalizer()
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

    # --- OPTIMIZATION: hoist extract_from_text() calls outside the loop ---
    candidate_skills = sn.extract_from_text(candidate.profile_text)

    try:
        feature_names = rs.get_feature_names()
    except FileNotFoundError:
        feature_names = None

    SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "0.3"))

    offer_ids_to_load = [oid for oid, _ in faiss_results if oid]
    offers_from_db = db.query(JobOffer).filter(JobOffer.id.in_(offer_ids_to_load)).all()
    offer_map = {o.id: o for o in offers_from_db}

    # Pre-extract offer skills for all offers in one pass
    offer_skills_map: Dict[str, list] = {}
    for offer_id, _ in faiss_results:
        offer = offer_map.get(offer_id)
        if not offer:
            continue
        offer_skills_raw = sn.extract_from_text(offer.competences_recherchees or "")
        if not offer_skills_raw:
            offer_skills_raw = sn.extract_from_text(offer.description or "")
        offer_skills_map[offer_id] = offer_skills_raw

    job_graph, secteur_graph, speciality_graph = get_graphs()

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

        offer_skills = offer_skills_map.get(offer_id, [])

        if feature_names:
            feat = extract_features(
                cand_data, offer_data, semantic_score,
                job_graph=job_graph, secteur_graph=secteur_graph, speciality_graph=speciality_graph,
                cand_skills=candidate_skills, offer_skills=offer_skills,
            )
            feat_vec = np.array([[feat[f] for f in feature_names]], dtype=np.float32)
            try:
                score = float(rs.rerank(feat_vec)[0])
            except Exception:
                score = semantic_score
        else:
            score = semantic_score

        if score < SCORE_THRESHOLD:
            continue

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

    sn = _get_skill_normalizer()
    spec_norm = _get_specialty_normalizer()
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

    # --- OPTIMIZATION: pre-extract offer skills once ---
    offer_skills = sn.extract_from_text(offer.competences_recherchees or "")
    if not offer_skills:
        offer_skills = sn.extract_from_text(offer.description or "")

    candidate_ids_to_load = [cid for cid, _ in candidate_scores if cid]
    candidates_from_db = db.query(Candidate).filter(Candidate.id.in_(candidate_ids_to_load)).all()
    cand_map = {c.id: c for c in candidates_from_db}

    try:
        feature_names = rs.get_feature_names()
    except FileNotFoundError:
        feature_names = None

    SCORE_THRESHOLD = float(os.environ.get("SCORE_THRESHOLD", "0.3"))

    job_graph, secteur_graph, speciality_graph = get_graphs()

    # --- OPTIMIZATION: batch-load candidate skills ---
    candidate_skills_map: Dict[str, list] = {}
    for cand_id, _ in candidate_scores:
        candidate = cand_map.get(cand_id)
        if not candidate or not candidate.profile_text:
            continue
        candidate_skills_map[cand_id] = sn.extract_from_text(candidate.profile_text)

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

        cand_skills = candidate_skills_map.get(cand_id, [])

        if feature_names:
            feat = extract_features(
                cand_data, offer_data, semantic_score,
                job_graph=job_graph, secteur_graph=secteur_graph, speciality_graph=speciality_graph,
                cand_skills=cand_skills, offer_skills=offer_skills,
            )
            feat_vec = np.array([[feat[f] for f in feature_names]], dtype=np.float32)
            try:
                score = float(rs.rerank(feat_vec)[0])
            except Exception:
                score = semantic_score
        else:
            score = semantic_score

        if score < SCORE_THRESHOLD:
            continue

        skill_gap = compute_skill_gap(cand_skills, offer_skills)

        recommendations.append({
            "candidate_id": cand_id,
            "score": round(score, 4),
            "skill_gap": skill_gap,
        })

    recommendations.sort(key=lambda x: x["score"], reverse=True)
    return recommendations[:k]


def nl_offer_search(
    db: Session,
    query: str,
    k: int = 10,
    embedding_fn=None,
) -> Dict[str, Any]:
    """Recherche d'offres en langage naturel (robuste, semantique-first).

    Approche generalisable (pas de patch cas-par-cas) :
      1. Extraction des competences/secteur depuis le texte libre (Normalizers),
         puis inference du secteur cible a partir du domaine majoritaire des
         competences detectees (utile quand le texte ne nomme pas le secteur).
      2. Alignement de representation : on reconstruit un `profile_text`
         normalise via le MEME pipeline de denormalisation que les offres
         (ProfileBuilder), puis on l'embed pour le FAISS. On compare donc
         offres contre offres (espace d'embedding aligne) au lieu d'une
         auto-description contre des offres.
      3. Ranking semantique en premier : similarite FAISS + boosts (competences
         matchees, secteur cible). Le seuil CatBoost 0.3 est abandonne (il est
         mal calibre pour une entree candidate synthetique sparse -> tous les
         scores sont negatifs). Le score CatBoost est conserve en sortie a
         titre informatif uniquement.
    """
    empty = {
        "query": query,
        "query_profile_text": "",
        "extracted_skills": [],
        "target_secteur": None,
        "total_offers_compared": 0,
        "recommendations": [],
    }
    if not query or not query.strip():
        return empty

    from . import faiss_index as fi
    from . import ranker_service as rs
    from .profile_builder import ProfileBuilder

    sn = _get_skill_normalizer()
    sec_norm = _get_sector_normalizer()
    pb = ProfileBuilder()

    # --- OPTIMIZATION: hoist extract_from_text() outside loop ---
    cand_skills = sn.extract_from_text(query)
    detected_skill_ids = [s.get("id_skill") for s in cand_skills if s.get("id_skill")]
    extracted_skill_labels = sorted(
        {s.get("libelle_canonique", "") for s in cand_skills if s.get("libelle_canonique")}
    )

    target_secteurs = sec_norm.extract_from_text(query)
    target_secteur = target_secteurs[0] if target_secteurs else None

    # Inference de secteur a partir des domaines de competences detectees.
    if not target_secteur or target_secteur.get("id_secteur") == "SEC_AUTRES":
        domain_counts = {}
        for sid in detected_skill_ids:
            dom = SKILL_ID_TO_DOMAIN.get(sid)
            if dom and dom != "transversal":
                domain_counts[dom] = domain_counts.get(dom, 0) + 1
        if domain_counts:
            best_domain = max(domain_counts.items(), key=lambda kv: kv[1])[0]
            inferred_id = DOMAIN_TO_SECTOR.get(best_domain)
            if inferred_id and inferred_id != "SEC_AUTRES":
                target_secteur = sec_norm.sector_metadata.get(inferred_id)

    # 2. Alignement de representation : profil offre normalise.
    pb_result = pb.build_job_offer_profile(
        secteur=target_secteur["secteur_canonique"] if target_secteur else None,
        competences_recherchees=", ".join(extracted_skill_labels) if extracted_skill_labels else None,
    )
    query_profile_text = (pb_result.get("profile_text") or "").strip()

    if embedding_fn is None:
        from .embedding_service import encode
        embedding_fn = encode

    text_to_embed = query_profile_text if len(query_profile_text) > 10 else query
    q_emb = np.array(embedding_fn(text_to_embed), dtype=np.float32)
    q_emb_normed = q_emb / (np.linalg.norm(q_emb) or 1.0)

    try:
        faiss_results = fi.search(q_emb_normed, top_k=min(k * 20, 200))
    except FileNotFoundError:
        empty["error"] = (
            "Index FAISS introuvable. Executez build_faiss_index.py apres le seed."
        )
        return empty

    cand_data = {
        "id_famille": None,
        "id_sous_famille": None,
        "id_secteur": target_secteur["id_secteur"] if target_secteur else None,
        "code_departement": None,
        "code_niveau_etude": None,
        "age": None,
        "mobilite": None,
        "profile_text": query,
        "metier_vise": None,
        "genre": None,
        "specialite": None,
        "qualification": None,
        "secteur_demande": target_secteur["secteur_canonique"] if target_secteur else None,
    }
    cand_data["_spec_result"] = {}

    try:
        feature_names = rs.get_feature_names()
    except FileNotFoundError:
        feature_names = None

    offer_ids_to_load = [oid for oid, _ in faiss_results if oid]
    offers_from_db = db.query(JobOffer).filter(JobOffer.id.in_(offer_ids_to_load)).all()
    offer_map = {o.id: o for o in offers_from_db}

    job_graph, secteur_graph, speciality_graph = get_graphs()

    # --- OPTIMIZATION: batch pre-extract offer skills ---
    offer_skills_map: Dict[str, list] = {}
    for offer_id, _ in faiss_results:
        offer = offer_map.get(offer_id)
        if not offer:
            continue
        offer_skills_raw = sn.extract_from_text(offer.competences_recherchees or "")
        if not offer_skills_raw:
            offer_skills_raw = sn.extract_from_text(offer.description or "")
        offer_skills_map[offer_id] = offer_skills_raw

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

        offer_skills = offer_skills_map.get(offer_id, [])

        offer_labels = sorted(
            {s.get("libelle_canonique", "") for s in offer_skills if s.get("libelle_canonique")}
        )
        matched = sorted(set(extracted_skill_labels) & set(offer_labels))
        sector_match = bool(
            target_secteur and offer.id_secteur
            and target_secteur["id_secteur"] == offer.id_secteur
        )

        # 3. Ranking semantique + boosts (pas de gate CatBoost 0.3).
        relevance = semantic_score
        relevance += 0.05 * len(matched)
        relevance += 0.10 * (1 if sector_match else 0)

        # Score CatBoost conserve a titre informatif.
        catboost_score = None
        if feature_names:
            feat = extract_features(
                cand_data, offer_data, semantic_score,
                job_graph=job_graph, secteur_graph=secteur_graph, speciality_graph=speciality_graph,
                cand_skills=cand_skills, offer_skills=offer_skills,
            )
            feat_vec = np.array([[feat[f] for f in feature_names]], dtype=np.float32)
            try:
                catboost_score = float(rs.rerank(feat_vec)[0])
            except Exception:
                catboost_score = None

        recommendations.append({
            "offer_id": offer_id,
            "intitule": offer.intitule,
            "entreprise": offer.entreprise,
            "secteur": offer.secteur,
            "score": round(relevance, 4),
            "semantic_score": round(semantic_score, 4),
            "catboost_score": round(catboost_score, 4) if catboost_score is not None else None,
            "matched_skills": matched,
            "sector_match": sector_match,
        })

    recommendations.sort(key=lambda x: x["score"], reverse=True)
    empty["query_profile_text"] = query_profile_text
    empty["extracted_skills"] = extracted_skill_labels
    empty["target_secteur"] = target_secteur
    empty["total_offers_compared"] = len(faiss_results)
    empty["recommendations"] = recommendations[:k]
    return empty

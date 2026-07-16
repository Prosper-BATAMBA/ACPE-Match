"""
feature_extractor.py

Unified feature extraction for matching engine and CatBoost ranker training.
Shared constants, tokenizers, and feature computation.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Set

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

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

SKILL_ID_TO_DOMAIN = {}
for _dom, _ids in SKILL_DOMAINS.items():
    for _sid in _ids:
        SKILL_ID_TO_DOMAIN[_sid] = _dom

DOMAIN_TO_SECTOR = {
    "finance": "SEC_FIN_COMPTA",
    "hse_maintenance": "SEC_INDU_MAINT",
    "it_digital": "SEC_IT_DIGITAL",
    "commerce": "SEC_COMM_VENTE",
    "logistique": "SEC_TRANS_LOG",
    "sante": "SEC_SANTE_PHARMA",
    "btp_industrie": "SEC_BTP_AMENAGE",
    "restauration": "SEC_HOTEL_RESTAU",
    "agriculture": "SEC_AGRO",
    "droit_securite": "SEC_DROIT_SOCIAL",
    "marine_aero": "SEC_TRANS_LOG",
    "transversal": None,
}

EDUCATION_RANKS = {
    "NV_0_AUCUN": 0, "NV_1_PRIMARY": 1, "NV_2_COLLEGE": 2,
    "NV_3_PRO_N1": 3, "NV_4_BAC": 4, "NV_5_BAC_2": 5,
    "NV_6_BAC_3": 6, "NV_7_BAC_5": 7, "NV_8_DOCTORAT": 8,
}

TENSION_MAP = {"faible": 0, "moyenne": 1, "forte": 2}

# French stopwords (common words that don't carry skill/sector meaning)
FRENCH_STOPWORDS = {
    "le", "la", "les", "de", "du", "des", "un", "une", "et", "en", "au",
    "aux", "ce", "que", "qui", "dans", "pour", "par", "pas", "sur", "plus",
    "ne", "se", "son", "sa", "ses", "avec", "nous", "vous", "ils", "elle",
    "tout", "cette", "mais", "ou", "donc", "car", "si", "bien", "aussi",
    "tres", "peu", "meme", "encore", "entre", "apres", "avant", "chez",
    "comment", "combien", "quel", "quelle", "quels", "quelles", "mon",
    "ton", "votre", "notre", "leurs", "ces", "quelques", "chaque",
    "autre", "meme", "tel", "telle", "tels", "telles", "type", "peut",
    "fait", "faire", "dit", "etre", "avoir", "aller", "sans", "sous",
}

# ---------------------------------------------------------------------------
# Lazy-loaded graph singletons
# ---------------------------------------------------------------------------

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
            raw = json.load(f)
            _secteur_graph = raw.get("graph", raw)
    if os.path.exists(specpath):
        with open(specpath, "r", encoding="utf-8") as f:
            raw = json.load(f)
            _speciality_graph = raw.get("graph", raw)


def get_graphs() -> tuple:
    _load_graphs()
    return _job_graph, _secteur_graph, _speciality_graph

# ---------------------------------------------------------------------------
# Tokenizer & similarity helpers
# ---------------------------------------------------------------------------


def tokenize_french(text: str) -> Set[str]:
    """Tokenize French text, removing punctuation and stopwords."""
    if not text:
        return set()
    text = text.lower()
    for ch in "(),.:-;/":
        text = text.replace(ch, " ")
    return set(w for w in text.split() if len(w) > 2 and w not in FRENCH_STOPWORDS)


def jaccard(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a or not set_b:
        return 0.0
    inter = len(set_a & set_b)
    union = len(set_a | set_b)
    return inter / union if union > 0 else 0.0


def get_education_rank(code: Optional[str]) -> int:
    if not code:
        return 4
    return EDUCATION_RANKS.get(code, 4)


def _get_primary_domain(skill_ids: set) -> str:
    """Return the domain with the most skill IDs, or empty string."""
    domain_counts: Dict[str, int] = {}
    for sid in skill_ids:
        dom = SKILL_ID_TO_DOMAIN.get(sid)
        if dom:
            domain_counts[dom] = domain_counts.get(dom, 0) + 1
    if domain_counts:
        return max(domain_counts, key=domain_counts.get)
    return ""


# ---------------------------------------------------------------------------
# Unified feature extraction
# ---------------------------------------------------------------------------


def extract_features(
    cand: Dict[str, Any],
    offer: Dict[str, Any],
    semantic_score: float,
    sn=None,
    job_graph: Optional[dict] = None,
    secteur_graph: Optional[dict] = None,
    speciality_graph: Optional[dict] = None,
    cand_skills: Optional[List[Dict]] = None,
    offer_skills: Optional[List[Dict]] = None,
) -> Dict[str, float]:
    """Compute features for candidate-offer pair.

    If ``cand_skills`` / ``offer_skills`` are provided (pre-extracted), they
    are used directly. Otherwise, ``sn.extract_from_text()`` is called.
    If graphs are provided they are used; otherwise the lazy singletons are loaded.
    """
    if job_graph is None or secteur_graph is None or speciality_graph is None:
        _jg, _sg, _spg = get_graphs()
        job_graph = job_graph or _jg
        secteur_graph = secteur_graph or _sg
        speciality_graph = speciality_graph or _spg

    features: Dict[str, float] = {}

    # --- categorical matches ---
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

    # --- candidate demographics ---
    mobilite = (cand.get("mobilite") or "").lower()
    features["candidate_mobilite"] = 0 if "non" in mobilite else 1
    features["candidate_age"] = cand.get("age") or 30
    features["candidate_niveau_rang"] = get_education_rank(cand.get("code_niveau_etude"))
    features["education_gap"] = features["candidate_niveau_rang"] - 4
    features["candidate_has_famille"] = int(bool(cand.get("id_famille")))
    features["candidate_has_secteur"] = int(bool(cand.get("id_secteur")))
    features["offer_has_famille"] = int(bool(offer.get("id_famille")))
    features["offer_has_secteur"] = int(bool(offer.get("id_secteur")))

    # --- graph-based proximity ---
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

    # --- semantic similarity ---
    features["semantic_similarity"] = semantic_score

    # --- skill extraction ---
    if cand_skills is not None:
        cand_skills_raw = cand_skills
    elif sn is not None:
        cand_skills_raw = sn.extract_from_text(cand.get("profile_text") or "")
    else:
        cand_skills_raw = []

    if offer_skills is not None:
        offer_skills_raw = offer_skills
    elif sn is not None:
        offer_skills_raw = sn.extract_from_text(offer.get("competences_recherchees") or "")
        if len(offer_skills_raw) < 2:
            combined = f"{offer.get('intitule') or ''} {offer.get('description') or ''} {offer.get('competences_recherchees') or ''}"
            offer_skills_raw = sn.extract_from_text(combined)
    else:
        offer_skills_raw = []

    features["n_candidate_skills"] = len(cand_skills_raw)
    features["n_offer_skills"] = len(offer_skills_raw)

    cand_labels = {s.get("libelle_canonique", "").lower() for s in cand_skills_raw if s.get("libelle_canonique")}
    offer_labels = {s.get("libelle_canonique", "").lower() for s in offer_skills_raw if s.get("libelle_canonique")}
    common = cand_labels & offer_labels
    features["skill_gap_score"] = 1 - (len(common) / max(len(offer_labels), 1)) if offer_labels else 0.5

    # --- new skill-match features ---
    features["has_all_required_skills"] = int(
        bool(offer_labels) and offer_labels.issubset(cand_labels)
    )
    features["skill_match_ratio"] = len(common) / max(len(offer_labels), 1) if offer_labels else 0.0

    cand_skill_ids = set(s.get("id_skill", "") for s in cand_skills_raw)
    offer_skill_ids = set(s.get("id_skill", "") for s in offer_skills_raw)
    cand_domain = _get_primary_domain(cand_skill_ids)
    offer_domain = _get_primary_domain(offer_skill_ids)
    features["domain_match"] = int(bool(cand_domain) and cand_domain == offer_domain)

    features["has_description"] = int(bool(offer.get("description")))
    features["has_competences"] = int(bool(offer.get("competences_recherchees")))

    # --- sector tension ---
    cand_tension = 0
    if cand.get("id_secteur") and cand["id_secteur"] in secteur_graph:
        tension_str = secteur_graph[cand["id_secteur"]].get("tension_marche", "")
        cand_tension = TENSION_MAP.get(tension_str, 1)
    features["sector_tension"] = cand_tension

    # --- text length features ---
    features["candidate_profile_length"] = len(cand.get("profile_text") or "")
    features["offer_profile_length"] = len(offer.get("profile_text") or "")
    features["profile_length_ratio"] = features["candidate_profile_length"] / max(features["offer_profile_length"], 1)
    features["offer_intitule_length"] = len(offer.get("intitule") or "")

    # --- offer skill domains ---
    features["n_offer_skills_total"] = len(offer_skill_ids)
    features["offer_has_any_skill"] = 1 if offer_skill_ids else 0
    for domain, skill_ids in SKILL_DOMAINS.items():
        domain_skills = set(skill_ids) & offer_skill_ids
        features[f"offer_domain_{domain}_count"] = len(domain_skills)
        features[f"offer_domain_{domain}_has"] = 1 if domain_skills else 0
    features["candidate_gender"] = 1 if (cand.get("genre") or "").lower().startswith("h") else 0

    # --- specialty ---
    cand_spec_family = cand.get("_spec_result", {}).get("id_famille_affiliation", "")
    features["same_specialty_family"] = int(bool(cand_spec_family) and bool(offer_id_fam) and cand_spec_family == offer_id_fam)
    features["candidate_has_specialite"] = int(bool(cand_spec_family))

    # --- contract type ---
    contrat = (offer.get("type_contrat") or "").upper()
    features["offer_type_contrat_cdi"] = int("CDI" in contrat)
    features["offer_type_contrat_cdd"] = int("CDD" in contrat)
    features["offer_type_contrat_stage"] = int("STAGE" in contrat)

    features["candidate_qualification_length"] = len(cand.get("qualification") or "")

    # --- sector graph proximity ---
    cand_sec_proches = secteur_graph.get(cand.get("id_secteur", ""), {}).get("secteurs_proches", [])
    features["secteur_proximity_from_secteur_graph"] = int(offer_id_sect in cand_sec_proches)

    # --- specialty education gap ---
    spec_graph_data = speciality_graph.get("graph", {})
    spec_entry = spec_graph_data.get(cand_spec_family, {})
    typical_level = spec_entry.get("niveau_etude_typique", "")
    features["family_education_gap"] = get_education_rank(cand.get("code_niveau_etude")) - get_education_rank(typical_level)

    features["offer_n_sectors_proches"] = len(secteur_graph.get(offer_id_sect, {}).get("secteurs_proches", []))
    features["cand_n_sectors_proches"] = len(cand_sec_proches)
    features["offer_n_competences_hors_ref"] = len(spec_entry.get("competences_inferred_hors_referentiel", []))

    # --- text similarity ---
    cand_metier_tokens = tokenize_french(cand.get("metier_vise") or "")
    offer_intitule_tokens = tokenize_french(offer.get("intitule") or "")
    features["metier_intitule_jaccard"] = jaccard(cand_metier_tokens, offer_intitule_tokens)
    features["metier_intitule_contains"] = int(bool(cand_metier_tokens and offer_intitule_tokens and cand_metier_tokens.issubset(offer_intitule_tokens)))
    features["cand_metier_vise_len"] = len(cand.get("metier_vise") or "")

    cand_sec_tokens = tokenize_french(cand.get("secteur_demande") or "")
    offer_sec_tokens = tokenize_french(offer.get("secteur") or "")
    features["secteur_demande_jaccard"] = jaccard(cand_sec_tokens, offer_sec_tokens)

    features["same_id_sous_famille"] = int(bool(cand.get("id_sous_famille")) and bool(offer.get("id_sous_famille")) and cand["id_sous_famille"] == offer["id_sous_famille"])
    features["candidate_has_sous_famille"] = int(bool(cand.get("id_sous_famille")))
    features["offer_has_sous_famille"] = int(bool(offer.get("id_sous_famille")))

    return features

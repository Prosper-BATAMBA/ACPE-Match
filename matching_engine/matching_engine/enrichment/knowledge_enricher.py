"""
knowledge_enricher.py

Le Chercheur / Le Cerveau.

Ce module SAIT où trouver les relations implicites entre les concepts déjà
normalisés (métier, spécialité, secteur), mais ne les rédige jamais sous
forme de texte — c'est le rôle du TextEnricher, en aval.

Sources consommées :
    - job_knowledge_graph.json      (trois niveaux : "metiers" précis + "sous_familles"
                                      intermédiaire + "familles" en repli ultime — v3)
    - speciality_knowledge_graph.json (plat, clé = id_famille)
    - secteur_knowledge_graph.json    (plat, clé = id_secteur)

Logique de résolution pour le métier (décrite dans job_knowledge_graph.json
lui-même, v3) :
    1. On cherche une correspondance précise dans "metiers" en comparant le
       "metier_canonique" renvoyé par JobNormalizer au "libelle_source" de
       chaque entrée (même normalisation que les Normalizers : minuscule,
       sans accent, sans ponctuation).
    2. Si rien de précis n'est trouvé, on regarde si le JobNormalizer a résolu
       un "id_sous_famille" (métier reconnu dans une sous-famille affinée,
       ex: Data Analyst -> sous-famille Data Science) : réponse spécifique,
       mais sans niveau_etude_min ni metiers_proches (absents à ce niveau de
       granularité, seules les fiches "metiers" les renseignent).
    3. Sinon, repli sur "familles" via l'id_famille déjà résolu par
       JobNormalizer — réponse générique, mais jamais vide tant que
       l'id_famille existe dans le graphe.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from typing import Any, Dict, List, Optional

from ..config import (
    JOB_KNOWLEDGE_GRAPH_PATH,
    SPECIALITY_KNOWLEDGE_GRAPH_PATH,
    SECTEUR_KNOWLEDGE_GRAPH_PATH,
)


class KnowledgeEnricher:

    def __init__(
        self,
        job_graph_path: Optional[str] = None,
        speciality_graph_path: Optional[str] = None,
        secteur_graph_path: Optional[str] = None,
    ):
        job_graph_path = job_graph_path or str(JOB_KNOWLEDGE_GRAPH_PATH)
        speciality_graph_path = speciality_graph_path or str(SPECIALITY_KNOWLEDGE_GRAPH_PATH)
        secteur_graph_path = secteur_graph_path or str(SECTEUR_KNOWLEDGE_GRAPH_PATH)

        self._job_metiers: Dict[str, Any] = {}
        self._job_sous_familles: Dict[str, Any] = {}
        self._job_familles: Dict[str, Any] = {}
        self._job_libelle_index: Dict[str, str] = {}  # libelle_source nettoyé -> JOB_id

        self._load_job_graph(job_graph_path)
        self.speciality_graph = self._load_flat_graph(speciality_graph_path)
        self.secteur_graph = self._load_flat_graph(secteur_graph_path)

    # ---------- chargement ----------

    @staticmethod
    def _clean_text(text: str) -> str:
        """Même normalisation que les Normalizers, pour que 'metier_canonique'
        et 'libelle_source' se comparent sur un pied d'égalité."""
        if not text or not isinstance(text, str):
            return ""
        text = text.lower()
        text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _load_job_graph(self, filepath: str) -> None:
        if not filepath or not os.path.exists(filepath):
            return
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._job_metiers = data.get("metiers", {})
        self._job_sous_familles = data.get("sous_familles", {})
        self._job_familles = data.get("familles", {})

        for job_id, entry in self._job_metiers.items():
            libelle = entry.get("libelle_source", "")
            clean_libelle = self._clean_text(libelle)
            if clean_libelle and clean_libelle not in self._job_libelle_index:
                self._job_libelle_index[clean_libelle] = job_id

    @staticmethod
    def _load_flat_graph(filepath: str) -> Dict[str, Any]:
        if not filepath or not os.path.exists(filepath):
            return {}
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("graph", {})

    # ---------- job : résolution précise -> sous-famille -> repli famille ----------

    def _resolve_job_id(self, metier_canonique: str) -> Optional[str]:
        clean_input = self._clean_text(metier_canonique)
        return self._job_libelle_index.get(clean_input)

    def enrich_job(self, job_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        job_result : dictionnaire déjà renvoyé par JobNormalizer.normalize()
        (doit contenir au minimum "metier_canonique" et idéalement "id_famille",
        et depuis la mise à jour du JobNormalizer, "id_sous_famille" quand le
        métier a pu être affiné au-delà de la famille générique).

        Renvoie un contexte brut, sans mise en forme :
            {
              "niveau_precision": "metier" | "sous_famille" | "famille" | "aucune",
              "competences_inferred": [...],
              "niveau_etude_min": "NV_x_...",  # None au niveau "sous_famille"
              "metiers_proches": [...],   # vide au niveau "sous_famille"
              "famille": "..."
            }
        """
        fallback = {
            "niveau_precision": "aucune",
            "competences_inferred": [],
            "niveau_etude_min": None,
            "metiers_proches": [],
            "famille": None,
        }

        if not job_result:
            return fallback

        metier_canonique = job_result.get("metier_canonique", "")
        id_famille = job_result.get("id_famille")
        id_sous_famille = job_result.get("id_sous_famille")

        # 1. Tentative de résolution précise (métier)
        job_id = self._resolve_job_id(metier_canonique)
        if job_id and job_id in self._job_metiers:
            entry = self._job_metiers[job_id]
            metiers_proches_ids = entry.get("metiers_proches", []) or []
            metiers_proches_libelles = [
                self._job_metiers[pid]["libelle_source"]
                for pid in metiers_proches_ids
                if pid in self._job_metiers
            ]
            return {
                "niveau_precision": "metier",
                "competences_inferred": entry.get("competences_inferred", []) or [],
                "niveau_etude_min": entry.get("niveau_etude_min"),
                "metiers_proches": metiers_proches_libelles,
                "famille": entry.get("famille"),
            }

        # 2. Repli sur la sous-famille (plus précis qu'une famille entière,
        # mais pas de niveau_etude_min / metiers_proches à ce niveau de granularité)
        if id_sous_famille and id_sous_famille in self._job_sous_familles:
            entry = self._job_sous_familles[id_sous_famille]
            return {
                "niveau_precision": "sous_famille",
                "competences_inferred": entry.get("competences_inferred", []) or [],
                "niveau_etude_min": None,
                "metiers_proches": [],
                "famille": entry.get("famille"),
            }

        # 3. Repli ultime sur la famille
        if id_famille and id_famille in self._job_familles:
            entry = self._job_familles[id_famille]
            return {
                "niveau_precision": "famille",
                "competences_inferred": entry.get("competences_inferred", []) or [],
                "niveau_etude_min": entry.get("niveau_etude_min"),
                # Au niveau famille, "metiers_proches" pointe vers d'autres id_famille
                # (pas des libellés de métier) : on renvoie les libellés de famille lisibles.
                "metiers_proches": [
                    self._job_familles[fid]["famille"]
                    for fid in entry.get("metiers_proches", []) or []
                    if fid in self._job_familles
                ],
                "famille": entry.get("famille"),
            }

        return fallback

    # ---------- spécialité et secteur : lookup plat ----------

    def enrich_famille_specialite(self, id_famille: str) -> Dict[str, Any]:
        """Contexte brut du graphe de spécialité pour un id_famille donné."""
        return self.speciality_graph.get(id_famille, {})

    def enrich_secteur(self, id_secteur: str) -> Dict[str, Any]:
        """Contexte brut du graphe sectoriel pour un id_secteur donné."""
        return self.secteur_graph.get(id_secteur, {})


# =============================================================================
# === EXEMPLE D'UTILISATION ===
# =============================================================================
if __name__ == "__main__":

    # Les chemins par défaut (config.py) fonctionnent maintenant quel que soit
    # le répertoire depuis lequel ce script est lancé.
    enricher = KnowledgeEnricher()

    # Cas 1 : métier identifié précisément (correspond à un libelle_source du graphe)
    job_result_precis = {"metier_canonique": "Agent De Transit", "id_famille": "FAM_TRANS_LOG"}
    print("=== Résolution précise ===")
    print(json.dumps(enricher.enrich_job(job_result_precis), ensure_ascii=False, indent=2))

    # Cas 2 : métier reconnu dans une sous-famille affinée (pas de fiche précise,
    # mais le JobNormalizer a résolu id_sous_famille) -> réponse spécifique
    job_result_sous_famille = {
        "metier_canonique": "Data Analyst", "id_famille": "FAM_IT_DATA",
        "id_sous_famille": "FAM_IT_DATA_SCIENCE",
    }
    print("\n=== Repli sur la sous-famille ===")
    print(json.dumps(enricher.enrich_job(job_result_sous_famille), ensure_ascii=False, indent=2))

    # Cas 3 : métier totalement inconnu (ni fiche précise, ni sous-famille résolue)
    # -> repli sur la famille générique
    job_result_repli = {"metier_canonique": "Coursier En Moto", "id_famille": "FAM_TRANS_LOG"}
    print("\n=== Repli sur la famille ===")
    print(json.dumps(enricher.enrich_job(job_result_repli), ensure_ascii=False, indent=2))

    # Cas 3 : spécialité et secteur
    print("\n=== Spécialité ===")
    print(json.dumps(enricher.enrich_famille_specialite("FAM_TRANS_LOG"), ensure_ascii=False, indent=2))

    print("\n=== Secteur ===")
    print(json.dumps(enricher.enrich_secteur("SEC_TRANS_LOG"), ensure_ascii=False, indent=2))

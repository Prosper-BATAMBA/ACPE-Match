"""
text_enricher.py

Le Rédacteur / Chef d'Orchestre du pipeline de normalisation.

Ce module NE NORMALISE RIEN lui-même. Il reçoit :
    1. Les résultats déjà produits par les Normalizers
       (JobNormalizer, SectorNormalizer, EducationNormalizer,
        SpecialtyNormalizer, SkillNormalizer, LocationNormalizer),
    2. Le contexte d'enrichissement du KnowledgeEnricher (graphes de
       connaissances speciality/secteur — voir remarque plus bas pour le job),
    3. Le texte libre non structuré du CV (expérience professionnelle, etc.)

... et assemble le tout en une seule chaîne de caractères propre, déterministe
et structurée par sections, prête à être vectorisée par un Sentence Transformer
ou envoyée à un LLM pour un embedding de qualité.

Le KnowledgeEnricher (job à deux niveaux, spécialité, secteur) vit désormais
dans son propre module : knowledge_enricher.py. Ce fichier se contente de
l'importer et de consommer ses sorties brutes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .knowledge_enricher import KnowledgeEnricher


# ---------------------------------------------------------------------------
# TextEnricher
# ---------------------------------------------------------------------------
class TextEnricher:

    def __init__(self, knowledge_enricher: Optional[KnowledgeEnricher] = None):
        """
        knowledge_enricher : instance optionnelle. Si None, la section
        "compétences probables (inférées)" liée à la spécialité/secteur est
        simplement omise — le texte reste valide, juste moins riche.
        """
        self.knowledge_enricher = knowledge_enricher

    # ---------- petites aides internes ----------

    @staticmethod
    def _dedup_keep_order(items: List[str]) -> List[str]:
        seen = set()
        result = []
        for item in items:
            if not item:
                continue
            key = item.strip().lower()
            if key and key not in seen:
                seen.add(key)
                result.append(item.strip())
        return result

    @staticmethod
    def _section(title: str, lines: List[str]) -> str:
        """Ne formate la section que si elle a du contenu (évite les blocs vides)."""
        content = [l for l in lines if l and l.strip()]
        if not content:
            return ""
        body = "\n".join(f"- {l}" for l in content)
        return f"{title} :\n{body}"

    # ---------- sous-blocs ----------

    def _build_identite_block(
        self,
        job_result: Optional[Dict[str, Any]],
        secteur_result: Optional[Dict[str, Any]],
        localisation_result: Optional[Dict[str, Any]],
    ) -> str:
        lignes = []
        if job_result and job_result.get("metier_canonique") not in (None, "", "Non spécifié"):
            lignes.append(f"Métier : {job_result['metier_canonique']}")
        if job_result and job_result.get("famille_fr"):
            lignes.append(f"Famille de métier : {job_result['famille_fr']}")
        if secteur_result and secteur_result.get("secteur_canonique"):
            lignes.append(f"Secteur d'activité : {secteur_result['secteur_canonique']}")
        if localisation_result and localisation_result.get("nom_departement"):
            lignes.append(f"Localisation : {localisation_result['nom_departement']}")
        return "\n".join(lignes)

    def _build_formation_block(
        self,
        education_result: Optional[Dict[str, Any]],
        specialty_result: Optional[Dict[str, Any]],
    ) -> str:
        lignes = []
        if specialty_result and specialty_result.get("specialite_brute_nettoyee"):
            lignes.append(f"Spécialité / Filière : {specialty_result['specialite_brute_nettoyee']}")
        if specialty_result and specialty_result.get("famille_affiliation"):
            lignes.append(f"Domaine de formation : {specialty_result['famille_affiliation']}")
        if education_result and education_result.get("libelle_standard"):
            lignes.append(f"Niveau d'étude : {education_result['libelle_standard']}")
        return "\n".join(lignes)

    def _build_competences_declarees_block(self, skill_results: List[Dict[str, Any]]) -> str:
        if not skill_results:
            return ""
        hard = [s["libelle_canonique"] for s in skill_results if s.get("type") == "Hard Skill"]
        soft = [s["libelle_canonique"] for s in skill_results if s.get("type") == "Soft Skill"]

        blocs = []
        hard_section = self._section("Compétences techniques déclarées", self._dedup_keep_order(hard))
        soft_section = self._section("Savoir-être déclarés", self._dedup_keep_order(soft))
        for section in (hard_section, soft_section):
            if section:
                blocs.append(section)
        return "\n\n".join(blocs)

    def _collect_inferred_competences(
        self,
        job_result: Optional[Dict[str, Any]],
        specialty_result: Optional[Dict[str, Any]],
        secteur_result: Optional[Dict[str, Any]],
        already_declared: List[str],
    ) -> List[str]:
        """
        Regroupe les compétences implicites issues de trois sources :
          1. job_result["competences_cles_fr"] — déjà natif du JobNormalizer
             (embarqué dans job_mapping.json), ne nécessite aucun graphe.
          2. KnowledgeEnricher.enrich_job() — résolution précise (métier exact
             trouvé dans job_knowledge_graph.json) ou repli sur la famille.
          3. Le graphe de spécialité et le graphe sectoriel (KnowledgeEnricher),
             si un id_famille_affiliation / id_secteur a été résolu en amont.
        Puis retire tout ce qui est déjà explicitement déclaré par le candidat,
        pour ne jamais répéter une compétence déjà affirmée.
        """
        inferred: List[str] = []

        if job_result:
            inferred.extend(job_result.get("competences_cles_fr", []) or [])

        if self.knowledge_enricher is not None:
            if job_result:
                job_context = self.knowledge_enricher.enrich_job(job_result)
                inferred.extend(job_context.get("competences_inferred", []) or [])

            if specialty_result and specialty_result.get("id_famille_affiliation"):
                fam_data = self.knowledge_enricher.enrich_famille_specialite(
                    specialty_result["id_famille_affiliation"]
                )
                inferred.extend(fam_data.get("competences_inferred_hors_referentiel", []) or [])

            if secteur_result and secteur_result.get("id_secteur"):
                sect_data = self.knowledge_enricher.enrich_secteur(secteur_result["id_secteur"])
                inferred.extend(sect_data.get("competences_dominantes_hors_referentiel", []) or [])

        declared_lower = {d.strip().lower() for d in already_declared}
        return [
            c for c in self._dedup_keep_order(inferred)
            if c.strip().lower() not in declared_lower
        ]

    # ---------- point d'entrée principal ----------

    def build_profile_text(
        self,
        job_result: Optional[Dict[str, Any]] = None,
        secteur_result: Optional[Dict[str, Any]] = None,
        education_result: Optional[Dict[str, Any]] = None,
        specialty_result: Optional[Dict[str, Any]] = None,
        skill_results: Optional[List[Dict[str, Any]]] = None,
        localisation_result: Optional[Dict[str, Any]] = None,
        experience_libre: Optional[str] = None,
    ) -> str:
        """
        Assemble le texte final destiné à l'embedding.

        Tous les arguments sont les dictionnaires DÉJÀ retournés par les
        Normalizers correspondants — ce module ne recalcule jamais une
        normalisation, il ne fait qu'orchestrer et rédiger.

        Retourne une chaîne vide si aucune donnée exploitable n'est fournie.
        """
        skill_results = skill_results or []
        blocs: List[str] = []

        # 1. Identité professionnelle
        identite = self._build_identite_block(job_result, secteur_result, localisation_result)
        if identite:
            blocs.append(identite)

        # 2. Formation
        formation = self._build_formation_block(education_result, specialty_result)
        if formation:
            blocs.append(formation)

        # 3. Compétences déclarées (celles réellement affirmées par le candidat)
        declared_labels = [s["libelle_canonique"] for s in skill_results if s.get("libelle_canonique")]
        competences_declarees = self._build_competences_declarees_block(skill_results)
        if competences_declarees:
            blocs.append(competences_declarees)

        # 4. Compétences probables (inférées, jamais déjà déclarées explicitement)
        inferred = self._collect_inferred_competences(
            job_result, specialty_result, secteur_result, declared_labels
        )
        inferred_section = self._section(
            "Compétences probables (déduites du profil, non déclarées explicitement)",
            inferred,
        )
        if inferred_section:
            blocs.append(inferred_section)

        # 5. Texte libre du CV (expérience, parcours) — non normalisé, tel quel
        if experience_libre and experience_libre.strip():
            blocs.append(f"Expérience et parcours :\n{experience_libre.strip()}")

        return "\n\n".join(blocs).strip()


# =============================================================================
# === EXEMPLE D'UTILISATION (pipeline complet) ===
# =============================================================================
if __name__ == "__main__":

    # --- Résultats simulés des Normalizers (normalement produits en amont) ---

    job_result = {
        "metier_canonique": "Agent De Transit",
        "id_famille": "FAM_TRANS_LOG",
        "famille_fr": "Transport, Logistique & Supply Chain",
        "famille_en": "Transport, Logistics & Supply Chain",
        "competences_cles_fr": [
            "Gestion des stocks",
            "Planification de tournées",
            "Réglementation douanière",
            "Gestion de flotte",
            "Transit",
        ],
        "competences_cles_en": [],
    }

    secteur_result = {
        "id_secteur": "SEC_TRANS_LOG",
        "secteur_canonique": "Transport, Logistique & Supply Chain",
    }

    education_result = {
        "code_niveau": "NV_4_BAC",
        "libelle_standard": "Baccalauréat (Général, Technique ou Professionnel)",
        "rang_ordinal": 4,
    }

    specialty_result = {
        "specialite_brute_nettoyee": "Genie Logistique",
        "id_famille_affiliation": "FAM_TRANS_LOG",
        "famille_affiliation": "Transport, Logistique & Supply Chain",
    }

    skill_results = [
        {"id_skill": "SKILL_ADMIN_GESTION", "libelle_canonique": "Gestion Administrative",
         "categorie": "Bureautique & Administration", "type": "Hard Skill"},
        {"id_skill": "SKILL_SOFT_RELATIONNEL", "libelle_canonique": "Aisance Relationnelle",
         "categorie": "Compétences Transversales (Soft Skills)", "type": "Soft Skill"},
    ]

    localisation_result = {
        "code_departement": "PNR",
        "nom_departement": "Pointe-Noire",
        "localite_brute_nettoyee": "Pointe Noire",
    }

    experience_libre = (
        "3 ans d'expérience en tant qu'assistant transit au sein d'une compagnie "
        "maritime à Pointe-Noire. Suivi des dossiers de dédouanement et coordination "
        "avec les transporteurs locaux."
    )

    # --- KnowledgeEnricher : pointe vers vos graphes réels une fois déployés ---
    # (ici les chemins n'existent pas dans cet environnement de démo, donc les
    #  graphes seront simplement vides — le texte reste valide sans section 4 riche)
    # Les chemins par défaut (config.py) fonctionnent maintenant quel que soit
    # le répertoire depuis lequel ce script est lancé.
    enricher = KnowledgeEnricher()

    text_enricher = TextEnricher(knowledge_enricher=enricher)

    final_text = text_enricher.build_profile_text(
        job_result=job_result,
        secteur_result=secteur_result,
        education_result=education_result,
        specialty_result=specialty_result,
        skill_results=skill_results,
        localisation_result=localisation_result,
        experience_libre=experience_libre,
    )

    print(final_text)

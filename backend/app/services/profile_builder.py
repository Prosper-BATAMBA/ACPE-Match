from __future__ import annotations

from typing import Any, Dict, List, Optional

from matching_engine import (
    JobNormalizer,
    SectorNormalizer,
    EducationNormalizer,
    SpecialtyNormalizer,
    SkillNormalizer,
    LocationNormalizer,
    KnowledgeEnricher,
    TextEnricher,
)


class ProfileBuilder:
    """Transforme des données brutes (candidat ou offre) en texte structuré prêt pour l'embedding."""

    def __init__(self):
        self.job_normalizer = JobNormalizer()
        self.sector_normalizer = SectorNormalizer()
        self.education_normalizer = EducationNormalizer()
        self.specialty_normalizer = SpecialtyNormalizer()
        self.skill_normalizer = SkillNormalizer()
        self.location_normalizer = LocationNormalizer()
        self.knowledge_enricher = KnowledgeEnricher()
        self.text_enricher = TextEnricher(knowledge_enricher=self.knowledge_enricher)

    def build_candidate_profile(
        self,
        metier_vise: Optional[str] = None,
        secteur_demande: Optional[str] = None,
        etudes: Optional[str] = None,
        specialite: Optional[str] = None,
        competences_brutes: Optional[str] = None,
        lieu: Optional[str] = None,
        experience_libre: Optional[str] = None,
    ) -> Dict[str, Any]:
        job_result = self.job_normalizer.normalize(metier_vise) if metier_vise else None
        secteur_result = self.sector_normalizer.normalize(secteur_demande) if secteur_demande else None
        education_result = self.education_normalizer.normalize(etudes) if etudes else None
        specialty_result = self.specialty_normalizer.normalize(specialite) if specialite else None

        skill_results: List[Dict[str, Any]] = []
        if competences_brutes:
            raw_skills = [s.strip() for s in competences_brutes.split(",") if s.strip()]
            skill_results = self.skill_normalizer.normalize_list(raw_skills)

        localisation_result = self.location_normalizer.normalize(lieu) if lieu else None

        profile_text = self.text_enricher.build_profile_text(
            job_result=job_result,
            secteur_result=secteur_result,
            education_result=education_result,
            specialty_result=specialty_result,
            skill_results=skill_results,
            localisation_result=localisation_result,
            experience_libre=experience_libre,
        )

        return {
            "job_result": job_result,
            "secteur_result": secteur_result,
            "education_result": education_result,
            "specialty_result": specialty_result,
            "skill_results": skill_results,
            "localisation_result": localisation_result,
            "profile_text": profile_text,
        }

    def build_job_offer_profile(
        self,
        intitule: Optional[str] = None,
        secteur: Optional[str] = None,
        competences_recherchees: Optional[str] = None,
        localisation: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        job_result = self.job_normalizer.normalize(intitule) if intitule else None
        secteur_result = self.sector_normalizer.normalize(secteur) if secteur else None
        localisation_result = self.location_normalizer.normalize(localisation) if localisation else None

        skill_results: List[Dict[str, Any]] = []
        if competences_recherchees:
            raw_skills = [s.strip() for s in competences_recherchees.split(",") if s.strip()]
            skill_results = self.skill_normalizer.normalize_list(raw_skills)

        profile_text = self.text_enricher.build_profile_text(
            job_result=job_result,
            secteur_result=secteur_result,
            skill_results=skill_results,
            localisation_result=localisation_result,
            experience_libre=description,
        )

        return {
            "job_result": job_result,
            "secteur_result": secteur_result,
            "skill_results": skill_results,
            "localisation_result": localisation_result,
            "profile_text": profile_text,
        }

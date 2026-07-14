"""
matching_engine

Pipeline de normalisation et d'enrichissement de profils (candidats/offres)
pour l'ACPE : texte brut -> ID canonique -> contexte enrichi -> texte prêt
pour l'embedding.

Usage typique depuis un autre projet :

    from matching_engine import (
        JobNormalizer, SectorNormalizer, EducationNormalizer,
        SpecialtyNormalizer, SkillNormalizer, LocationNormalizer,
        KnowledgeEnricher, TextEnricher,
    )

    job_normalizer = JobNormalizer()          # chemins de données résolus automatiquement
    knowledge_enricher = KnowledgeEnricher()
    text_enricher = TextEnricher(knowledge_enricher=knowledge_enricher)

    job_result = job_normalizer.normalize("Agent de transit")
    profil_texte = text_enricher.build_profile_text(job_result=job_result, ...)
"""

from .normalizers import (
    JobNormalizer,
    SectorNormalizer,
    EducationNormalizer,
    SpecialtyNormalizer,
    SkillNormalizer,
    LocationNormalizer,
)
from .enrichment import KnowledgeEnricher, TextEnricher

__all__ = [
    "JobNormalizer",
    "SectorNormalizer",
    "EducationNormalizer",
    "SpecialtyNormalizer",
    "SkillNormalizer",
    "LocationNormalizer",
    "KnowledgeEnricher",
    "TextEnricher",
]

__version__ = "1.0.0"

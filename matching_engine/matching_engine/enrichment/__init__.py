"""
Sous-package enrichment : couches 2 et 3 du pipeline.
    - KnowledgeEnricher : lit les graphes de connaissances (ID -> contexte brut)
    - TextEnricher       : assemble tout en texte prêt pour l'embedding
"""

from .knowledge_enricher import KnowledgeEnricher
from .text_enricher import TextEnricher

__all__ = ["KnowledgeEnricher", "TextEnricher"]

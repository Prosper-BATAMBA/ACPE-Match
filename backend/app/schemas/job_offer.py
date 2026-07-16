from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class JobOfferCreate(BaseModel):
    id: str
    intitule: Optional[str] = None
    type_contrat: Optional[str] = None
    entreprise: Optional[str] = None
    secteur: Optional[str] = None
    localisation: Optional[str] = None
    date_publication: Optional[str] = None
    date_cloture: Optional[str] = None
    description: Optional[str] = None
    competences_recherchees: Optional[str] = None


class JobOfferResponse(BaseModel):
    id: str
    intitule: Optional[str] = None
    type_contrat: Optional[str] = None
    entreprise: Optional[str] = None
    secteur: Optional[str] = None
    localisation: Optional[str] = None
    date_publication: Optional[str] = None
    date_cloture: Optional[str] = None
    description: Optional[str] = None
    competences_recherchees: Optional[str] = None
    id_famille: Optional[str] = None
    id_sous_famille: Optional[str] = None
    id_secteur: Optional[str] = None
    code_departement: Optional[str] = None
    profile_text: Optional[str] = None

    model_config = {"from_attributes": True}

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel


class CandidateCreate(BaseModel):
    id: str
    nom: Optional[str] = None
    prenom: Optional[str] = None
    genre: Optional[str] = None
    age: Optional[int] = None
    lieu: Optional[str] = None
    etudes: Optional[str] = None
    qualification: Optional[str] = None
    specialite: Optional[str] = None
    secteur_demande: Optional[str] = None
    metier_vise: Optional[str] = None
    mobilite: Optional[str] = None
    competences_brutes: Optional[str] = None
    experience_libre: Optional[str] = None


class CandidateResponse(BaseModel):
    id: str
    nom: Optional[str] = None
    prenom: Optional[str] = None
    lieu: Optional[str] = None
    metier_vise: Optional[str] = None
    secteur_demande: Optional[str] = None
    profile_text: Optional[str] = None

    model_config = {"from_attributes": True}

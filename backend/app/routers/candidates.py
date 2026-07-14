from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.candidate import Candidate
from ..schemas.candidate import CandidateCreate, CandidateResponse
from ..services.profile_builder import ProfileBuilder
from ..services.embedding_service import encode
from ..chromadb_client import get_candidates_collection

router = APIRouter(prefix="/api/v1/candidates", tags=["candidates"])

_profile_builder: ProfileBuilder | None = None


def get_profile_builder() -> ProfileBuilder:
    global _profile_builder
    if _profile_builder is None:
        _profile_builder = ProfileBuilder()
    return _profile_builder


@router.post("", response_model=CandidateResponse, status_code=201)
def create_candidate(payload: CandidateCreate, db: Session = Depends(get_db)):
    existing = db.query(Candidate).filter(Candidate.id == payload.id).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Candidate {payload.id} already exists")

    pb = get_profile_builder()
    result = pb.build_candidate_profile(
        metier_vise=payload.metier_vise,
        secteur_demande=payload.secteur_demande,
        etudes=payload.etudes,
        specialite=payload.specialite,
        competences_brutes=payload.competences_brutes,
        lieu=payload.lieu,
        experience_libre=payload.experience_libre,
    )

    profile_text = result["profile_text"]
    job_r = result.get("job_result") or {}
    secteur_r = result.get("secteur_result") or {}
    edu_r = result.get("education_result") or {}
    spec_r = result.get("specialty_result") or {}
    loc_r = result.get("localisation_result") or {}

    candidate = Candidate(
        id=payload.id,
        nom=payload.nom,
        prenom=payload.prenom,
        genre=payload.genre,
        age=payload.age,
        lieu=payload.lieu,
        etudes=payload.etudes,
        qualification=payload.qualification,
        specialite=payload.specialite,
        secteur_demande=payload.secteur_demande,
        metier_vise=payload.metier_vise,
        mobilite=payload.mobilite,
        competences_brutes=payload.competences_brutes,
        experience_libre=payload.experience_libre,
        id_famille=job_r.get("id_famille"),
        id_secteur=secteur_r.get("id_secteur"),
        code_niveau_etude=edu_r.get("code_niveau"),
        code_departement=loc_r.get("code_departement"),
        profile_text=profile_text,
    )
    db.add(candidate)
    db.commit()
    db.refresh(candidate)

    if profile_text:
        embedding = encode(profile_text)
        collection = get_candidates_collection()
        collection.upsert(
            ids=[payload.id],
            embeddings=[embedding],
            documents=[profile_text],
            metadatas=[{"id": payload.id}],
        )

    return candidate


@router.get("", response_model=list[CandidateResponse])
def list_candidates(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    return db.query(Candidate).offset(skip).limit(limit).all()


@router.get("/{candidate_id}", response_model=CandidateResponse)
def get_candidate(candidate_id: str, db: Session = Depends(get_db)):
    candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")
    return candidate

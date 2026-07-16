from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import or_
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.job_offer import JobOffer
from ..schemas.job_offer import JobOfferCreate, JobOfferResponse
from ..services.profile_builder import ProfileBuilder
from ..services.embedding_service import encode
from ..chromadb_client import get_offers_collection
from ..stats_cache import invalidate as invalidate_stats

router = APIRouter(prefix="/api/v1/job-offers", tags=["job-offers"])

_profile_builder: ProfileBuilder | None = None


def get_profile_builder() -> ProfileBuilder:
    global _profile_builder
    if _profile_builder is None:
        _profile_builder = ProfileBuilder()
    return _profile_builder


@router.post("", response_model=JobOfferResponse, status_code=201)
def create_job_offer(payload: JobOfferCreate, db: Session = Depends(get_db)):
    existing = db.query(JobOffer).filter(JobOffer.id == payload.id).first()
    if existing:
        raise HTTPException(status_code=409, detail=f"Job offer {payload.id} already exists")

    pb = get_profile_builder()
    result = pb.build_job_offer_profile(
        intitule=payload.intitule,
        secteur=payload.secteur,
        competences_recherchees=payload.competences_recherchees,
        localisation=payload.localisation,
        description=payload.description,
    )

    profile_text = result["profile_text"]
    job_r = result.get("job_result") or {}
    secteur_r = result.get("secteur_result") or {}
    loc_r = result.get("localisation_result") or {}

    offer = JobOffer(
        id=payload.id,
        intitule=payload.intitule,
        type_contrat=payload.type_contrat,
        entreprise=payload.entreprise,
        secteur=payload.secteur,
        localisation=payload.localisation,
        date_publication=payload.date_publication,
        date_cloture=payload.date_cloture,
        description=payload.description,
        competences_recherchees=payload.competences_recherchees,
        id_famille=job_r.get("id_famille"),
        id_sous_famille=job_r.get("id_sous_famille"),
        id_secteur=secteur_r.get("id_secteur"),
        code_departement=loc_r.get("code_departement"),
        profile_text=profile_text,
    )
    db.add(offer)
    db.commit()
    db.refresh(offer)

    if profile_text:
        embedding = encode(profile_text)
        collection = get_offers_collection()
        collection.upsert(
            ids=[payload.id],
            embeddings=[embedding],
            documents=[profile_text],
            metadatas=[{"id": payload.id}],
        )

    invalidate_stats()

    return offer


@router.get("/search")
def search_job_offers(
    q: str = Query("", description="Recherche par référence, intitulé, secteur ou entreprise"),
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    db: Session = Depends(get_db),
):
    query = db.query(JobOffer)
    if q:
        like_q = f"%{q}%"
        query = query.filter(
            or_(
                JobOffer.id.ilike(like_q),
                JobOffer.intitule.ilike(like_q),
                JobOffer.secteur.ilike(like_q),
                JobOffer.entreprise.ilike(like_q),
            )
        )
    total = query.count()
    results = query.offset(skip).limit(limit).all()
    return {"total": total, "results": [JobOfferResponse.model_validate(o) for o in results]}


@router.get("", response_model=list[JobOfferResponse])
def list_job_offers(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db),
):
    return db.query(JobOffer).offset(skip).limit(limit).all()


@router.get("/{offer_id}", response_model=JobOfferResponse)
def get_job_offer(offer_id: str, db: Session = Depends(get_db)):
    offer = db.query(JobOffer).filter(JobOffer.id == offer_id).first()
    if not offer:
        raise HTTPException(status_code=404, detail="Job offer not found")
    return offer

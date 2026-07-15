from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.candidate import Candidate
from ..models.job_offer import JobOffer
from ..chromadb_client import get_offers_collection, get_candidates_collection
from ..services.matching_engine_service import get_recommendations, get_candidates_for_offer
from ..services.embedding_service import encode

router = APIRouter(prefix="/api/v1/matching", tags=["matching"])


@router.get("/candidate/{candidate_id}")
def match_candidate(
    candidate_id: str,
    top_k: Optional[int] = 10,
    db: Session = Depends(get_db),
):
    candidate = db.query(Candidate).filter(Candidate.id == candidate_id).first()
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    offers_collection = get_offers_collection()

    recommendations = get_recommendations(
        db=db,
        chroma_collection=offers_collection,
        candidate_id=candidate_id,
        k=top_k,
        embedding_fn=encode,
    )

    return {
        "candidate_id": candidate_id,
        "candidate_name": f"{candidate.prenom or ''} {candidate.nom or ''}".strip(),
        "total_offers_compared": offers_collection.count(),
        "top_k": top_k,
        "recommendations": recommendations,
    }


@router.get("/offer/{offer_id}")
def match_offer(
    offer_id: str,
    top_k: Optional[int] = 5,
    db: Session = Depends(get_db),
):
    offer = db.query(JobOffer).filter(JobOffer.id == offer_id).first()
    if not offer:
        raise HTTPException(status_code=404, detail="Job offer not found")

    candidates_collection = get_candidates_collection()

    recommendations = get_candidates_for_offer(
        db=db,
        chroma_collection=candidates_collection,
        offer_id=offer_id,
        k=top_k,
        embedding_fn=encode,
    )

    enriched = []
    for rec in recommendations:
        cand = db.query(Candidate).filter(Candidate.id == rec["candidate_id"]).first()
        cand_name = f"{cand.prenom or ''} {cand.nom or ''}".strip() if cand else ""
        metier = cand.metier_vise if cand else ""
        lieu = cand.lieu if cand else ""
        enriched.append({
            "candidate_id": rec["candidate_id"],
            "candidate_name": cand_name,
            "metier_vise": metier,
            "lieu": lieu,
            "score": rec["score"],
            "skill_gap": rec["skill_gap"],
        })

    return {
        "offer_id": offer_id,
        "offer_intitule": offer.intitule,
        "offer_entreprise": offer.entreprise,
        "total_candidates_compared": candidates_collection.count(),
        "top_k": top_k,
        "recommendations": enriched,
    }

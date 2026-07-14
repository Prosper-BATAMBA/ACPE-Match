from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.candidate import Candidate
from ..chromadb_client import get_offers_collection
from ..services.matching_engine_service import get_recommendations
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

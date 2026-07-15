from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..database import get_db
from ..chromadb_client import get_candidates_collection, get_offers_collection
from ..stats_cache import get_stats

router = APIRouter(prefix="/api/v1", tags=["stats"])


@router.get("/stats")
def get_stats_endpoint(db: Session = Depends(get_db)):
    return get_stats(db, get_candidates_collection(), get_offers_collection())

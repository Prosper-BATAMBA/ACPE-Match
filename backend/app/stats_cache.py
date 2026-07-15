from __future__ import annotations

import threading
import time
from datetime import datetime, timezone

from sqlalchemy import func
from sqlalchemy.orm import Session

from .models.candidate import Candidate
from .models.job_offer import JobOffer

_cache: dict = {}
_last_computed: float = 0
_lock = threading.Lock()
_CACHE_TTL = 300  # 5 minutes


def _compute_stats(
    db: Session,
    chroma_candidates_col=None,
    chroma_offers_col=None,
) -> dict:
    total_candidates = db.query(func.count()).select_from(Candidate).scalar() or 0
    total_offers = db.query(func.count()).select_from(JobOffer).scalar() or 0

    encoded_candidates = 0
    encoded_offers = 0
    try:
        if chroma_candidates_col is not None:
            encoded_candidates = chroma_candidates_col.count()
    except Exception:
        pass
    try:
        if chroma_offers_col is not None:
            encoded_offers = chroma_offers_col.count()
    except Exception:
        pass

    cand_by_dept = dict(
        db.query(Candidate.code_departement, func.count())
        .group_by(Candidate.code_departement)
        .order_by(func.count().desc())
        .all()
    )
    off_by_secteur = dict(
        db.query(JobOffer.id_secteur, func.count())
        .group_by(JobOffer.id_secteur)
        .order_by(func.count().desc())
        .all()
    )
    off_by_contrat = dict(
        db.query(JobOffer.type_contrat, func.count())
        .group_by(JobOffer.type_contrat)
        .order_by(func.count().desc())
        .all()
    )
    cand_by_education = dict(
        db.query(Candidate.code_niveau_etude, func.count())
        .group_by(Candidate.code_niveau_etude)
        .order_by(func.count().desc())
        .all()
    )

    top_familles_raw = (
        db.query(JobOffer.id_famille, func.count())
        .filter(JobOffer.id_famille.isnot(None))
        .group_by(JobOffer.id_famille)
        .order_by(func.count().desc())
        .limit(10)
        .all()
    )

    return {
        "total_candidates": total_candidates,
        "total_offers": total_offers,
        "encoded_candidates": encoded_candidates,
        "encoded_offers": encoded_offers,
        "encoding_rate": round(encoded_candidates / max(total_candidates, 1), 3),
        "last_updated": datetime.now(timezone.utc).isoformat(),
        "distributions": {
            "candidates_by_department": cand_by_dept,
            "offers_by_sector": off_by_secteur,
            "offers_by_contract": off_by_contrat,
            "candidates_by_education": cand_by_education,
        },
        "top_familles_offers": [
            {"id": f, "count": c} for f, c in top_familles_raw
        ],
    }


def get_stats(
    db: Session,
    chroma_candidates_col=None,
    chroma_offers_col=None,
    force: bool = False,
) -> dict:
    global _cache, _last_computed
    with _lock:
        if force or not _cache or (time.time() - _last_computed > _CACHE_TTL):
            _cache = _compute_stats(db, chroma_candidates_col, chroma_offers_col)
            _last_computed = time.time()
        return _cache


def invalidate() -> None:
    global _cache
    with _lock:
        _cache = {}

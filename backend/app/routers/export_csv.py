from __future__ import annotations

import csv
import io
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..database import get_db
from ..models.candidate import Candidate
from ..models.job_offer import JobOffer
from ..chromadb_client import get_offers_collection, get_candidates_collection
from ..services.matching_engine_service import get_recommendations, get_candidates_for_offer
from ..services.embedding_service import encode

router = APIRouter(prefix="/api/v1/matching", tags=["export"])

MAX_EXPORT_CANDIDATES = 100
MAX_EXPORT_OFFERS = 100


@router.get("/export-csv-by-offer")
def export_csv_by_offer(
    offer_ids: Optional[str] = Query(
        None,
        description="IDs d'offres séparés par des virgules. Si vide, exporte les 100 premières offres.",
    ),
    top_k: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    chroma_col = get_candidates_collection()

    if offer_ids:
        ids = [i.strip() for i in offer_ids.split(",") if i.strip()]
    else:
        ids = [
            str(r[0])
            for r in db.query(JobOffer.id).limit(MAX_EXPORT_OFFERS).all()
        ]

    if not ids:
        raise HTTPException(status_code=404, detail="Aucune offre trouvée")

    if len(ids) > MAX_EXPORT_OFFERS:
        ids = ids[:MAX_EXPORT_OFFERS]

    rows = []
    for oid in ids:
        offer = db.query(JobOffer).filter(JobOffer.id == oid).first()
        if not offer:
            continue

        recs = get_candidates_for_offer(
            db=db,
            chroma_collection=chroma_col,
            offer_id=oid,
            k=top_k,
            embedding_fn=encode,
        )

        for r in recs:
            cand = db.query(Candidate).filter(Candidate.id == r["candidate_id"]).first()
            rows.append(
                {
                    "offer_id": oid,
                    "offer_intitule": offer.intitule or "",
                    "offer_entreprise": offer.entreprise or "",
                    "candidate_id": r["candidate_id"],
                    "candidate_nom": cand.nom if cand else "",
                    "candidate_prenom": cand.prenom if cand else "",
                    "candidate_metier": cand.metier_vise if cand else "",
                    "score": r["score"],
                    "skills_acquired": " | ".join(r["skill_gap"].get("acquired", [])),
                    "skills_missing": " | ".join(r["skill_gap"].get("missing", [])),
                    "gap_score": r["skill_gap"].get("gap_score", 0),
                }
            )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="Aucune recommandation générée pour les offres sélectionnées",
        )

    fieldnames = list(rows[0].keys())
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    content = output.getvalue()
    output.close()

    return StreamingResponse(
        iter([content]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=acpe_matching_export_by_offer.csv"
        },
    )


@router.get("/export-csv")
def export_csv(
    candidate_ids: Optional[str] = Query(
        None,
        description="IDs séparés par des virgules. Si vide, exporte les 100 premiers candidats encodés.",
    ),
    top_k: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    chroma_col = get_offers_collection()

    if candidate_ids:
        ids = [i.strip() for i in candidate_ids.split(",") if i.strip()]
    else:
        ids = [
            str(r[0])
            for r in db.query(Candidate.id)
            .filter(Candidate.profile_text.isnot(None))
            .limit(MAX_EXPORT_CANDIDATES)
            .all()
        ]

    if not ids:
        raise HTTPException(status_code=404, detail="Aucun candidat trouvé")

    if len(ids) > MAX_EXPORT_CANDIDATES:
        ids = ids[:MAX_EXPORT_CANDIDATES]

    rows = []
    for cid in ids:
        candidate = db.query(Candidate).filter(Candidate.id == cid).first()
        if not candidate:
            continue

        recs = get_recommendations(
            db=db,
            chroma_collection=chroma_col,
            candidate_id=cid,
            k=top_k,
            embedding_fn=encode,
        )

        for r in recs:
            rows.append(
                {
                    "candidate_id": cid,
                    "candidate_nom": candidate.nom or "",
                    "candidate_prenom": candidate.prenom or "",
                    "candidate_metier": candidate.metier_vise or "",
                    "offer_id": r["offer_id"],
                    "intitule": r["intitule"] or "",
                    "entreprise": r["entreprise"] or "",
                    "score": r["score"],
                    "skills_acquired": " | ".join(r["skill_gap"].get("acquired", [])),
                    "skills_missing": " | ".join(r["skill_gap"].get("missing", [])),
                    "gap_score": r["skill_gap"].get("gap_score", 0),
                }
            )

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="Aucune recommandation générée pour les candidats sélectionnés",
        )

    fieldnames = list(rows[0].keys())
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    content = output.getvalue()
    output.close()

    return StreamingResponse(
        iter([content]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=acpe_matching_export.csv"
        },
    )


@router.get("/export-csv-minimal")
def export_csv_minimal(
    candidate_ids: Optional[str] = Query(
        None,
        description="IDs separes par des virgules. Si vide, exporte les 100 premiers candidats encodes.",
    ),
    top_k: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db),
):
    """Export minimal au format requis : candidate_id, rank, job_id, score."""
    chroma_col = get_offers_collection()

    if candidate_ids:
        ids = [i.strip() for i in candidate_ids.split(",") if i.strip()]
    else:
        ids = [
            str(r[0])
            for r in db.query(Candidate.id)
            .filter(Candidate.profile_text.isnot(None))
            .limit(MAX_EXPORT_CANDIDATES)
            .all()
        ]

    if not ids:
        raise HTTPException(status_code=404, detail="Aucun candidat trouve")

    if len(ids) > MAX_EXPORT_CANDIDATES:
        ids = ids[:MAX_EXPORT_CANDIDATES]

    rows = []
    for cid in ids:
        recs = get_recommendations(
            db=db,
            chroma_collection=chroma_col,
            candidate_id=cid,
            k=top_k,
            embedding_fn=encode,
        )

        for rank, r in enumerate(recs, 1):
            rows.append({
                "candidate_id": cid,
                "rank": rank,
                "job_id": r["offer_id"],
                "score": round(r["score"], 4),
            })

    if not rows:
        raise HTTPException(
            status_code=404,
            detail="Aucune recommandation generee",
        )

    fieldnames = ["candidate_id", "rank", "job_id", "score"]
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

    content = output.getvalue()
    output.close()

    return StreamingResponse(
        iter([content]),
        media_type="text/csv",
        headers={
            "Content-Disposition": "attachment; filename=acpe_recommendations.csv"
        },
    )

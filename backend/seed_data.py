"""
seed_data.py

Importe les données depuis les fichiers Excel (Demandeurs, Offres, Extensions)
vers SQLite + ChromaDB via le pipeline matching_engine + sentence-transformers.

Usage :
    cd backend
    python -m seed_data
"""

import sys
import os
import logging
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import init_db, SessionLocal
from app.models.candidate import Candidate
from app.models.job_offer import JobOffer
from app.services.profile_builder import ProfileBuilder
from app.services.embedding_service import encode_batch
from app.chromadb_client import get_candidates_collection, get_offers_collection

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
HACKATON_DIR = os.path.dirname(BACKEND_DIR)
DATA_DIR = os.path.join(
    HACKATON_DIR, "matching_engine", "matching_engine", "data", "raw",
)

DEMANDEURS_PATH = os.path.join(DATA_DIR, "Demandeurs.xlsx")
OFFRES_PATH = os.path.join(DATA_DIR, "Offres_ACPE.xlsx")

BATCH_SIZE = 256


def load_excel(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        logger.warning(f"Fichier introuvable : {path}")
        return pd.DataFrame()
    df = pd.read_excel(path, engine="openpyxl")
    logger.info(f"Chargé {len(df)} lignes depuis {os.path.basename(path)}")
    return df


def safe_str(val) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip()


def safe_int(val):
    if pd.isna(val):
        return None
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return None


def _fmt_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"


def import_candidates(df: pd.DataFrame, pb: ProfileBuilder, chroma_candidates):
    t0 = time.time()
    db = SessionLocal()
    seen_ids = set()
    skipped = 0
    errors = 0
    total_rows = len(df)

    rows_data = []
    for idx, (_, row) in enumerate(df.iterrows(), 1):
        candidate_id = safe_str(row.get("Matricule", ""))
        if not candidate_id or candidate_id in seen_ids:
            skipped += 1
            continue
        if db.query(Candidate).filter(Candidate.id == candidate_id).first():
            seen_ids.add(candidate_id)
            skipped += 1
            continue

        niveau_etude = safe_str(row.get("niveau_etude", "")) or safe_str(row.get("Diplome", ""))
        metier_vise = safe_str(row.get("qualification_metier", "")) or safe_str(row.get("Métier visé / Qualification visée", ""))
        secteur_demande = safe_str(row.get("secteur_metier", "")) or safe_str(row.get("Secteur demandé", ""))
        specialite = safe_str(row.get("Filière / Spécialité", ""))
        mobilite = safe_str(row.get("Mobilité géographique", ""))

        try:
            result = pb.build_candidate_profile(
                metier_vise=metier_vise,
                secteur_demande=secteur_demande,
                etudes=niveau_etude,
                specialite=specialite,
                lieu=mobilite,
            )
            profile_text = result["profile_text"]
            job_r = result.get("job_result") or {}
            secteur_r = result.get("secteur_result") or {}
            edu_r = result.get("education_result") or {}
            loc_r = result.get("localisation_result") or {}

            candidate = Candidate(
                id=candidate_id,
                nom=safe_str(row.get("Nom", "")) or None,
                prenom=safe_str(row.get("Prénom", "")) or None,
                genre=safe_str(row.get("Genre", "")) or None,
                age=safe_int(row.get("Age")),
                lieu=mobilite or None,
                etudes=niveau_etude or None,
                qualification=safe_str(row.get("Qualification", "")) or None,
                specialite=specialite or None,
                secteur_demande=secteur_demande or None,
                metier_vise=metier_vise or None,
                mobilite=mobilite or None,
                id_famille=job_r.get("id_famille"),
                id_secteur=secteur_r.get("id_secteur"),
                code_niveau_etude=edu_r.get("code_niveau"),
                code_departement=loc_r.get("code_departement"),
                profile_text=profile_text,
            )
            db.add(candidate)
            seen_ids.add(candidate_id)
            rows_data.append((candidate_id, profile_text))
        except Exception as e:
            errors += 1
            logger.error(f"Erreur candidat {candidate_id} : {e}")

        if idx % 500 == 0 or idx == total_rows:
            pct = idx / total_rows * 100
            elapsed = time.time() - t0
            speed = idx / elapsed if elapsed > 0 else 0
            eta = (total_rows - idx) / speed if speed > 0 else 0
            logger.info(
                f"  Phase 1/2 candidats: {idx}/{total_rows} ({pct:.1f}%) "
                f"| {len(rows_data)} insérés, {skipped} ignorés, {errors} erreurs "
                f"| {_fmt_eta(eta)} restant"
            )

    db.commit()
    logger.info(f"Phase 1/2 : {len(rows_data)} candidats en base ({skipped} ignorés, {errors} erreurs)")

    t1 = time.time()
    logger.info(f"Phase 2/2 : Embedding + ChromaDB ({len(rows_data)} candidats)...")
    for i in range(0, len(rows_data), BATCH_SIZE):
        batch = rows_data[i:i + BATCH_SIZE]
        ids = [b[0] for b in batch]
        texts = [b[1] if b[1] else " " for b in batch]
        embeddings = encode_batch(texts)
        chroma_candidates.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"id": cid} for cid in ids],
        )
        done = min(i + BATCH_SIZE, len(rows_data))
        pct = done / len(rows_data) * 100 if rows_data else 0
        elapsed2 = time.time() - t1
        speed = done / elapsed2 if elapsed2 > 0 else 0
        eta = (len(rows_data) - done) / speed if speed > 0 else 0
        logger.info(
            f"  Phase 2/2 candidats: {done}/{len(rows_data)} ({pct:.1f}%) "
            f"| {_fmt_eta(eta)} restant"
        )

    elapsed = time.time() - t0
    logger.info(f"Candidats terminés : {len(rows_data)} en {elapsed:.1f}s")
    return len(rows_data)


def import_offers(df: pd.DataFrame, pb: ProfileBuilder, chroma_offers):
    t0 = time.time()
    db = SessionLocal()
    seen_ids = set()
    skipped = 0
    errors = 0
    total_rows = len(df)

    rows_data = []
    for idx, (_, row) in enumerate(df.iterrows(), 1):
        offer_id = safe_str(row.get("Référence offre", ""))
        if not offer_id or offer_id in seen_ids:
            skipped += 1
            continue
        if db.query(JobOffer).filter(JobOffer.id == offer_id).first():
            seen_ids.add(offer_id)
            skipped += 1
            continue

        intitule = safe_str(row.get("Intitule", ""))
        secteur = safe_str(row.get("Secteur activité", ""))
        lieu = safe_str(row.get("Lieu", ""))
        description = safe_str(row.get("Description", ""))
        profil_text = safe_str(row.get("Profil", ""))
        comp_declarees = safe_str(row.get("Compétences", ""))
        competences_inferees = safe_str(row.get("Competences_inferees", ""))
        famille_inferee = safe_str(row.get("Famille_metier_inferee", ""))
        sous_famille_inferee = safe_str(row.get("Sous_famille_inferee", ""))

        competences_recherchees = competences_inferees
        if not competences_recherchees:
            parts = []
            if profil_text:
                parts.append(profil_text)
            if comp_declarees:
                parts.append(comp_declarees)
            competences_recherchees = " | ".join(parts) if parts else ""

        try:
            result = pb.build_job_offer_profile(
                intitule=intitule,
                secteur=secteur,
                competences_recherchees=competences_recherchees,
                localisation=lieu,
                description=description,
            )
            profile_text = result["profile_text"]
            job_r = result.get("job_result") or {}
            secteur_r = result.get("secteur_result") or {}
            loc_r = result.get("localisation_result") or {}

            offer = JobOffer(
                id=offer_id,
                intitule=intitule or None,
                poste=safe_str(row.get("Poste", "")) or None,
                type_contrat=safe_str(row.get("Type contrat", "")) or None,
                type_entreprise=safe_str(row.get("Type d'entreprise", "")) or None,
                entreprise=safe_str(row.get("Entreprise", "")) or None,
                secteur=secteur or None,
                localisation=lieu or None,
                date_publication=safe_str(row.get("Date de publication ", "")) or None,
                description=description or None,
                competences_recherchees=competences_recherchees or None,
                id_famille=famille_inferee or job_r.get("id_famille"),
                id_sous_famille=sous_famille_inferee or job_r.get("id_sous_famille"),
                id_secteur=secteur_r.get("id_secteur"),
                code_departement=loc_r.get("code_departement"),
                profile_text=profile_text,
            )
            db.add(offer)
            seen_ids.add(offer_id)
            rows_data.append((offer_id, profile_text))
        except Exception as e:
            errors += 1
            logger.error(f"Erreur offre {offer_id} : {e}")

        if idx % 100 == 0 or idx == total_rows:
            pct = idx / total_rows * 100
            elapsed = time.time() - t0
            speed = idx / elapsed if elapsed > 0 else 0
            eta = (total_rows - idx) / speed if speed > 0 else 0
            logger.info(
                f"  Phase 1/2 offres: {idx}/{total_rows} ({pct:.1f}%) "
                f"| {len(rows_data)} insérées, {skipped} ignorées, {errors} erreurs "
                f"| {_fmt_eta(eta)} restant"
            )

    db.commit()
    logger.info(f"Phase 1/2 : {len(rows_data)} offres en base ({skipped} ignorés, {errors} erreurs)")

    t1 = time.time()
    logger.info(f"Phase 2/2 : Embedding + ChromaDB ({len(rows_data)} offres)...")
    for i in range(0, len(rows_data), BATCH_SIZE):
        batch = rows_data[i:i + BATCH_SIZE]
        ids = [b[0] for b in batch]
        texts = [b[1] if b[1] else " " for b in batch]
        embeddings = encode_batch(texts)
        chroma_offers.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=[{"id": oid} for oid in ids],
        )
        done = min(i + BATCH_SIZE, len(rows_data))
        pct = done / len(rows_data) * 100 if rows_data else 0
        elapsed2 = time.time() - t1
        speed = done / elapsed2 if elapsed2 > 0 else 0
        eta = (len(rows_data) - done) / speed if speed > 0 else 0
        logger.info(
            f"  Phase 2/2 offres: {done}/{len(rows_data)} ({pct:.1f}%) "
            f"| {_fmt_eta(eta)} restant"
        )

    elapsed = time.time() - t0
    logger.info(f"Offres terminées : {len(rows_data)} en {elapsed:.1f}s")
    return len(rows_data)


def main():
    logger.info("=== Initialisation de la base de données ===")
    init_db()

    logger.info("=== Chargement du ProfileBuilder ===")
    pb = ProfileBuilder()

    logger.info("=== Chargement des données Excel ===")
    demandeurs_df = load_excel(DEMANDEURS_PATH)
    offres_df = load_excel(OFFRES_PATH)

    chroma_candidates = get_candidates_collection()
    chroma_offers = get_offers_collection()

    n_candidates = import_candidates(demandeurs_df, pb, chroma_candidates)
    n_offers = import_offers(offres_df, pb, chroma_offers)

    logger.info("=" * 50)
    logger.info(f"Terminé ! {n_candidates} candidats, {n_offers} offres importés.")
    logger.info(f"ChromaDB candidats : {chroma_candidates.count()} vecteurs")
    logger.info(f"ChromaDB offres : {chroma_offers.count()} vecteurs")


if __name__ == "__main__":
    main()

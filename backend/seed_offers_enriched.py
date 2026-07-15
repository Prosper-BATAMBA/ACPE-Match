"""
seed_offers_enriched.py

Re-import only offers from Offres_enrichi.xlsx.
Clears existing offer data first.
"""

import sys
import os
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import init_db, SessionLocal
from app.models.job_offer import JobOffer
from app.services.profile_builder import ProfileBuilder
from app.services.embedding_service import encode_batch
from app.chromadb_client import get_offers_collection

BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
HACKATON_DIR = os.path.dirname(BACKEND_DIR)
DATA_DIR = os.path.join(HACKATON_DIR, "matching_engine", "matching_engine", "data", "raw")
OFFRES_PATH = os.path.join(DATA_DIR, "Offres_enrichi.xlsx")

BATCH_SIZE = 256


def safe_str(val) -> str:
    if pd.isna(val):
        return ""
    return str(val).strip()


def main():
    print("=== Re-import des offres enrichies ===")

    df = pd.read_excel(OFFRES_PATH, engine="openpyxl")
    print(f"Charge {len(df)} offres depuis {OFFRES_PATH}")

    chroma_offers = get_offers_collection()
    print(f"ChromaDB offres avant: {chroma_offers.count()}")

    init_db()
    db = SessionLocal()

    db.query(JobOffer).delete()
    db.commit()
    print("SQLite job_offers vide")

    pb = ProfileBuilder()
    t0 = time.time()

    rows_data = []
    errors = 0
    seen_ids = set()
    for idx, row in df.iterrows():
        offer_id = safe_str(row.get("Référence offre", ""))
        if not offer_id or offer_id in seen_ids:
            continue
        seen_ids.add(offer_id)

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
                type_contrat=safe_str(row.get("Type contrat", "")) or None,
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
            rows_data.append((offer_id, profile_text))
        except Exception as e:
            errors += 1
            if errors <= 5:
                print(f"  ERREUR offre {offer_id}: {e}")

        if (idx + 1) % 500 == 0:
            print(f"  ... {idx+1}/{len(df)}")

    db.commit()
    print(f"Phase 1: {len(rows_data)} offres en base ({errors} erreurs)")

    print(f"Phase 2: Embedding + ChromaDB...")
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
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"  ... {i + len(batch)}/{len(rows_data)} upsertes")

    elapsed = time.time() - t0
    print(f"Termine: {len(rows_data)} offres en {elapsed:.1f}s")
    print(f"ChromaDB offres: {chroma_offers.count()}")


if __name__ == "__main__":
    main()

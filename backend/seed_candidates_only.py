"""
seed_candidates_only.py

Encode ONLY candidates with bge-m3 into ChromaDB.
Optimized: batch_size=64, checkpoint resume.

Usage: cd backend && python seed_candidates_only.py
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.database import SessionLocal
from app.models.candidate import Candidate
from app.services.embedding_service import encode_batch
from app.chromadb_client import get_candidates_collection

BATCH_SIZE = 64
CHECKPOINT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "seed_checkpoint_candidates.json")


def main():
    db = SessionLocal()
    col = get_candidates_collection()

    total = db.query(Candidate).count()
    existing = col.count()
    print(f"Candidates in DB: {total}, in ChromaDB: {existing}")

    remaining = total - existing
    if remaining <= 0:
        print("All candidates already embedded.")
        db.close()
        return

    processed_ids = set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            processed_ids = set(json.load(f))
        print(f"Checkpoint found: {len(processed_ids)} already processed")

    print(f"\nEncoding {remaining} candidates with bge-m3 (batch={BATCH_SIZE})...")
    t0 = time.time()
    query = db.query(Candidate).filter(Candidate.profile_text.isnot(None))

    ids, texts = [], []
    count = 0
    skipped = 0
    for c in query.yield_per(1000):
        if c.id in processed_ids:
            skipped += 1
            continue
        ids.append(c.id)
        texts.append(c.profile_text or " ")
        count += 1
        if len(ids) >= BATCH_SIZE:
            embeddings = encode_batch(texts)
            col.upsert(
                ids=ids, embeddings=embeddings, documents=texts,
                metadatas=[{"id": cid} for cid in ids],
            )
            processed_ids.update(ids)
            ids, texts = [], []
            if count % 500 == 0:
                elapsed = time.time() - t0
                print(f"  ... {count}/{remaining} ({elapsed:.1f}s)")
                with open(CHECKPOINT_FILE, "w") as f:
                    json.dump(list(processed_ids), f)

    if ids:
        embeddings = encode_batch(texts)
        col.upsert(
            ids=ids, embeddings=embeddings, documents=texts,
            metadatas=[{"id": cid} for cid in ids],
        )
        processed_ids.update(ids)

    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(list(processed_ids), f)

    elapsed = time.time() - t0
    print(f"\nDone: {col.count()} candidates in ChromaDB ({elapsed:.1f}s, {skipped} skipped)")
    db.close()


if __name__ == "__main__":
    main()

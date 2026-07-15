"""
evaluate_matching.py

Evalue la qualite du matching en comparant les resultats de l'API
avec le fichier d'appariement (ground truth).

Usage:
    cd backend
    python evaluate_matching.py --sample 500
    python evaluate_matching.py --full
"""

import sys
import os
import time
import json
import math
import urllib.request
import urllib.error
import argparse
import random

import pandas as pd
import chromadb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = "http://localhost:8000"
GT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "matching_engine", "matching_engine", "data", "raw",
    "Appariement_Demandeurs_Offres.xlsx"
)


def api_get(path):
    url = BASE + path
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8")), resp.status
    except urllib.error.HTTPError as e:
        return None, e.code
    except Exception as e:
        return None, str(e)


def load_ground_truth():
    df = pd.read_excel(GT_PATH, engine="openpyxl")
    df = df.drop_duplicates(subset="id_demandeur", keep="first")
    gt = {}
    for _, row in df.iterrows():
        cid = str(row["id_demandeur"]).strip()
        offers = [
            str(row["id_offre1"]).strip(),
            str(row["id_offre2"]).strip(),
            str(row["id_offre3"]).strip(),
        ]
        gt[cid] = offers
    return gt


def precision_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    hits = sum(1 for r in rec_k if r in relevant)
    return hits / k


def recall_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    hits = sum(1 for r in rec_k if r in relevant)
    return hits / max(len(relevant), 1)


def hit_rate_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    return 1.0 if any(r in relevant for r in rec_k) else 0.0


def ndcg_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    dcg = 0.0
    for i, r in enumerate(rec_k):
        if r in relevant:
            dcg += 1.0 / math.log2(i + 2)
    ideal_dcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / max(ideal_dcg, 1e-10)


def main():
    parser = argparse.ArgumentParser(description="Evaluate matching quality")
    parser.add_argument("--sample", type=int, default=0, help="Number of candidates to evaluate (0=all)")
    parser.add_argument("--top-k", type=int, default=10, help="Max recommendations per candidate")
    args = parser.parse_args()

    print("=" * 60)
    print("  EVALUATION DU MATCHING ACPE MATCH")
    print("=" * 60)

    print("\n[1/4] Chargement du ground truth...")
    gt = load_ground_truth()
    print(f"  {len(gt)} candidats uniques dans le ground truth")

    print("\n[2/4] Verification de la disponibilite ChromaDB...")
    chroma = chromadb.PersistentClient(path="chroma_data")
    col_cand = chroma.get_collection("candidate_embeddings")
    chroma_ids = set(col_cand.get()["ids"])
    eval_candidates = [cid for cid in gt if cid in chroma_ids]
    print(f"  {len(eval_candidates)}/{len(gt)} candidats avec embedding ChromaDB")

    if args.sample > 0:
        random.seed(42)
        eval_candidates = random.sample(eval_candidates, min(args.sample, len(eval_candidates)))
        print(f"  Echantillon: {len(eval_candidates)} candidats")

    print(f"\n[3/4] Evaluation sur {len(eval_candidates)} candidats (top_k={args.top_k})...")
    print(f"  Temps estime: ~{len(eval_candidates) * 2.2 / 60:.0f} min")

    metrics = {
        "precision": {1: [], 3: [], 5: [], 10: []},
        "recall": {1: [], 3: [], 5: [], 10: []},
        "hit_rate": {1: [], 3: [], 5: [], 10: []},
        "ndcg": {1: [], 3: [], 5: [], 10: []},
    }

    good_examples = []
    bad_examples = []
    errors = 0
    t0 = time.time()

    for idx, cid in enumerate(eval_candidates):
        if (idx + 1) % 50 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            remaining = (len(eval_candidates) - idx - 1) / max(rate, 0.001)
            print(f"  ... {idx+1}/{len(eval_candidates)} "
                  f"({elapsed:.0f}s ecoulées, ~{remaining:.0f}s restantes)")

        data, status = api_get(f"/api/v1/matching/candidate/{cid}?top_k={args.top_k}")
        if status != 200 or not data or not data.get("recommendations"):
            errors += 1
            continue

        recommended = [r["offer_id"] for r in data["recommendations"]]
        relevant = set(gt[cid])

        for k in [1, 3, 5, 10]:
            if k <= args.top_k:
                metrics["precision"][k].append(precision_at_k(recommended, relevant, k))
                metrics["recall"][k].append(recall_at_k(recommended, relevant, k))
                metrics["hit_rate"][k].append(hit_rate_at_k(recommended, relevant, k))
                metrics["ndcg"][k].append(ndcg_at_k(recommended, relevant, k))

        hits_3 = sum(1 for r in recommended[:3] if r in relevant)
        if hits_3 >= 2:
            good_examples.append((cid, recommended[:3], relevant, hits_3))
        elif hits_3 == 0:
            bad_examples.append((cid, recommended[:3], relevant))

        time.sleep(0.05)

    total_time = time.time() - t0

    print(f"\n[4/4] Resultats ({total_time:.0f}s, {errors} erreurs)")
    print("=" * 60)

    print(f"\n{'Metrique':<20} {'@1':>8} {'@3':>8} {'@5':>8} {'@10':>8}")
    print("-" * 52)
    for name in ["precision", "recall", "hit_rate", "ndcg"]:
        vals = []
        for k in [1, 3, 5, 10]:
            data_list = metrics[name][k]
            vals.append(sum(data_list) / max(len(data_list), 1) if data_list else 0)
        print(f"{name:<20} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f} {vals[3]:>8.4f}")

    print(f"\n{'Echantillon':<20} {len(eval_candidates)}")
    print(f"{'Erreurs':<20} {errors}")
    print(f"{'Temps total':<20} {total_time:.1f}s")

    if good_examples:
        print(f"\nTOP 5 BONS MATCHINGS (>=2 hits dans top 3):")
        print("-" * 52)
        for cid, recs, rel, hits in good_examples[:5]:
            print(f"  Candidat: {cid}")
            print(f"    Ground truth: {rel}")
            print(f"    Top 3:        {recs}")
            print(f"    Hits: {hits}/3")
            print()

    if bad_examples:
        print(f"TOP 5 MAUVAIS MATCHINGS (0 hits dans top 3):")
        print("-" * 52)
        for cid, recs, rel in bad_examples[:5]:
            print(f"  Candidat: {cid}")
            print(f"    Ground truth: {rel}")
            print(f"    Top 3:        {recs}")
            print()


if __name__ == "__main__":
    main()

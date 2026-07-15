"""
test_backend.py
Tests complets du backend ACPE Match sur l'echantillon disponible.
"""

import sys
import os
import time
import json
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

BASE = "http://localhost:8000"

passed = 0
failed = 0
results = []


def log(icon, msg):
    print(f"  {icon} {msg}")


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        log("OK", f"{name}")
    else:
        failed += 1
        log("FAIL", f"{name} -- {detail}")
    results.append((name, condition, detail))


def api_get(path):
    url = BASE + path
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8")), resp.status
    except urllib.error.HTTPError as e:
        return None, e.code
    except Exception as e:
        return None, str(e)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


# ============================================================
section("1. SANTE DE L'API")
# ============================================================

data, status = api_get("/")
test("GET / repond 200", status == 200, f"status={status}")
test("GET / contient 'ACPE Match'", data and "ACPE Match" in data.get("message", ""), str(data))

t0 = time.time()
data, status = api_get("/docs")
latency_root = time.time() - t0
test("GET /docs repond 200", status == 200, f"status={status}")


# ============================================================
section("2. COUNTS BASE DE DONNEES")
# ============================================================

import sqlite3
conn = sqlite3.connect("acpe.db")
c = conn.cursor()
db_candidates = c.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
db_offers = c.execute("SELECT COUNT(*) FROM job_offers").fetchone()[0]
conn.close()

import chromadb
chroma = chromadb.PersistentClient(path="chroma_data")
col_cand = chroma.get_collection("candidate_embeddings")
col_off = chroma.get_collection("offer_embeddings")
chroma_candidates = col_cand.count()
chroma_offers = col_off.count()

test("SQLite candidats > 0", db_candidates > 0, f"count={db_candidates}")
test("SQLite offres > 0", db_offers > 0, f"count={db_offers}")
test("ChromaDB candidats > 0", chroma_candidates > 0, f"count={chroma_candidates}")
test("ChromaDB offres > 0", chroma_offers > 0, f"count={chroma_offers}")
test("ChromaDB candidats <= SQLite candidats", chroma_candidates <= db_candidates,
     f"chroma={chroma_candidates} > sqlite={db_candidates}")
log("INFO", f"SQLite: {db_candidates} candidats, {db_offers} offres")
log("INFO", f"ChromaDB: {chroma_candidates} candidats, {chroma_offers} offres")


# ============================================================
section("3. ENDPOINTS CANDIDATS")
# ============================================================

data, status = api_get("/api/v1/candidates?limit=5")
test("GET /candidates?limit=5 -> 200", status == 200, f"status={status}")
test("Retourne 5 candidats", data and len(data) == 5, f"len={len(data) if data else 0}")

if data and len(data) > 0:
    c0 = data[0]
    test("Candidat a 'id'", "id" in c0)
    test("Candidat a 'profile_text'", "profile_text" in c0 and len(c0["profile_text"]) > 10)

data, status = api_get("/api/v1/candidates?limit=100")
test("GET /candidates?limit=100 -> 200", status == 200, f"status={status}")
test("Retourne 100 candidats", data and len(data) == 100, f"len={len(data) if data else 0}")

# Test candidat specifique
test_cand_id = "PPKOU2501080016340"
data, status = api_get(f"/api/v1/candidates/{test_cand_id}")
test(f"GET /candidates/{test_cand_id} -> 200", status == 200, f"status={status}")
test("Candidat ID correspond", data and data["id"] == test_cand_id)

data, status = api_get("/api/v1/candidates/ZZZZ_INEXISTANT")
test("GET /candidates/inexistant -> 404", status == 404, f"status={status}")


# ============================================================
section("4. ENDPOINTS OFFRES")
# ============================================================

data, status = api_get("/api/v1/job-offers?limit=5")
test("GET /job-offers?limit=5 -> 200", status == 200, f"status={status}")
test("Retourne 5 offres", data and len(data) == 5, f"len={len(data) if data else 0}")

if data and len(data) > 0:
    o0 = data[0]
    test("Offre a 'id'", "id" in o0)
    test("Offre a 'intitule'", "intitule" in o0)
    test("Offre a 'profile_text'", "profile_text" in o0 and len(o0["profile_text"]) > 10)


# ============================================================
section("5. NORMALISEURS DIRECTEMENT")
# ============================================================

from matching_engine import JobNormalizer, SectorNormalizer, EducationNormalizer, LocationNormalizer, SpecialtyNormalizer

jn = JobNormalizer()
sn = SectorNormalizer()
en = EducationNormalizer()
ln = LocationNormalizer()
spn = SpecialtyNormalizer()

# JobNormalizer
job_tests = [
    ("Agent logistique", "FAM_TRANS_LOG"),
    ("Comptable", "FAM_COMPTA_FIN"),
    ("Informatique", "FAM_IT_DATA"),
    ("Mecanicien", "FAM_HSE_INDUS"),
    ("Vendeur", "FAM_COMM_VENTE"),
    ("Medecin", "FAM_SANTE"),
]
for raw, expected_fam in job_tests:
    r = jn.normalize(raw)
    ok = r and r.get("id_famille") == expected_fam
    test(f"JobNormalizer '{raw}' -> {expected_fam}", ok,
         f"got={r.get('id_famille') if r else None}")

# SectorNormalizer
sec_tests = [
    ("Transport, Logistique & Supply Chain", "SEC_TRANS_LOG"),
    ("Commerce, Vente, Marketing & Distribution", "SEC_COMM_VENTE"),
    ("Informatique, Data & Numerique", "SEC_IT_DIGITAL"),
]
for raw, expected_sec in sec_tests:
    r = sn.normalize(raw)
    ok = r and r.get("id_secteur") == expected_sec
    test(f"SectorNormalizer '{raw[:30]}...' -> {expected_sec}", ok,
         f"got={r.get('id_secteur') if r else None}")

# EducationNormalizer
edu_tests = [
    ("Bac +3", "NV_6_BAC_3"),
    ("Licence", "NV_6_BAC_3"),
    ("Master 2", "NV_7_BAC_5"),
    ("BEP", "NV_3_PRO_N1"),
    ("CEPE", "NV_1_PRIMARY"),
    ("Doctorat", "NV_8_DOCTORAT"),
]
for raw, expected_nv in edu_tests:
    r = en.normalize(raw)
    ok = r and r.get("code_niveau") == expected_nv
    test(f"EducationNormalizer '{raw}' -> {expected_nv}", ok,
         f"got={r.get('code_niveau') if r else None}")

# LocationNormalizer
loc_tests = [
    ("Brazzaville", "BZV"),
    ("Pointe-Noire", "PNR"),
    ("Dolisie", "NIA"),
]
for raw, expected_loc in loc_tests:
    r = ln.normalize(raw)
    ok = r and r.get("code_departement") == expected_loc
    test(f"LocationNormalizer '{raw}' -> {expected_loc}", ok,
         f"got={r.get('code_departement') if r else None}")


# ============================================================
section("6. MATCHING - TESTS ENDPOINT")
# ============================================================

# Recuperer un candidat qui a un embedding dans ChromaDB
resp_cand, _ = api_get("/api/v1/candidates?limit=200")
cand_ids = [c["id"] for c in (resp_cand or [])] if resp_cand else []

# Trouver un candidat qui est dans ChromaDB
cand_in_chroma = None
for cid in cand_ids[:50]:
    try:
        col_cand.get(ids=[cid])
        cand_in_chroma = cid
        break
    except:
        continue

if cand_in_chroma:
    log("INFO", f"Candidat test: {cand_in_chroma}")

    t0 = time.time()
    data, status = api_get(f"/api/v1/matching/candidate/{cand_in_chroma}?top_k=10")
    match_latency = time.time() - t0

    test("GET matching -> 200", status == 200, f"status={status}")
    test("Reponse contient 'recommendations'", data and "recommendations" in data)
    test("Reponse contient 'total_offers_compared'", data and "total_offers_compared" in data)
    log("INFO", f"Latence matching: {match_latency:.2f}s")

    if data and data.get("recommendations"):
        recs = data["recommendations"]
        test(f"Retourne max 10 resultats", len(recs) <= 10, f"len={len(recs)}")
        test(f"Retourne au moins 1 resultat", len(recs) >= 1, f"len={len(recs)}")

        # Scores
        scores = [r["score"] for r in recs]
        test("Scores en ordre decroissant", scores == sorted(scores, reverse=True),
             f"scores={[round(s,3) for s in scores]}")
        test("Scores entre 0 et 1", all(0 <= s <= 1 for s in scores),
             f"min={min(scores):.3f}, max={max(scores):.3f}")
        log("INFO", f"Scores: {[round(s,3) for s in scores]}")

        # Skill gap
        first = recs[0]
        test("Reco a 'skill_gap'", "skill_gap" in first)
        test("Reco a 'offer_id'", "offer_id" in first)
        test("Reco a 'intitule'", "intitule" in first)
        log("INFO", f"Top 1: {first['intitule']} (score={first['score']:.3f})")
        log("INFO", f"  skill_gap: acquired={len(first.get('skill_gap',{}).get('acquired',[]))}, "
            f"missing={len(first.get('skill_gap',{}).get('missing',[]))}, "
            f"gap_score={first.get('skill_gap',{}).get('gap_score',0):.3f}")

    # Test 404
    data, status = api_get("/api/v1/matching/candidate/ZZZZ_FAKE?top_k=5")
    test("Matching candidat inexistant -> 404", status == 404, f"status={status}")
else:
    log("WARN", "Aucun candidat test trouve dans ChromaDB, tests matching sautes")


# ============================================================
section("7. QUALITE DU MATCHING (echantillon)")
# ============================================================

import random

# Recuperer des candidats avec leur metier vise
conn = sqlite3.connect("acpe.db")
conn.row_factory = sqlite3.Row
sample_cands = conn.execute(
    "SELECT id, metier_vise, secteur_demande FROM candidates WHERE metier_vise IS NOT NULL ORDER BY RANDOM() LIMIT 20"
).fetchall()
conn.close()

# Filtrer ceux qui sont dans ChromaDB
cand_pool = []
for row in sample_cands:
    try:
        col_cand.get(ids=[row["id"]])
        cand_pool.append(dict(row))
    except:
        continue

log("INFO", f"{len(cand_pool)} candidats testables sur ChromaDB")

coherence_scores = []
for cand in cand_pool[:10]:
    data, status = api_get(f"/api/v1/matching/candidate/{cand['id']}?top_k=5")
    if status != 200 or not data or not data.get("recommendations"):
        continue

    recs = data["recommendations"]
    # Verifier que les scores sont positifs
    for r in recs:
        coherence_scores.append(r["score"])

    log("INFO", f"\n  Candidat: {cand['id']}")
    log("INFO", f"    Metier vise: {cand['metier_vise']}")
    log("INFO", f"    Secteur: {cand['secteur_demande']}")
    for r in recs[:3]:
        log("INFO", f"    -> {r['intitule'][:50]:50s} score={r['score']:.3f}")

if coherence_scores:
    avg = sum(coherence_scores) / len(coherence_scores)
    test("Score moyen > 0.5", avg > 0.5, f"avg={avg:.3f}")
    log("INFO", f"\n  Score moyen global: {avg:.3f} (sur {len(coherence_scores)} resultats)")


# ============================================================
section("8. PERFORMANCE MATCHING")
# ============================================================

if cand_in_chroma:
    times = []
    for _ in range(5):
        t0 = time.time()
        data, status = api_get(f"/api/v1/matching/candidate/{cand_in_chroma}?top_k=10")
        times.append(time.time() - t0)

    avg_time = sum(times) / len(times)
    test("Temps moyen matching < 5s", avg_time < 5, f"avg={avg_time:.2f}s")
    log("INFO", f"Temps: {[f'{t:.2f}s' for t in times]}, moyenne: {avg_time:.2f}s")


# ============================================================
section("RESULTATS")
# ============================================================

print(f"\n{'='*60}")
print(f"  PASSED: {passed}  |  FAILED: {failed}  |  TOTAL: {passed+failed}")
print(f"{'='*60}")

if failed > 0:
    print("\nEchecs:")
    for name, ok, detail in results:
        if not ok:
            print(f"  - {name}: {detail}")

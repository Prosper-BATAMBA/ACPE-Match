# ACPE Match

Moteur de matching candidats / offres d'emploi pour l'Agence Congolaise pour l'Emploi (ACPE).
Projet realise pour IndabaX Congo 2026.

## Architecture

```
Donnees brutes --> Normalisation (6 normalizers) --> Enrichissement (graphes)
    --> Embedding (bge-m3) --> FAISS (top-200) --> CatBoost Ranker --> Top-K
```

| Composant | Role |
|-----------|------|
| `backend/` | API FastAPI centralisee (seule entree du systeme) |
| `matching_engine/` | Package local : normalizers + enrichment |
| `backend/dashboard.py` | Dashboard Streamlit (interface conseiller) |
| `docker-compose.yml` | Infrastructure (PostgreSQL + ChromaDB) |

## Pre-requis

- Python 3.10+
- ~3 Go de disque libre (premier lancement : telechargement du modele bge-m3)
- Fichiers sources (exclus du repo pour RGPD) :

```
matching_engine/matching_engine/data/raw/
    Demandeurs.xlsx
    Offres_enrichi.xlsx
```

## Installation

```bash
# 1. Environnement Python
cd backend
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 2. Peupler la base (SQLite + ChromaDB)
python -m seed_data

# 3. Construire l'index FAISS (obligatoire apres le seed)
python build_faiss_index.py

# 4. Lancer l'API
uvicorn app.main:app --host 0.0.0.0 --port 8000
# Docs : http://localhost:8000/docs

# 5. (Optionnel) Lancer le dashboard
streamlit run dashboard.py --server.port 8501
```

## Endpoints

| Methode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/api/v1/stats` | Statistiques globales |
| GET | `/api/v1/candidates/search?q=` | Recherche candidats |
| GET | `/api/v1/job-offers/search?q=` | Recherche offres |
| GET | `/api/v1/matching/candidate/{id}` | Top offres pour un candidat |
| GET | `/api/v1/matching/offer/{id}` | Top candidats pour une offre |
| POST | `/api/v1/matching/nl-offer-search` | Recherche NL d'offres |
| GET | `/api/v1/matching/export-csv` | Export CSV candidat -> offres |
| GET | `/api/v1/matching/export-csv-by-offer` | Export CSV offre -> candidats |

## Donnees versionnees vs exclues

| Fichier | Statut | Raison |
|---------|--------|--------|
| Code source (`backend/`, `matching_engine/`) | Inclus | Code applicatif |
| `catboost_ranker.cbm` + `ranker_config.json` | Inclus | Modele entraine (~574 KB) |
| Graphes + mappings JSON | Inclus | Config normalizers |
| `acpe.db` | Exclu | Donnees personnelles (RGPD) |
| `chroma_data/` | Exclu | Derive des donnees perso |
| `*.xlsx` sources | Exclu | Noms, contacts reels |
| `faiss_offers.index` | Exclu | Binaire stale, a reconstruire |
| `.opencode/` | Exclu | Outils internes |

## Notes

- Le modele d'embedding `BAAI/bge-m3` se telecharge automatiquement au premier lancement
  (necessite une connexion internet).
- Apres chaque changement de donnees, relancer `seed_data` puis `build_faiss_index.py`.
- `docker-compose.yml` est fourni a titre de reference (PostgreSQL + ChromaDB conteneurises).
  La configuration documentee ci-dessus utilise SQLite + ChromaDB persistante (plus simple
  pour une demo locale).

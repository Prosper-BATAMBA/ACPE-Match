# ACPE Match — Intelligent Job Matching (IndabaX Congo 2026)

Moteur de matching entre candidats et offres d'emploi de l'Agence Congolaise pour l'Emploi (ACPE).
Pipeline : embeddings **bge-m3** → recherche **FAISS** (top-200) → **CatBoost Ranker** (YetiRank).

> 📐 Architecture détaillée (centralisation Backend, circuits d'ingestion/matching) : voir [`ACPE_Match.md`](./ACPE_Match.md).

---

## ⚠️ Point clé : le modèle d'embedding s'auto-télécharge

Le modèle **`BAAI/bge-m3`** (≈ 2,3 Go) est chargé **automatiquement** par `sentence-transformers`
au premier appel d'encodage (`backend/app/services/embedding_service.py`). **Aucune installation
manuelle du modèle n'est requise** : il se télécharge depuis HuggingFace au 1er lancement
(il faut donc un accès internet et ~3 Go de disque libres pour ce 1er run).

---

## 🔒 Données personnelles — volontairement EXCLUES du dépôt (RGPD)

Pour des raisons de confidentialité, les données réelles ne sont **pas** versionnées :

| Élément | Statut | Raison |
|---------|--------|--------|
| `acpe.db` (candidats/offres, PII) | ❌ exclu | Données personnelles |
| `chroma_data/` (embeddings) | ❌ exclu | Dérivé des PII |
| `matching_engine/.../data/raw/*.xlsx` (sources brutes) | ❌ exclu | Noms, contacts réels |
| `catboost_ranker.cbm` + `ranker_config.json` | ✅ inclus | Modèle entraîné |
| `faiss_offers.index` | ✅ inclus* | Index FAISS (*stale — à reconstruire, voir dessous) |
| Code + graphes de connaissances + `skills_enriched.json` | ✅ inclus | Nécessaires au run |

**Pour faire tourner le projet, tu dois fournir tes propres fichiers sources** (hors Git) :

```
matching_engine/matching_engine/data/raw/
├── Demandeurs.xlsx      # candidats
└── Offres_enrichi.xlsx  # offres
```

Ces fichiers restent locaux (ignorés par Git). Place-les avant d'exécuter le seed.

---

## 🚀 Installation & Lancement

### 1. Environnement

```bash
cd backend
python -m venv venv
# Windows :
venv\Scripts\activate
# Linux / macOS :
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Peupler la base (SQLite + ChromaDB)

```bash
python -m seed_data
```

Importe les candidats/offres depuis tes Excel et génère leurs embeddings dans ChromaDB.

### 3. Reconstruire l'index FAISS  ⚠️ OBLIGATOIRE après le seed

L'index `faiss_offers.index` commité a été bâti sur les offres réelles d'origine. Après avoir
seedé **tes** données, reconstruis-le pour qu'il soit cohérent :

```bash
python build_faiss_index.py
```

> Si tu oublies cette étape, le matching renverra des résultats vides (l'index ne correspond
> pas aux offres en base).

### 4. Démarrer l'API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Documentation interactive : http://localhost:8000/docs

### 5. Démarrer le Dashboard (optionnel, autre terminal)

```bash
streamlit run dashboard.py --server.port 8501
```

---

## 🔌 Endpoints de l'API

Le **Backend FastAPI est la seule porte d'entrée** : tout (dashboard, scripts, frontends) passe par l'API.

| Méthode | Endpoint | Description |
|---------|----------|-------------|
| GET | `/api/v1/stats` | Statistiques globales (cache 5 min) |
| GET | `/api/v1/candidates/search?q=` | Recherche candidats (nom, métier, secteur, lieu, **ID**) |
| GET | `/api/v1/job-offers/search?q=` | Recherche offres (référence, intitulé, secteur, entreprise) |
| GET | `/api/v1/matching/candidate/{id}?top_k=` | Top offres pour un candidat |
| GET | `/api/v1/matching/offer/{id}?top_k=` | Top candidats pour une offre |
| GET | `/api/v1/matching/export-csv?candidate_ids=&top_k=` | Export CSV candidat → offres |
| GET | `/api/v1/matching/export-csv-by-offer?offer_ids=&top_k=` | Export CSV offre → candidats |

---

## 📝 Notes

- **Modèle entraîné inclus** : `catboost_ranker.cbm` + `ranker_config.json` sont versionnés.
  Pour un ranking fidèle sur tes propres données, ré-entraîne via `train_ranker.py`.
- **`docker-compose.yml`** est fourni à titre de référence (PostgreSQL + ChromaDB conteneurisés),
  mais la configuration de lancement documentée ci-dessus utilise **SQLite + ChromaDB persistante**
  (plus simple pour une démo locale).
- **Reproductibilité** : le seeding et l'indexation sont déterministes ; relance-les après tout
  changement de données source.

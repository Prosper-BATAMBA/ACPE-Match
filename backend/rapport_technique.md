# Rapport Technique — ACPE Match

## IndabaX Congo 2026 — Agence Congolaise pour l'Emploi (ACPE)

**Équipe :** ACPE Match
**Date :** Juillet 2026
**Problème :** Appariement intelligent entre demandeurs d'emploi et offres d'emploi en République du Congo

---

## 1. Contexte et Objectifs

L'Agence Congolaise pour l'Emploi (ACPE) gère un volume important de candidatures et d'offres d'emploi. Le processus manuel d'appariement est chronophile et peu optimal. Notre objectif est de construire un **système de matching intelligent** capable de :

1. **Retrouver les meilleures offres** pour chaque candidat en quelques millisecondes
2. **Reclasser (re-rank)** les résultats par un modèle de machine learning entraîné sur des données historiques
3. **Analyser les écarts de compétences** (skill gap) pour orienter la formation des candidats
4. **Fournir un dashboard interactif** aux conseillers ACPE

### Données disponibles

| Source | Volume | Description |
|--------|--------|-------------|
| Candidats | 41 285 | Profils bruts (nom, diplôme, métier visé, secteur, lieu) |
| Offres d'emploi | 2 531 | Offres enrichies (intitulé, entreprise, compétences, contrat) |
| Ground truth | 1 000 candidats | Appariements historiques (3 offres par candidat) |
| Compétences | 73 | Référentiel normalisé avec sous-familles |
| Secteurs | 12 domaines | Hiérarchie secteur → famille → spécialité |

---

## 2. Architecture

### Principe : Backend Centralisé

Le Backend FastAPI est la **seule porte d'entrée et de sortie**. Le dashboard, les scripts et les futurs frontends communiquent exclusivement via l'API REST.

```
┌──────────────┐     HTTP/REST     ┌──────────────────┐
│  Dashboard   │◄─────────────────►│  Backend FastAPI  │
│  Streamlit   │                   │  (port 8000)      │
└──────────────┘                   │                  │
                                   │  ┌────────────┐  │
                                   │  │matching_eng│  │
                                   │  │(normalizers│  │
                                   │  │+enrichment)│  │
                                   │  └─────┬──────┘  │
                                   │        ▼         │
                                   │  ┌────────────┐  │
                                   │  │sentence-   │  │
                                   │  │transformers│  │
                                   │  │(bge-m3)    │  │
                                   │  └─────┬──────┘  │
                                   └────────┼─────────┘
                                            ▼
                              ┌───────────┬───────────┐
                              ▼           ▼           ▼
                         ┌─────────┐ ┌─────────┐ ┌────────┐
                         │ SQLite  │ │ChromaDB │ │  FAISS │
                         │(SQL)    │ │(vecteurs│ │(index) │
                         │         │ │ candidats│ │        │
                         └─────────┘ └─────────┘ └────────┘
```

### Endpoints API (15)

| Méthode | Endpoint | Rôle |
|---------|----------|------|
| `GET` | `/` | Health check |
| `POST` | `/api/v1/candidates` | Créer un candidat |
| `GET` | `/api/v1/candidates` | Lister les candidats |
| `GET` | `/api/v1/candidates/search?q=` | Rechercher un candidat |
| `GET` | `/api/v1/candidates/{id}` | Détail d'un candidat |
| `POST` | `/api/v1/job-offers` | Créer une offre |
| `GET` | `/api/v1/job-offers` | Lister les offres |
| `GET` | `/api/v1/job-offers/search?q=` | Rechercher une offre |
| `GET` | `/api/v1/job-offers/{id}` | Détail d'une offre |
| `GET` | `/api/v1/matching/candidate/{id}` | Top offres pour un candidat |
| `GET` | `/api/v1/matching/offer/{id}` | Top candidats pour une offre |
| `POST` | `/api/v1/matching/nl-offer-search` | Recherche NL d'offres |
| `GET` | `/api/v1/matching/export-csv` | Export CSV candidat → offres |
| `GET` | `/api/v1/matching/export-csv-by-offer` | Export CSV offre → candidats |
| `GET` | `/api/v1/stats` | Statistiques globales |

---

## 3. Pipeline ML

### 3.1 Embeddings — BAAI/bge-m3

**Modèle :** `BAAI/bge-m3` (1024 dimensions, 8192 tokens max, licence MIT)

Le texte de profil (structuré par le `TextEnricher`) est encodé en vecteur de 1024 dimensions. Le modèle supporte le français et le multilingue, ce qui est adapté au contexte congolais.

**Encodage :** 20 288 candidats + 2 531 offres = 22 819 vecteurs stockés dans ChromaDB.

### 3.2 Retrieval — FAISS IndexFlatIP

**Index :** `faiss.IndexFlatIP` (Inner Product sur vecteurs L2-normalisés = similarité cosinus)

Pour chaque candidat, on récupère les **top-200 offres** les plus proches en similarité sémantique. L'index FAISS est persisté sur disque (`faiss_offers.index`).

**Performance :** Recherche en <10ms sur 2 531 offres.

### 3.3 Re-ranking — CatBoost YetiRank

**Modèle :** CatBoost Ranker avec loss `YetiRank` et métrique `NDCG:top=5`

Le ranker prend les top-200 offres de FAISS et les re-classe selon 79 features. Le modèle est entraîné sur **1 million de paires** candidat-offre (80% train / 20% test), avec des hard negatives issues de FAISS.

**Entraînement :** 500 itérations, learning_rate=0.1, depth=6, early stopping à 50.

### 3.4 Feature Engineering — 79 features

Les features se répartissent en 6 catégories :

| Catégorie | Nombre | Exemples |
|-----------|--------|----------|
| Matching structurel | 12 | same_id_famille, same_id_secteur, same_departement |
| Profil candidat | 8 | candidate_age, education_gap, candidate_mobilite |
| Profil offre | 8 | offer_type_contrat, intitule_length, offer_profile_length |
| Sémantique | 6 | semantic_similarity, skill_gap_score, metier_intitule_jaccard |
| Domaines de compétences | 26 | offer_domain_finance_count, offer_domain_it_digital_has |
| Contexte graphe | 11 | sector_proximity, family_proximity, sector_tension |
| Sous-famille + divers | 8 | same_id_sous_famille, candidate_has_specialite |

**Réduction de dimensionnalité :** Les 73 compétences individuelles (`offer_skill_SKILL_*`) ont été agrégées en **12 domaines** (finance, it_digital, commerce, etc.), réduisant de 73 features binaires à 26 features (count + has par domaine).

---

## 4. Données et Normalisation

### Pipeline d'ingestion

Données brutes → **6 normalizers** → **KnowledgeEnricher** → **TextEnricher** → Texte structuré → **bge-m3** → Vecteur

| Normalizer | Entrée | Sortie |
|------------|--------|--------|
| `JobNormalizer` | "Comptable" | `FAM_COMPTA_FIN` |
| `SectorNormalizer` | "Transport" | `SEC_TRANS_LOG` |
| `EducationNormalizer` | "Bac+3" | `NV_6_BAC_3` |
| `SpecialtyNormalizer` | "Audit" | `{id_famille_affiliation: "FAM_COMPTA_FIN"}` |
| `SkillNormalizer` | "Python, SQL" | `[{id_skill: "SKILL_IT_PYTHON"}, ...]` |
| `LocationNormalizer` | "Brazzaville" | `BZV` |

### Défis des données

- **Profils extrêmement uniformes** : écart-type de 57 caractères sur les textes de profil (99.5% des candidats ont `code_departement = "INC"`)
- **Compétences/expérience manquantes** : `competences_brutes` et `experience_libre` toujours NULL dans les données source
- **Données d'apprentissage** : ground truth limité à 1 000 candidats avec 3 offres chacun

---

## 5. Résultats d'Évaluation

### Méthodologie

- **4 728 candidats** évalués (ensemble disjoint de l'entraînement)
- **Ground truth** : appariements historiques ACPE
- **Pool** : top-200 offres FAISS par candidat
- **Métriques** : Hit Rate@K, NDCG@K, Precision@K, Recall@K, MRR, MAP

### Résultats

| Métrique | FAISS Seul | CatBoost Ranker | Gain |
|----------|-----------|-----------------|------|
| **Hit Rate@1** | 40.7% | **83.2%** | +42.5 pts |
| **Hit Rate@3** | 56.7% | **91.9%** | +35.2 pts |
| **Hit Rate@5** | 64.7% | **96.0%** | +31.3 pts |
| **Hit Rate@10** | 75.5% | **98.2%** | +22.7 pts |
| **NDCG@3** | — | **0.879** | — |
| **NDCG@10** | — | **0.880** | — |
| **Precision@3** | 31.6% | **65.2%** | +33.6 pts |
| **Recall@5** | 41.4% | **83.2%** | +41.8 pts |
| **MRR** | — | **0.885** | — |
| **MAP** | — | **0.789** | — |

### Top 10 features (importance)

| Rang | Feature | Importance |
|------|---------|-----------|
| 1 | metier_intitule_jaccard | 6.76% |
| 2 | semantic_similarity | 4.63% |
| 3 | secteur_demande_jaccard | 3.36% |
| 4 | offer_profile_length | 1.38% |
| 5 | intitule_length | 0.81% |
| 6 | candidate_qualification_length | 0.80% |
| 7 | offer_intitule_length | 0.48% |
| 8 | profile_length_ratio | 0.27% |
| 9 | cand_metier_vise_len | 0.25% |
| 10 | candidate_metier_length | 0.24% |

**Analyse** : Les features textuelles (Jaccard métiers/secteurs) dominent, suivies de la similarité sémantique. Les features structurelles (même famille, même secteur) ont un impact marginal car la plupart des candidats ont `code_departement = "INC"`.

---

## 6. Livrables

### Fichiers livrés

| Composant | Description |
|-----------|-------------|
| `backend/app/` | API FastAPI complète (9 endpoints) |
| `backend/dashboard.py` | Dashboard Streamlit 4 pages |
| `backend/rapport_technique.md` | Ce rapport |
| `matching_engine/` | Package de normalisation + enrichissement |
| `backend/catboost_ranker.cbm` | Modèle CatBoost entraîné |
| `backend/faiss_offers.index` | Index FAISS persistant |
| `backend/acpe.db` | Base SQLite |
| `backend/chroma_data/` | ChromaDB persistant |

### Dashboard

| Page | Contenu |
|------|---------|
| Vue d'ensemble | KPI, graphiques distributions, top familles |
| Matching | Recherche candidat → top-K offres + skill gap |
| Export CSV | Multi-select → téléchargement CSV |
| Rapport | Ce document technique |

### Lancement

```bash
# API
cd backend && uvicorn app.main:app --reload --port 8000

# Dashboard
cd backend && streamlit run dashboard.py --server.port 8501
```

---

## 7. Perspectives

### Court terme
- **Encoder les 21 000 candidats restants** — `seed_candidates_only.py` avec checkpoint
- **Sélection de features** — réduire de 79 à ~50 features (domaines peu discriminants)
- **Dashboard temps réel** — WebSocket pour mise à jour instantanée

### Moyen terme
- **Intégrer les compétences candidats** — quand les données le permettront
- **Fine-tuning du modèle** — plus de données ground truth
- **Déploiement Docker** — 4 conteneurs (backend, dashboard, PostgreSQL, ChromaDB)

### Défis identifiés
- Uniformité des profils candidats (99.5% sans département)
- Absence de compétences/expérience dans les données source
- Ground truth limité (1 000 candidats)

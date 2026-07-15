# 🏗️ Architecture Globale (Centralisation Backend) — ACPE Match

Ce document explique comment notre Backend (FastAPI) centralise toutes les demandes de la plateforme ACPE et pilote lui-même le calcul IA (normalisation, enrichissement et embeddings), sans dépendre d'un orchestrateur externe.

---

## 1. Le Plan de Centralisation (Architecture)

Ici, le **Backend FastAPI** est la seule porte d'entrée et de sortie. La plateforme ACPE ne parle qu'à lui. Contrairement à la première version, le calcul IA n'est plus délégué à un service externe (n8n + Ollama) : le Backend embarque directement la librairie `matching_engine` (normalisation + enrichissement) et le modèle `sentence-transformers` (embeddings). Tout se joue dans le même processus Python.

```text
┌────────────────────────────────────────────────────────────────────────┐
│                              acpe-network                              │
│                                                                        │
│   ┌─────────────────┐     1. Requête Unique      ┌──────────────────┐  │
│   │ Plateforme ACPE │───────────────────────────>│    BACKEND       │  │
│   │   (ou Front)    │<───────────────────────────│   (FastAPI)      │  │
│   └─────────────────┘     4. Réponse Finale      │                  │  │
│                                                   │ ┌──────────────┐ │  │
│                                                   │ │matching_engine│ │  │
│                                                   │ │(normalizers + │ │  │
│                                                   │ │ enrichment)   │ │  │
│                                                   │ └──────┬───────┘ │  │
│                                                   │        ▼         │  │
│                                                   │ ┌──────────────┐ │  │
│                                                   │ │sentence-      │ │  │
│                                                   │ │transformers   │ │  │
│                                                   │ │(embeddings)   │ │  │
│                                                   │ └──────┬───────┘ │  │
│                                                   └────────┼─────────┘  │
│                                             2/3. Range la donnée        │
│                                                            ▼            │
│                                                  ┌───────────────┐      │
│                                                  │ LES MEMOIRES  │      │
│                                                  │ 1. PostgreSQL │      │
│                                                  │ 2. ChromaDB   │      │
│                                                  └───────────────┘      │
└────────────────────────────────────────────────────────────────────────┘
```

## 2. Les 2 Circuits Centralisés (Schémas d'Exécution)

### Circuit A : L'Ingestion Centralisée (Quand un CV ou une Offre arrive)

Tout passe par le Backend, qui fait maintenant tout le travail lui-même, en interne — normalisation, enrichissement puis embedding — sans aller-retour réseau vers un service tiers.

```text
[ Plateforme ACPE ] ───(1. POST /api/v1/data)───► [ Backend (FastAPI) ]
                                                          │
                                        (2. matching_engine : normalise
                                         métier/secteur/études/spécialité/
                                         compétences/localité, puis enrichit
                                         via les graphes de connaissances)
                                                          │
                                                          ▼
                                              [ Texte de profil structuré ]
                                                          │
                                        (3. sentence-transformers : encode
                                         ce texte en vecteur, en local,
                                         dans le même processus)
                                                          │
                                                          ▼
                                                [ Backend (FastAPI) ]
                                                          │
                         ┌────────────────────────────────┴────────────────────────────────┐
                         ▼ (4a. Mémoire Classique)                                         ▼ (4b. Mémoire IA)
                  [ PostgreSQL ]                                                    [ ChromaDB ]
            Range le Nom, Ville, Diplôme,                                          Range le Vecteur calculé
            IDs canoniques (métier, secteur...)                                    par sentence-transformers
```

### Circuit B : Le Matching Centralisé (Quand le conseiller demande les résultats)

Inchangé dans son principe : la recherche reste instantanée car toutes les données ont déjà été pré-calculées et centralisées au bon endroit.

```text
[ Plateforme ACPE ] ───(1. GET /api/v1/matching/{id})───► [ Backend (FastAPI) ]
                                                                  │
                                    ├──► 2. [ Filtre PostgreSQL ] -> Élimine les mauvaises villes/diplômes.
                                    │
                                    ├──► 3. [ Tri ChromaDB ] -> Trouve le Top 5 des meilleures offres sémantiques
                                    │        (vecteurs sentence-transformers, similarité cosinus).
                                    │
                                    └──► 4. [ Code Python ] -> Calcule le pourcentage final et le Skill Gap.
                                    │
[ Réponse JSON Structurée ] <───────┘ (Le site ACPE reçoit tout d'un coup et met à jour l'affichage)
```

## 3. Les Cas d'Utilisation du Système

### Cas 1 : Ajout d'une offre ou d'un candidat

- **Action :** L'utilisateur clique sur "Valider" sur le site ACPE.
- **Coulisses :** Le site envoie les infos brutes au Backend. Le Backend appelle `matching_engine` (normalizers puis `TextEnricher.build_profile_text`) pour produire un texte de profil propre et structuré, puis encode ce texte avec `sentence-transformers` (chargé une seule fois en mémoire au démarrage du Backend). Il enregistre ensuite les champs canoniques dans PostgreSQL et le vecteur dans ChromaDB.

### Cas 2 : Demande de recommandations (Matching)

- **Action :** Le conseiller clique sur "Voir les profils recommandés".
- **Coulisses :** Le Backend fait une requête SQL rapide combinée à une recherche dans ChromaDB. En moins de 50 millisecondes, il renvoie le Top 5 avec les compétences manquantes sous forme de badges (Vert = Acquis, Rouge = Manquant).

## 4. Architecture des fichiers

```text
acpe-ia-module/
│
├── docker-compose.yml          # Orchestration des 4 conteneurs
├── SPEC.md                     # Ce fichier de spécifications
│
├── matching_engine/            # --- PACKAGE PARTAGÉ (normalisation + enrichissement) ---
│   ├── pyproject.toml           # dépendances : rapidfuzz
│   └── matching_engine/
│       ├── __init__.py
│       ├── config.py            # Chemins des mappings/graphes, résolus via __file__
│       ├── normalizers/         # Couche 1 : texte brut -> ID canonique
│       │   ├── job_normalizer.py
│       │   ├── secteur_normalizer.py
│       │   ├── education_normalizer.py
│       │   ├── speciality_normalizer.py
│       │   ├── skill_normalizer.py
│       │   └── localite_normalizer.py
│       ├── enrichment/          # Couche 2 : ID canonique -> contexte enrichi
│       │   ├── knowledge_enricher.py   # Résout les compétences/contexte via les graphes
│       │   └── text_enricher.py        # Assemble le texte final prêt pour l'embedding
│       └── data/
│           ├── mappings/        # job_mapping.json, secteur_mapping.json, skills.json, ...
│           └── graphs/          # job_knowledge_graph.json, secteur_knowledge_graph.json, ...
│
├── backend/                    # --- COMPOSANT BACKEND (FastAPI) ---
│   ├── Dockerfile
│   ├── requirements.txt        # fastapi, uvicorn, sqlalchemy, psycopg2-binary, pydantic,
│   │                           # sentence-transformers, torch, matching-engine (package local ci-dessus)
│   └── app/
│       ├── __init__.py
│       ├── main.py              # Initialisation de FastAPI et inclusion des routeurs
│       ├── config.py            # Chargement de os.getenv (DB, Chroma, nom du modèle ST)
│       ├── database.py          # Session SQLAlchemy et création du moteur PostgreSQL
│       ├── chromadb_client.py   # Connexion persistante au client ChromaDB
│       │
│       ├── models/              # Modèles ORM (PostgreSQL)
│       │   ├── candidate.py     # Table 'candidates' (id, nom, lieu, etudes, exp, competences)
│       │   └── job_offer.py     # Table 'job_offers' (id, entreprise, titre, lieu, etudes_min, exp_min)
│       │
│       ├── schemas/             # Schémas de validation Pydantic (Données Entrées/Sorties)
│       │   ├── candidate.py     # CandidateCreate, CandidateResponse
│       │   └── job_offer.py     # JobOfferCreate, JobOfferResponse
│       │
│       ├── services/            # Logique Algorithmique & Intelligence Artificielle
│       │   ├── profile_builder.py # Appelle matching_engine (normalizers + TextEnricher)
│       │   │                      # pour transformer les données brutes en texte de profil
│       │   ├── embedding_service.py # Charge sentence-transformers une seule fois (singleton)
│       │   │                        # et expose encode(texte) -> vecteur
│       │   └── matching_engine_service.py # Algorithme du calcul du score hybride (SQL + Sémantique)
│       │
│       └── routers/             # Points d'accès API (Endpoints)
│           ├── candidates.py    # POST /api/v1/candidates
│           ├── job_offers.py    # POST /api/v1/job-offers
│           └── matching.py      # GET /api/v1/matching/candidate/{id}
│
└── frontend/
```

**Ce qui change par rapport à la V1 :**
- Le dossier `matching_engine/` (déjà développé) devient une dépendance locale installée dans le conteneur Backend (`pip install -e ../matching_engine` ou copié et installé pendant le build Docker).
- `services/profile_builder.py` remplace l'appel réseau vers n8n : il importe directement les `Normalizers` et le `TextEnricher` du package `matching_engine`.
- `services/embedding_service.py` remplace `ai_service.py` (qui appelait Ollama en HTTP) : le modèle `sentence-transformers` est chargé une seule fois au démarrage du Backend et réutilisé pour chaque requête, en mémoire, sans latence réseau.
- Les conteneurs `n8n` et `ollama` disparaissent : plus besoin d'orchestrateur de workflow ni de moteur LLM local séparé, puisque `sentence-transformers` tourne directement dans le processus Python du Backend.

## 5. L'Architecture Docker (Configuration de l'Infrastructure)

On passe de 6 à **4 conteneurs**. Le Backend embarque maintenant tout le pipeline IA (normalisation, enrichissement, embedding), donc n8n et Ollama ne sont plus nécessaires.

Chaque conteneur est configuré pour démarrer dans le bon ordre grâce aux règles `depends_on`, garantissant que le Backend ne se lance pas avant que ses bases de données ne soient prêtes.

### Fichier `docker-compose.yml`

```yaml
version: '3.8'

# Le réseau privé qui permet à nos conteneurs de discuter entre eux en circuit fermé
networks:
  acpe-network:
    driver: bridge

services:
  # 1. Le Hub Central (Logique Métier, API, Normalisation, Enrichissement et Embeddings)
  backend:
    build: ./backend
    container_name: acpe-backend
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
      - st_models_cache:/root/.cache/torch/sentence_transformers  # Cache du modèle (évite de le re-télécharger)
    environment:
      - DATABASE_URL=postgresql://postgres:acpe2026@postgresql:5432/acpe_db
      - CHROMADB_HOST=chromadb
      - CHROMADB_PORT=8000
      - SENTENCE_TRANSFORMER_MODEL=BAAI/bge-m3
    networks:
      - acpe-network
    depends_on:
      postgresql:
        condition: service_healthy
      chromadb:
        condition: service_started

  # 2. L'Interface Conseiller (Widget / Extension Démo)
  frontend:
    build: ./frontend
    container_name: acpe-frontend
    ports:
      - "3000:3000"
    environment:
      - API_URL=http://backend:8000
    networks:
      - acpe-network
    depends_on:
      - backend

  # 3. Base de Données Classique (Filtres métiers, Villes, Diplômes)
  postgresql:
    image: postgres:15-alpine
    container_name: acpe-postgres
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=acpe2026
      - POSTGRES_DB=acpe_db
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - acpe-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 5s
      timeout: 5s
      retries: 5

  # 4. Base Vectorielle (Indexation des sens des textes, vecteurs sentence-transformers)
  chromadb:
    image: chromadb/chroma:latest
    container_name: acpe-chromadb
    ports:
      - "8010:8000"
    volumes:
      - chroma_data:/chroma/chroma
    networks:
      - acpe-network

volumes:
  postgres_data:
  chroma_data:
  st_models_cache:
```

### Pourquoi ce changement simplifie l'architecture

- **Moins de conteneurs à faire tourner en équipe** sur l'ordinateur de démo (4 au lieu de 6) : plus de gestion des workflows n8n, plus de téléchargement/pilotage de modèles Ollama.
- **Moins de latence** : l'embedding se fait en mémoire dans le même processus que l'API, sans aller-retour HTTP vers un service externe.
- **Reproductibilité** : `sentence-transformers` (ex. `BAAI/bge-m3`, adapté au français) est un modèle figé et versionné, contrairement à un pipeline n8n qu'il faut réimporter/reconfigurer sur chaque machine de l'équipe.
- **Cohérence du pipeline texte** : le module `matching_engine` (normalizers + `TextEnricher`) a été conçu précisément pour produire "une chaîne de caractères propre, déterministe et structurée par sections, prête à être vectorisée par un Sentence Transformer" — l'intégration est donc directe, sans adaptation.

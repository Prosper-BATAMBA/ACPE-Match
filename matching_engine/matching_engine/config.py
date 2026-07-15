"""
config.py

Point UNIQUE de vérité pour les chemins de données du package.

Pourquoi ce fichier existe :
Un chemin relatif comme "data/mappings/job_mapping.json" est résolu par
Python par rapport au RÉPERTOIRE D'EXÉCUTION (cwd), pas par rapport à
l'emplacement du fichier .py qui contient ce chemin. Tant que ce package est
un script qu'on lance depuis sa propre racine, ça passe inaperçu. Mais dès
qu'il est installé et importé depuis un autre projet ou une API, le cwd
appartient à cet autre projet — et les chemins relatifs cassent.

Solution : calculer les chemins une seule fois ici, à partir de `__file__`
(l'emplacement réel de ce module sur le disque), puis les réutiliser partout
ailleurs dans le package. Ça fonctionne quel que soit l'endroit d'où le
package est importé ou exécuté.
"""

from pathlib import Path

# Racine du package = dossier contenant ce fichier config.py
PACKAGE_ROOT = Path(__file__).resolve().parent

DATA_DIR = PACKAGE_ROOT / "data"
MAPPINGS_DIR = DATA_DIR / "mappings"
GRAPHS_DIR = DATA_DIR / "graphs"
RAW_DATA_DIR = DATA_DIR / "raw"

# --- Datasets bruts (non versionnés, non embarqués dans le package pip) ---
# Utilisés uniquement par des scripts hors-runtime qui (re)génèrent les
# mappings et graphes ci-dessous. Aucun Normalizer ni Enricher n'y touche
# directement au runtime.
OFFRES_ACPE_PATH = RAW_DATA_DIR / "Offres_enrichi.xlsx"
OFFRES_ACPE_EXTENSIONS_PATH = RAW_DATA_DIR / "Offres_ACPE_Extensions.xlsx"
DEMANDEURS_PATH = RAW_DATA_DIR / "Demandeurs_.xlsx"

# --- Mappings (couche 1 : normalisation) ---
JOB_MAPPING_PATH = MAPPINGS_DIR / "job_mapping_enriched.json"
SECTEUR_MAPPING_PATH = MAPPINGS_DIR / "secteur_mapping.json"
NIVEAU_ETUDE_MAPPING_PATH = MAPPINGS_DIR / "niveau_etude.json"
SPECIALITY_MAPPING_PATH = MAPPINGS_DIR / "speciality_mapping.json"
SKILLS_MAPPING_PATH = MAPPINGS_DIR / "skills_enriched.json"
LOCALITE_MAPPING_PATH = MAPPINGS_DIR / "localite.json"

# --- Graphes (couche 2 : enrichissement) ---
JOB_KNOWLEDGE_GRAPH_PATH = GRAPHS_DIR / "job_knowledge_graph_v3.json"
SPECIALITY_KNOWLEDGE_GRAPH_PATH = GRAPHS_DIR / "speciality_knowledge_graph.json"
SECTEUR_KNOWLEDGE_GRAPH_PATH = GRAPHS_DIR / "secteur_knowledge_graph.json"

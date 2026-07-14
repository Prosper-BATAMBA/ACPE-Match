import json
import os
import re
import unicodedata

try:
    from rapidfuzz import process, fuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False

from ..config import LOCALITE_MAPPING_PATH


class LocationNormalizer:

    # Score minimum (0-100) pour accepter une correspondance floue.
    FUZZY_MATCH_THRESHOLD = 85

    def __init__(self, mapping_file_path=None):
        """
        Constructeur : Charge le mapping des départements et prépare l'index de recherche.
        """
        mapping_file_path = mapping_file_path or str(LOCALITE_MAPPING_PATH)
        self.location_metadata = {}  # Stocke le nom officiel (ex: "Pointe-Noire") via son code ("PNR")
        self.location_index = {}     # Index inversé pour la recherche (mot-clé -> code_departement)

        self._load_and_index(mapping_file_path)

    def _clean_text(self, text):
        """
        Nettoie le texte : minuscules, sans accents, caractères spéciaux remplacés par des espaces.
        """
        if not text or not isinstance(text, str):
            return ""

        text = text.lower()
        text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _load_and_index(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Fichier de mapping introuvable : {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        departments = data.get("departments_mapping", [])

        for dept in departments:
            code = dept.get("code_departement", "INC")
            nom = dept.get("nom_departement", "Inconnu")

            # 1. Sauvegarde des métadonnées du département (une seule fois par code)
            self.location_metadata[code] = {
                "code_departement": code,
                "nom_departement": nom
            }

            # Fonction interne pour indexer
            def index_location(loc):
                clean_loc = self._clean_text(loc)
                # On ignore les mots trop courts pour éviter les faux positifs
                if len(clean_loc) >= 3 and clean_loc not in self.location_index:
                    self.location_index[clean_loc] = code

            # 2. Indexer le nom officiel lui-même
            index_location(nom)

            # 3. Indexer toutes les villes et coquilles incluses
            for lieu in dept.get("lieux_inclus", []):
                index_location(lieu)

    def normalize(self, raw_location):
        """
        Prend un nom de ville ou de département brut et renvoie le dictionnaire
        avec le code département officiel.
        """
        fallback_meta = {
            "code_departement": "INC",  # INC pour Inconnu / Hors Congo
            "nom_departement": "Non spécifié / Étranger",
            "localite_brute_nettoyee": ""
        }

        if not raw_location or not isinstance(raw_location, str):
            return fallback_meta

        clean_input = self._clean_text(raw_location)
        # On conserve la localité brute pour l'embedding
        fallback_meta["localite_brute_nettoyee"] = raw_location.strip().title()

        # 1. Recherche exacte
        if clean_input in self.location_index:
            code = self.location_index[clean_input]
            return {
                "localite_brute_nettoyee": raw_location.strip().title(),
                **self.location_metadata[code]
            }

        # 2. Recherche partielle : on garde le lieu indexé LE PLUS LONG qui matche,
        # pour éviter qu'un nom générique/court ne masque un lieu plus spécifique
        # présent dans la même phrase (ex: un quartier vs le nom du département).
        best_match = None
        for indexed_lieu, code in self.location_index.items():
            if len(indexed_lieu) >= 3 and f" {indexed_lieu} " in f" {clean_input} ":
                if best_match is None or len(indexed_lieu) > len(best_match[0]):
                    best_match = (indexed_lieu, code)
        if best_match is not None:
            return {
                "localite_brute_nettoyee": raw_location.strip().title(),
                **self.location_metadata[best_match[1]]
            }

        # 3. Recherche floue (typos, coquilles) via RapidFuzz
        if _RAPIDFUZZ_AVAILABLE and self.location_index:
            match = process.extractOne(
                clean_input,
                self.location_index.keys(),
                scorer=fuzz.WRatio,
                score_cutoff=self.FUZZY_MATCH_THRESHOLD
            )
            if match is not None:
                matched_key = match[0]
                code = self.location_index[matched_key]
                return {
                    "localite_brute_nettoyee": raw_location.strip().title(),
                    **self.location_metadata[code]
                }

        # 4. Fallback : Non trouvé
        return fallback_meta

# === TEST D'UTILISATION ===
# normalizer = LocationNormalizer()
#
# # Test 1 : Coquille et orthographe
# print(normalizer.normalize("Pointe Noire"))
# # -> {'localite_brute_nettoyee': 'Pointe Noire', 'code_departement': 'PNR', 'nom_departement': 'Pointe-Noire'}
#
# # Test 2 : Extraction depuis une phrase
# print(normalizer.normalize("Disponible pour travailler dans la Cuvette-Ouest"))
# # -> {'localite_brute_nettoyee': 'Disponible Pour Travailler Dans La Cuvette-Ouest', 'code_departement': 'CVO', 'nom_departement': 'Cuvette-Ouest'}

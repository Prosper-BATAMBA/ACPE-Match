import json
import os
import re
import unicodedata

try:
    from rapidfuzz import process, fuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False

from ..config import SPECIALITY_MAPPING_PATH


class SpecialtyNormalizer:

    # Score minimum (0-100) pour accepter une correspondance floue.
    FUZZY_MATCH_THRESHOLD = 85

    def __init__(self, mapping_file_path=None):
        """
        Constructeur : Charge le JSON et prépare un index ultra-rapide en mémoire.
        """
        mapping_file_path = mapping_file_path or str(SPECIALITY_MAPPING_PATH)
        self.specialty_metadata = {}  # Stocke les métadonnées de la famille
        self.specialty_index = {}     # Index inversé pour la recherche (Spécialité -> id_famille)

        self._load_and_index(mapping_file_path)

    def _clean_text(self, text):
        """
        Nettoie le texte : minuscules, sans accents, sans ponctuation.
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

        mappings = data.get("specialty_mappings", [])

        for group in mappings:
            id_famille = group.get("id_famille", "FAM_GENERAL_OU_AUTRE")
            libelle = group.get("libelle_famille", "Tronc Commun / Formations Généralistes")

            # Sauvegarde des métadonnées de cette famille (une seule fois par id_famille)
            self.specialty_metadata[id_famille] = {
                "id_famille_affiliation": id_famille,
                "famille_affiliation": libelle
            }

            # Indexation de toutes les filières associées
            for filiere in group.get("filieres_associees", []):
                clean_filiere = self._clean_text(filiere)
                if clean_filiere and clean_filiere not in self.specialty_index:
                    self.specialty_index[clean_filiere] = id_famille

    def normalize(self, raw_specialty):
        """
        Prend une spécialité brute (ex: "Génie Logistique") et renvoie
        un dictionnaire avec l'affiliation à une famille de métiers.
        """
        fallback_meta = {
            "specialite_brute_nettoyee": "",
            "id_famille_affiliation": "FAM_GENERAL_OU_AUTRE",
            "famille_affiliation": "Tronc Commun / Formations Généralistes"
        }

        if not raw_specialty or not isinstance(raw_specialty, str):
            return fallback_meta

        clean_input = self._clean_text(raw_specialty)
        fallback_meta["specialite_brute_nettoyee"] = clean_input.title()

        # 1. Recherche exacte
        if clean_input in self.specialty_index:
            id_fam = self.specialty_index[clean_input]
            return {
                "specialite_brute_nettoyee": clean_input.title(),
                **self.specialty_metadata[id_fam]
            }

        # 2. Recherche partielle : on garde la filière indexée LA PLUS LONGUE qui matche,
        # pour éviter qu'une filière générique ("genie") ne masque une filière plus
        # spécifique présente dans la même phrase ("genie electrique").
        best_match = None
        for indexed_filiere, id_fam in self.specialty_index.items():
            if len(indexed_filiere) > 3 and f" {indexed_filiere} " in f" {clean_input} ":
                if best_match is None or len(indexed_filiere) > len(best_match[0]):
                    best_match = (indexed_filiere, id_fam)
        if best_match is not None:
            return {
                "specialite_brute_nettoyee": clean_input.title(),
                **self.specialty_metadata[best_match[1]]
            }

        # 3. Recherche floue (typos, variantes non listées) via RapidFuzz
        if _RAPIDFUZZ_AVAILABLE and self.specialty_index:
            match = process.extractOne(
                clean_input,
                self.specialty_index.keys(),
                scorer=fuzz.WRatio,
                score_cutoff=self.FUZZY_MATCH_THRESHOLD
            )
            if match is not None:
                matched_key = match[0]
                id_fam = self.specialty_index[matched_key]
                return {
                    "specialite_brute_nettoyee": clean_input.title(),
                    **self.specialty_metadata[id_fam]
                }

        # 4. Fallback : Si on ne trouve pas la spécialité, on la classe comme "Tronc Commun"
        # mais on garde la chaîne nettoyée pour ne pas perdre la donnée dans l'embedding.
        return fallback_meta

# === TEST D'UTILISATION ===
# normalizer = SpecialtyNormalizer()
#
# # Test 1 : L'orthographe correspond parfaitement (mais avec de la casse)
# print(normalizer.normalize("Génie Logistique"))
# # -> {'specialite_brute_nettoyee': 'Genie Logistique', 'id_famille_affiliation': 'FAM_TRANS_LOG', 'famille_affiliation': 'Transport, Logistique & Supply Chain'}
#
# # Test 2 : Une filière généraliste "Bac D"
# print(normalizer.normalize("Bac D"))
# # -> {'specialite_brute_nettoyee': 'Bac D', 'id_famille_affiliation': 'FAM_GENERAL_OU_AUTRE', 'famille_affiliation': 'Tronc Commun / Formations Généralistes'}

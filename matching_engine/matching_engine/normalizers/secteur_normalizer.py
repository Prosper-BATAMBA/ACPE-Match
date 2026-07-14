import json
import os
import re
import unicodedata

try:
    from rapidfuzz import process, fuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False

from ..config import SECTEUR_MAPPING_PATH


class SectorNormalizer:

    # Score minimum (0-100) pour accepter une correspondance floue.
    FUZZY_MATCH_THRESHOLD = 85

    def __init__(self, mapping_file_path=None):
        """
        Constructeur : Charge le JSON et prépare un index ultra-rapide en mémoire.
        """
        mapping_file_path = mapping_file_path or str(SECTEUR_MAPPING_PATH)
        self.sector_metadata = {}  # Stocke les métadonnées canoniques (id, libellé)
        self.sector_index = {}     # Index inversé pour la recherche rapide O(1)

        self._load_and_index(mapping_file_path)

    def _clean_text(self, text):
        """
        Nettoie une chaîne de caractères : minuscules, sans accents, sans caractères spéciaux.
        """
        if not text or not isinstance(text, str):
            return ""

        text = text.lower()
        text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def _load_and_index(self, filepath):
        """
        Charge le JSON et construit les dictionnaires de recherche.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Fichier de mapping introuvable : {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        secteurs = data.get("canonical_sectors", [])

        for secteur in secteurs:
            id_secteur = secteur.get("id_secteur", "SEC_AUTRES")
            libelle = secteur.get("libelle_canonique", "Autres Secteurs / Non Spécifié")

            # Sauvegarde des métadonnées du secteur (une seule fois par id_secteur)
            self.sector_metadata[id_secteur] = {
                "id_secteur": id_secteur,
                "secteur_canonique": libelle
            }

            # Ajout du libellé canonique lui-même dans l'index
            clean_canonique = self._clean_text(libelle)
            if clean_canonique:
                self.sector_index[clean_canonique] = id_secteur

            # Indexation de toutes les variations textuelles (les chaînes longues avec des tirets, etc.)
            for variation in secteur.get("variations_textuelles_associees", []):
                clean_variation = self._clean_text(variation)
                if clean_variation and clean_variation not in self.sector_index:
                    self.sector_index[clean_variation] = id_secteur

    def normalize(self, raw_sector):
        """
        Prend un secteur brut et renvoie un dictionnaire avec le secteur canonique et son ID.
        """
        fallback_meta = {
            "id_secteur": "SEC_AUTRES",
            "secteur_canonique": "Autres Secteurs / Non Spécifié"
        }

        if not raw_sector or not isinstance(raw_sector, str):
            return fallback_meta

        clean_input = self._clean_text(raw_sector)

        # 1. Recherche exacte
        if clean_input in self.sector_index:
            id_sect = self.sector_index[clean_input]
            return self.sector_metadata[id_sect]

        # 2. Recherche partielle : on garde la variation indexée LA PLUS LONGUE qui matche,
        # pour éviter qu'un terme générique ("elevage") ne masque une variation plus
        # spécifique et potentiellement plus pertinente ("elevage bovin biologique").
        best_match = None
        for indexed_variation, id_sect in self.sector_index.items():
            if len(indexed_variation) > 4 and f" {indexed_variation} " in f" {clean_input} ":
                if best_match is None or len(indexed_variation) > len(best_match[0]):
                    best_match = (indexed_variation, id_sect)
        if best_match is not None:
            return self.sector_metadata[best_match[1]]

        # 3. Recherche floue (typos, variantes non listées) via RapidFuzz
        if _RAPIDFUZZ_AVAILABLE and self.sector_index:
            match = process.extractOne(
                clean_input,
                self.sector_index.keys(),
                scorer=fuzz.WRatio,
                score_cutoff=self.FUZZY_MATCH_THRESHOLD
            )
            if match is not None:
                matched_key = match[0]
                id_sect = self.sector_index[matched_key]
                return self.sector_metadata[id_sect]

        # 4. Fallback si non trouvé
        # Plutôt que de renvoyer le texte brut qui polluerait ChromaDB avec des fautes,
        # on le classe dans "Autres" pour garder un vecteur propre.
        return fallback_meta

# === TEST D'UTILISATION ===
# normalizer = SectorNormalizer()
#
# # Test 1 : Chaîne exacte mais sale (majuscules, accents)
# print(normalizer.normalize("Agriculture - Agroalimentaire - Brasserie - Produits laitiers - Boulangerie"))
# # -> {'id_secteur': 'SEC_AGRO', 'secteur_canonique': 'Agriculture, Agroalimentaire & Élevage'}
#
# # Test 2 : Mot clé isolé
# print(normalizer.normalize("Elevage"))
# # -> {'id_secteur': 'SEC_AGRO', 'secteur_canonique': 'Agriculture, Agroalimentaire & Élevage'}

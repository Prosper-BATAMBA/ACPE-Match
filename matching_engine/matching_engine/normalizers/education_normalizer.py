import json
import os
import re
import unicodedata

try:
    from rapidfuzz import process, fuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False

from ..config import NIVEAU_ETUDE_MAPPING_PATH


class EducationNormalizer:

    # Score minimum (0-100) pour accepter une correspondance floue.
    FUZZY_MATCH_THRESHOLD = 85

    def __init__(self, mapping_file_path=None):
        """
        Constructeur : Charge le JSON, prépare l'index et trie par rang ordinal.
        """
        mapping_file_path = mapping_file_path or str(NIVEAU_ETUDE_MAPPING_PATH)
        self.education_metadata = {}  # Métadonnées (code, libelle, rang)
        self.education_index = {}     # Index inversé pour la recherche rapide
        self.levels_ordered = []      # Liste triée pour prioriser les hauts diplômes

        self._load_and_index(mapping_file_path)

    def _clean_text(self, text):
        """
        Nettoie le texte.
        Note: "Bac +5" et "Bac+5" deviendront tous les deux "bac 5",
        garantissant une correspondance parfaite.
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

        levels = data.get("education_levels", [])

        for level in levels:
            code_niveau = level.get("code_niveau", "NV_0_AUCUN")

            # 1. Sauvegarde des métadonnées
            self.education_metadata[code_niveau] = {
                "code_niveau": code_niveau,
                "libelle_standard": level.get("libelle_standard", "Non spécifié"),
                "rang_ordinal": level.get("rang_ordinal", 0)
            }

            # Fonction interne pour indexer une chaîne
            def index_string(s):
                clean_s = self._clean_text(s)
                if clean_s and clean_s not in self.education_index:
                    # On stocke aussi la longueur pour éviter les faux positifs sur les mots très courts
                    self.education_index[clean_s] = code_niveau

            # 2. Indexation du libellé standard
            index_string(level.get("libelle_standard", ""))

            # 3. Indexation des correspondances de base (Niveau d'étude)
            for corr in level.get("correspondances_niveau_etude", []):
                index_string(corr)

            # 4. Indexation des variations libres (Diplômes)
            for variation in level.get("variations_diplomes", []):
                index_string(variation)

        # 5. Astuce Algorithmique : Trier les codes par rang ordinal décroissant
        # Cela permettra au moteur de toujours vérifier si un diplôme supérieur existe dans la phrase
        self.levels_ordered = sorted(
            self.education_metadata.keys(),
            key=lambda k: self.education_metadata[k]['rang_ordinal'],
            reverse=True
        )

    def normalize(self, raw_education):
        """
        Prend une chaîne de diplôme brute et renvoie le dictionnaire normalisé
        correspondant au niveau le plus élevé trouvé.
        """
        fallback_meta = {
            "code_niveau": "NV_0_AUCUN",
            "libelle_standard": "Non spécifié / Aucun diplôme formel détecté",
            "rang_ordinal": 0
        }

        if not raw_education or not isinstance(raw_education, str):
            return fallback_meta

        clean_input = self._clean_text(raw_education)

        # 1. Recherche exacte
        if clean_input in self.education_index:
            code = self.education_index[clean_input]
            return self.education_metadata[code]

        # 2. Recherche par extraction de mots-clés (Substring Match)
        # On regroupe tous les codes trouvés dans la phrase
        codes_trouves = []
        for indexed_variation, code in self.education_index.items():
            # On ignore les correspondances de moins de 3 lettres pour éviter de matcher "bac" dans "bachelier"
            if len(indexed_variation) >= 3 and f" {indexed_variation} " in f" {clean_input} ":
                codes_trouves.append(code)

        if codes_trouves:
            # 3. Résolution de conflit : S'il a trouvé plusieurs diplômes,
            # on prend celui dont le rang_ordinal est le plus élevé.
            meilleur_code = max(
                codes_trouves,
                key=lambda c: self.education_metadata[c]['rang_ordinal']
            )
            return self.education_metadata[meilleur_code]

        # 4. Recherche floue (fautes de frappe : "maitrisee", "ingenieurr", etc.)
        # Contrairement à la recherche par mots-clés, elle compare l'entrée entière
        # à chaque variation indexée : elle est donc surtout fiable quand l'entrée
        # est déjà courte (un diplôme isolé, pas une phrase longue). Elle est
        # volontairement tentée en dernier, une fois les cas "phrase complexe"
        # écartés par l'étape 2.
        if _RAPIDFUZZ_AVAILABLE and self.education_index:
            match = process.extractOne(
                clean_input,
                self.education_index.keys(),
                scorer=fuzz.WRatio,
                score_cutoff=self.FUZZY_MATCH_THRESHOLD
            )
            if match is not None:
                matched_key = match[0]
                code = self.education_index[matched_key]
                return self.education_metadata[code]

        return fallback_meta

# === TEST D'UTILISATION ===
# normalizer = EducationNormalizer()
#
# # Test 1 : L'orthographe est mauvaise
# print(normalizer.normalize("maitrise"))
# # -> {'code_niveau': 'NV_7_BAC_5', 'libelle_standard': 'Master / Bac +4 / Bac +5 / Ingénieur', 'rang_ordinal': 7}
#
# # Test 2 : Phrase complexe (Détecte le Master au lieu du BEPC)
# print(normalizer.normalize("J'ai eu mon BEPC en 2015, et un Master 2 en Finance récemment."))
# # -> {'code_niveau': 'NV_7_BAC_5', 'libelle_standard': 'Master / Bac +4 / Bac +5 / Ingénieur', 'rang_ordinal': 7}

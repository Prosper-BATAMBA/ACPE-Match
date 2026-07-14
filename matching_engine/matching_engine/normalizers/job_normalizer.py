import json
import os
import re
import unicodedata

try:
    from rapidfuzz import process, fuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False

from ..config import JOB_MAPPING_PATH


class JobNormalizer:

    # Score minimum (0-100) pour accepter une correspondance floue.
    FUZZY_MATCH_THRESHOLD = 85

    def __init__(self, mapping_file_path=None):
        mapping_file_path = mapping_file_path or str(JOB_MAPPING_PATH)
        self.family_metadata = {}  # id_famille -> {famille_fr, famille_en, competences_cles_fr, competences_cles_en}
        self.job_to_family = {}    # nom canonique -> id_famille
        self.job_index = {}        # texte nettoyé -> nom canonique (recherche O(1))

        self._load_and_index(mapping_file_path)

    def _clean_text(self, text):
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

        clusters = data.get("metiers_clusters", [])

        for cluster in clusters:
            id_famille = cluster.get("id_famille", "INCONNU")

            # Les métadonnées de famille ne sont stockées qu'une seule fois par id_famille,
            # au lieu d'être dupliquées pour chaque métier qui y appartient.
            if id_famille not in self.family_metadata:
                self.family_metadata[id_famille] = {
                    "id_famille": id_famille,
                    "famille_fr": cluster.get("famille", "Autre"),
                    "famille_en": cluster.get("famille_en", "Other"),
                    "competences_cles_fr": cluster.get("competences_cles", []),
                    "competences_cles_en": cluster.get("competences_cles_en", []),
                }

            tous_les_metiers = (
                cluster.get("metiers_base_donnees", []) +
                cluster.get("metiers_extensions", []) +
                cluster.get("synonymes_et_variantes", [])
            )

            for job in tous_les_metiers:
                clean_job = self._clean_text(job)
                if clean_job and clean_job not in self.job_index:
                    canonical_name = job.strip().title()
                    self.job_to_family[canonical_name] = id_famille
                    self.job_index[clean_job] = canonical_name

    def _build_result(self, canonical_name):
        id_famille = self.job_to_family.get(canonical_name, "FAM_AUTRE")
        meta = self.family_metadata.get(id_famille, {
            "id_famille": "FAM_AUTRE", "famille_fr": "Autre", "famille_en": "Other",
            "competences_cles_fr": [], "competences_cles_en": []
        })
        return {"metier_canonique": canonical_name, **meta}

    def normalize(self, raw_job_title):
        """
        Prend un intitulé brut et renvoie un dictionnaire avec le métier canonique et ses métadonnées.
        """
        fallback_meta = {
            "id_famille": "FAM_AUTRE", "famille_fr": "Autre", "famille_en": "Other",
            "competences_cles_fr": [], "competences_cles_en": []
        }

        if not raw_job_title or not isinstance(raw_job_title, str):
            return {"metier_canonique": "Non spécifié", **fallback_meta}

        clean_input = self._clean_text(raw_job_title)

        # 1. Recherche exacte
        if clean_input in self.job_index:
            return self._build_result(self.job_index[clean_input])

        # 2. Recherche partielle : on garde le terme indexé le PLUS LONG qui matche,
        # pour éviter que "developpeur" ne masque "developpeur python".
        best_match = None
        for indexed_job, canonical in self.job_index.items():
            if len(indexed_job) > 4 and f" {indexed_job} " in f" {clean_input} ":
                if best_match is None or len(indexed_job) > len(best_match[0]):
                    best_match = (indexed_job, canonical)
        if best_match is not None:
            return self._build_result(best_match[1])

        # 3. Recherche floue (typos, variantes non listées) via RapidFuzz
        if _RAPIDFUZZ_AVAILABLE and self.job_index:
            match = process.extractOne(
                clean_input,
                self.job_index.keys(),
                scorer=fuzz.WRatio,
                score_cutoff=self.FUZZY_MATCH_THRESHOLD
            )
            if match is not None:
                matched_key = match[0]
                return self._build_result(self.job_index[matched_key])

        # 4. Fallback si non trouvé
        return {"metier_canonique": raw_job_title.strip().title(), **fallback_meta}

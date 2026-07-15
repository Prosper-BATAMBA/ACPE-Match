import json
import os
import re
import unicodedata

try:
    from rapidfuzz import process, fuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False

from ..config import JOB_MAPPING_PATH, SKILLS_MAPPING_PATH


class JobNormalizer:

    # Score minimum (0-100) pour accepter une correspondance floue.
    FUZZY_MATCH_THRESHOLD = 85

    def __init__(self, mapping_file_path=None, skills_file_path=None):
        mapping_file_path = mapping_file_path or str(JOB_MAPPING_PATH)
        skills_file_path = skills_file_path or str(SKILLS_MAPPING_PATH)

        self.family_metadata = {}      # id_famille -> {famille_fr, famille_en, competences_cles_fr/en}
        self.subfamily_metadata = {}   # id_sous_famille -> {sous_famille_fr, sous_famille_en, competences_cles_fr/en}
        self.job_to_family = {}        # nom canonique -> id_famille
        self.job_to_subfamily = {}     # nom canonique -> id_sous_famille (absent si le métier n'a pas été affiné)
        self.job_index = {}            # texte nettoyé -> nom canonique (recherche O(1))
        self.skill_ids_by_subfamily = {}  # id_sous_famille -> [(id_skill, libelle_fr), ...]

        self._load_and_index(mapping_file_path)
        self._load_skills(skills_file_path)

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

            # Index métier -> sous-famille, uniquement pour les métiers qui ont pu être
            # affinés au-delà de la famille générique (voir job_mapping_enriched.json).
            # Le libellé "..._GENERIQUE" désigne le seau fourre-tout d'une famille découpée :
            # on ne le traite pas comme une correspondance "spécifique".
            for sous_famille in cluster.get("sous_familles", []):
                id_sf = sous_famille.get("id_sous_famille")
                if id_sf and id_sf not in self.subfamily_metadata:
                    self.subfamily_metadata[id_sf] = {
                        "id_sous_famille": id_sf,
                        "id_famille": id_famille,
                        "sous_famille_fr": sous_famille.get("sous_famille", ""),
                        "sous_famille_en": sous_famille.get("sous_famille_en", ""),
                        "competences_cles_fr": sous_famille.get("competences_cles", []),
                        "competences_cles_en": sous_famille.get("competences_cles_en", []),
                        "est_generique": id_sf.endswith("_GENERIQUE"),
                    }
                for job in sous_famille.get("metiers", []):
                    clean_job = self._clean_text(job)
                    if clean_job:
                        canonical_name = job.strip().title()
                        # Un job peut apparaître dans metiers_extensions ET dans une
                        # sous-famille : on privilégie la sous-famille non générique
                        # si plusieurs candidats existent pour le même métier.
                        existing = self.job_to_subfamily.get(canonical_name)
                        if existing is None or (self.subfamily_metadata.get(existing, {}).get("est_generique")
                                                 and not id_sf.endswith("_GENERIQUE")):
                            self.job_to_subfamily[canonical_name] = id_sf

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

    def _load_skills(self, filepath):
        """
        Charge skills_enriched.json (facultatif : si absent, le normalizer continue
        de fonctionner mais sans renvoyer de skill_ids précis, uniquement les
        libellés de compétences du job_mapping).
        """
        if not filepath or not os.path.exists(filepath):
            return

        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for categorie in data.get("skills_taxonomy", []):
            for competence in categorie.get("competences", []):
                sous_famille_associee = competence.get("sous_famille_associee")
                if sous_famille_associee:
                    self.skill_ids_by_subfamily.setdefault(sous_famille_associee, []).append(
                        (competence["id_skill"], competence["libelle_canonique"])
                    )

    def _build_result(self, canonical_name):
        id_famille = self.job_to_family.get(canonical_name, "FAM_AUTRE")
        famille_meta = self.family_metadata.get(id_famille, {
            "id_famille": "FAM_AUTRE", "famille_fr": "Autre", "famille_en": "Other",
            "competences_cles_fr": [], "competences_cles_en": []
        })

        id_sf = self.job_to_subfamily.get(canonical_name)
        sf_meta = self.subfamily_metadata.get(id_sf) if id_sf else None

        if sf_meta and not sf_meta.get("est_generique"):
            # Niveau "spécifique" : compétences de la sous-famille + skill_ids précis
            competences_cles_fr = sf_meta["competences_cles_fr"]
            competences_cles_en = sf_meta["competences_cles_en"]
            skill_ids = self.skill_ids_by_subfamily.get(id_sf, [])
            niveau_confiance = "specifique"
            sous_famille_fr = sf_meta["sous_famille_fr"]
            sous_famille_en = sf_meta["sous_famille_en"]
            id_sous_famille = id_sf
        else:
            # Niveau "générique" : compétences de la famille uniquement (pas de skill_ids
            # dédiés, la famille n'a pas de sous-famille associée ou le métier tombe
            # dans le seau générique)
            competences_cles_fr = famille_meta["competences_cles_fr"]
            competences_cles_en = famille_meta["competences_cles_en"]
            skill_ids = []
            niveau_confiance = "generique"
            sous_famille_fr = ""
            sous_famille_en = ""
            id_sous_famille = None

        return {
            "metier_canonique": canonical_name,
            "id_famille": famille_meta["id_famille"],
            "famille_fr": famille_meta["famille_fr"],
            "famille_en": famille_meta["famille_en"],
            "id_sous_famille": id_sous_famille,
            "sous_famille_fr": sous_famille_fr,
            "sous_famille_en": sous_famille_en,
            "competences_cles_fr": competences_cles_fr,
            "competences_cles_en": competences_cles_en,
            "skill_ids": skill_ids,
            "niveau_confiance": niveau_confiance,
        }

    def normalize(self, raw_job_title):
        """
        Prend un intitulé brut et renvoie un dictionnaire avec le métier canonique et ses métadonnées
        (famille, sous-famille si reconnue précisément, compétences clés, skill_ids, niveau de confiance).
        """
        fallback = {
            "metier_canonique": "Non spécifié", "id_famille": "FAM_AUTRE",
            "famille_fr": "Autre", "famille_en": "Other",
            "id_sous_famille": None, "sous_famille_fr": "", "sous_famille_en": "",
            "competences_cles_fr": [], "competences_cles_en": [], "skill_ids": [],
            "niveau_confiance": "non_classifie",
        }

        if not raw_job_title or not isinstance(raw_job_title, str):
            return fallback

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
        result = dict(fallback)
        result["metier_canonique"] = raw_job_title.strip().title()
        return result

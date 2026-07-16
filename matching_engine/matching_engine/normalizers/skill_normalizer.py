import json
import os
import re
import unicodedata

try:
    from rapidfuzz import process, fuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False

from ..config import SKILLS_MAPPING_PATH


class SkillNormalizer:

    FUZZY_MATCH_THRESHOLD = 85

    def __init__(self, mapping_file_path=None):
        mapping_file_path = mapping_file_path or str(SKILLS_MAPPING_PATH)
        self.skill_metadata = {}
        self.skill_index = {}

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

        taxonomies = data.get("skills_taxonomy", [])

        for category_group in taxonomies:
            categorie = category_group.get("categorie", "Non catégorisé")

            for comp in category_group.get("competences", []):
                id_skill = comp.get("id_skill")
                libelle_canonique = comp.get("libelle_canonique", "")
                type_skill = comp.get("type", "Hard Skill")

                self.skill_metadata[id_skill] = {
                    "id_skill": id_skill,
                    "libelle_canonique": libelle_canonique,
                    "categorie": categorie,
                    "type": type_skill
                }

                def index_keyword(kw):
                    clean_kw = self._clean_text(kw)
                    if len(clean_kw) >= 2 and clean_kw not in self.skill_index:
                        self.skill_index[clean_kw] = id_skill

                index_keyword(libelle_canonique)

                for keyword in comp.get("mots_cles_extraction", []):
                    index_keyword(keyword)

    def normalize_list(self, raw_skills_list):
        if not raw_skills_list or not isinstance(raw_skills_list, list):
            return []

        detected_skills = {}

        for raw_skill in raw_skills_list:
            clean_skill = self._clean_text(raw_skill)
            if not clean_skill:
                continue

            if clean_skill in self.skill_index:
                id_skill = self.skill_index[clean_skill]
                detected_skills[id_skill] = self.skill_metadata[id_skill]
                continue

            found_in_this_item = False
            for indexed_kw, id_skill in self.skill_index.items():
                if len(indexed_kw) >= 2 and f" {indexed_kw} " in f" {clean_skill} ":
                    detected_skills[id_skill] = self.skill_metadata[id_skill]
                    found_in_this_item = True

            if not found_in_this_item and _RAPIDFUZZ_AVAILABLE and self.skill_index:
                match = process.extractOne(
                    clean_skill,
                    self.skill_index.keys(),
                    scorer=fuzz.WRatio,
                    score_cutoff=self.FUZZY_MATCH_THRESHOLD
                )
                if match is not None:
                    matched_key = match[0]
                    id_skill = self.skill_index[matched_key]
                    detected_skills[id_skill] = self.skill_metadata[id_skill]

        return list(detected_skills.values())

    def extract_from_text(self, text_block):
        """Extract skills from a text block.

        Uses exact/substring matching first, then falls back to word-level
        fuzzy matching for individual tokens to catch typos (e.g. "Exel" -> "Excel").
        """
        if not text_block or not isinstance(text_block, str):
            return []

        clean_text = f" {self._clean_text(text_block)} "
        detected_skills = {}

        for indexed_kw, id_skill in self.skill_index.items():
            if f" {indexed_kw} " in clean_text:
                detected_skills[id_skill] = self.skill_metadata[id_skill]

        if not _RAPIDFUZZ_AVAILABLE:
            return list(detected_skills.values())

        tokens = [t for t in clean_text.split() if len(t) >= 3]
        if not tokens:
            return list(detected_skills.values())

        for token in tokens:
            match = process.extractOne(
                token,
                self.skill_index.keys(),
                scorer=fuzz.ratio,
                score_cutoff=self.FUZZY_MATCH_THRESHOLD
            )
            if match is not None:
                matched_key = match[0]
                id_skill = self.skill_index[matched_key]
                detected_skills[id_skill] = self.skill_metadata[id_skill]

        return list(detected_skills.values())

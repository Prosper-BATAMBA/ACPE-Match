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

    # Score minimum (0-100) pour accepter une correspondance floue.
    FUZZY_MATCH_THRESHOLD = 85

    def __init__(self, mapping_file_path=None):
        """
        Constructeur : Charge la taxonomie des compétences et prépare l'index de recherche.
        """
        mapping_file_path = mapping_file_path or str(SKILLS_MAPPING_PATH)
        self.skill_metadata = {}  # Stocke les métadonnées de la compétence (catégorie, type, libellé)
        self.skill_index = {}     # Index inversé pour la recherche (mot-clé -> id_skill)

        self._load_and_index(mapping_file_path)

    def _clean_text(self, text):
        """
        Nettoie le texte : minuscules, sans accents, espaces normalisés.
        """
        if not text or not isinstance(text, str):
            return ""

        text = text.lower()
        text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
        # On garde les lettres et les chiffres
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

                # 1. Sauvegarde des métadonnées de cette compétence (une seule fois par id_skill)
                self.skill_metadata[id_skill] = {
                    "id_skill": id_skill,
                    "libelle_canonique": libelle_canonique,
                    "categorie": categorie,
                    "type": type_skill
                }

                # Fonction interne pour indexer
                def index_keyword(kw):
                    clean_kw = self._clean_text(kw)
                    # On ignore les mots-clés trop courts (moins de 2 lettres)
                    if len(clean_kw) >= 2 and clean_kw not in self.skill_index:
                        self.skill_index[clean_kw] = id_skill

                # 2. Indexer le libellé canonique lui-même
                index_keyword(libelle_canonique)

                # 3. Indexer tous les mots-clés d'extraction ("JS", "OHADA", etc.)
                for keyword in comp.get("mots_cles_extraction", []):
                    index_keyword(keyword)

    def normalize_list(self, raw_skills_list):
        """
        Prend une liste de compétences brutes (ex: ["JS", "Excel", "Travail d'équipe"])
        et renvoie la liste des métadonnées normalisées sans doublons.
        """
        if not raw_skills_list or not isinstance(raw_skills_list, list):
            return []

        detected_skills = {}  # Dictionnaire pour éviter les doublons (id_skill -> metadata)

        for raw_skill in raw_skills_list:
            clean_skill = self._clean_text(raw_skill)
            if not clean_skill:
                continue

            # 1. Recherche exacte
            if clean_skill in self.skill_index:
                id_skill = self.skill_index[clean_skill]
                detected_skills[id_skill] = self.skill_metadata[id_skill]
                continue

            # 2. Recherche partielle (substring) : on garde TOUS les mots-clés qui
            # matchent, pas un seul — un item de la liste peut être une phrase
            # ("Je connais bien les normes OHADA") qui contient plusieurs compétences.
            found_in_this_item = False
            for indexed_kw, id_skill in self.skill_index.items():
                # On s'assure d'avoir des mots entiers (ex: "js" ne doit pas matcher dans "majuscule")
                if len(indexed_kw) >= 2 and f" {indexed_kw} " in f" {clean_skill} ":
                    detected_skills[id_skill] = self.skill_metadata[id_skill]
                    found_in_this_item = True

            # 3. Recherche floue — seulement si rien n'a matché en exact/substring.
            # Contrairement à extract_from_text (bloc de texte libre), chaque élément
            # de raw_skills_list est court et isolé (un item de liste de CV), donc
            # comparer toute la chaîne à chaque mot-clé indexé a du sens ici :
            # ça rattrape "Exel" -> "Excel", "Pyton" -> "Python", etc.
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
        """
        Scanne un bloc de texte entier (ex: description d'une offre ou CV complet)
        et extrait toutes les compétences trouvées.

        Note : pas de recherche floue ici volontairement. RapidFuzz compare une
        chaîne entière à des mots-clés courts ; appliqué à un paragraphe complet,
        ça ne détecterait pas une faute de frappe isolée au milieu du texte et
        risquerait plutôt de renvoyer des faux positifs. Le fuzzy matching n'est
        fiable que sur des termes courts et isolés, comme dans normalize_list.
        """
        if not text_block or not isinstance(text_block, str):
            return []

        clean_text = f" {self._clean_text(text_block)} "
        detected_skills = {}

        for indexed_kw, id_skill in self.skill_index.items():
            if f" {indexed_kw} " in clean_text:
                detected_skills[id_skill] = self.skill_metadata[id_skill]

        return list(detected_skills.values())

# === TEST D'UTILISATION ===
# normalizer = SkillNormalizer()
#
# # Test 1 : Liste de compétences brutes du CV
# cv_skills = ["JS", "Excel", "Je connais bien les normes OHADA", "aisance relationnelle"]
# print("Test Liste:")
# for s in normalizer.normalize_list(cv_skills):
#     print(f" - {s['libelle_canonique']} ({s['type']})")
#
# # Test 2 : Extraction depuis un paragraphe
# description_offre = "Nous cherchons un profil maitrisant la comptabilité générale et la liasse fiscale."
# print("\nTest Extraction:")
# for s in normalizer.extract_from_text(description_offre):
#     print(f" - {s['libelle_canonique']} ({s['categorie']})")

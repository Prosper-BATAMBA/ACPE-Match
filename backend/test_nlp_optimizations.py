"""Quick smoke test for NLP optimizations."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.feature_extractor import extract_features, tokenize_french, FRENCH_STOPWORDS

# Test stopwords
tokens = tokenize_french("developpeur informatique dans le secteur de la sante")
print(f"Tokens: {tokens}")
print(f"Stopwords count: {len(FRENCH_STOPWORDS)}")
assert "le" in FRENCH_STOPWORDS
assert "dans" in FRENCH_STOPWORDS
assert "de" in FRENCH_STOPWORDS
print("Stopwords OK")

# Test feature count
feat = extract_features(
    {"id_famille": "FAM_IT", "id_secteur": "SEC_IT", "code_departement": "BZV",
     "code_niveau_etude": "NV_6_BAC_3", "age": 28, "mobilite": "oui",
     "profile_text": "dev python", "metier_vise": "dev", "genre": "homme",
     "specialite": "informatique", "qualification": "bac+3",
     "secteur_demande": "informatique", "_spec_result": {}},
    {"id_famille": "FAM_IT", "id_secteur": "SEC_IT", "code_departement": "BZV",
     "profile_text": "offre dev", "intitule": "dev python", "description": "dev",
     "competences_recherchees": "python", "type_contrat": "CDI",
     "secteur": "informatique"},
    0.8,
    cand_skills=[{"libelle_canonique": "Python", "id_skill": "SKILL_IT_PYTHON"}],
    offer_skills=[{"libelle_canonique": "Python", "id_skill": "SKILL_IT_PYTHON"}],
)
print(f"Feature count: {len(feat)}")
assert len(feat) >= 40, f"Expected >=40 features, got {len(feat)}"
print("Feature extraction OK")

# Test cached normalizers
from app.services.matching_engine_service import _get_skill_normalizer, _get_specialty_normalizer, _get_sector_normalizer
sn1 = _get_skill_normalizer()
sn2 = _get_skill_normalizer()
assert sn1 is sn2, "SkillNormalizer not cached"
print("Cached normalizers OK")

# Test embedding truncation
from app.services.embedding_service import _truncate_text, _MAX_CHARS
short = "hello"
long = "x" * 5000
assert _truncate_text(short) == short
assert len(_truncate_text(long)) == _MAX_CHARS
print("Truncation OK")

# Test shared module imports in train_ranker
from app.services.feature_extractor import extract_features, get_graphs
jg, sg, spg = get_graphs()
assert isinstance(jg, dict)
print("Graphs load OK")

print("\n=== ALL NLP OPTIMIZATION TESTS PASSED ===")

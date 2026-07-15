from sqlalchemy import Column, String, Text, Integer

from ..database import Base


class Candidate(Base):
    __tablename__ = "candidates"

    id = Column(String, primary_key=True, index=True)
    nom = Column(String, nullable=True)
    prenom = Column(String, nullable=True)
    genre = Column(String, nullable=True)
    age = Column(Integer, nullable=True)
    lieu = Column(String, nullable=True)
    etudes = Column(String, nullable=True)
    qualification = Column(String, nullable=True)
    specialite = Column(String, nullable=True)
    secteur_demande = Column(String, nullable=True)
    metier_vise = Column(String, nullable=True)
    mobilite = Column(String, nullable=True)
    competences_brutes = Column(Text, nullable=True)
    experience_libre = Column(Text, nullable=True)

    id_famille = Column(String, nullable=True)
    id_sous_famille = Column(String, nullable=True)
    id_secteur = Column(String, nullable=True)
    code_niveau_etude = Column(String, nullable=True)
    code_departement = Column(String, nullable=True)
    profile_text = Column(Text, nullable=True)

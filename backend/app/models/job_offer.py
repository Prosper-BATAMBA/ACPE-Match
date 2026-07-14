from sqlalchemy import Column, String, Text, Integer

from ..database import Base


class JobOffer(Base):
    __tablename__ = "job_offers"

    id = Column(String, primary_key=True, index=True)
    intitule = Column(String, nullable=True)
    type_contrat = Column(String, nullable=True)
    entreprise = Column(String, nullable=True)
    secteur = Column(String, nullable=True)
    localisation = Column(String, nullable=True)
    date_publication = Column(String, nullable=True)
    date_cloture = Column(String, nullable=True)
    description = Column(Text, nullable=True)
    competences_recherchees = Column(Text, nullable=True)

    id_famille = Column(String, nullable=True)
    id_secteur = Column(String, nullable=True)
    code_departement = Column(String, nullable=True)
    profile_text = Column(Text, nullable=True)

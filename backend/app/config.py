import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./acpe.db",
)

CHROMADB_PERSIST_DIR = os.getenv("CHROMADB_PERSIST_DIR", "./chroma_data")

SENTENCE_TRANSFORMER_MODEL = os.getenv(
    "SENTENCE_TRANSFORMER_MODEL",
    "BAAI/bge-m3",
)



import os

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./acpe.db",
)

CHROMADB_HOST = os.getenv("CHROMADB_HOST", "localhost")
CHROMADB_PORT = int(os.getenv("CHROMADB_PORT", "8000"))
CHROMADB_PERSIST_DIR = os.getenv("CHROMADB_PERSIST_DIR", "./chroma_data")

SENTENCE_TRANSFORMER_MODEL = os.getenv(
    "SENTENCE_TRANSFORMER_MODEL",
    "BAAI/bge-m3",
)

MATCHING_ALPHA = float(os.getenv("MATCHING_ALPHA", "0.7"))
MATCHING_BETA = float(os.getenv("MATCHING_BETA", "0.3"))

CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "BAAI/bge-reranker-v2-m3",
)
CROSS_ENCODER_TOP_K = int(os.getenv("CROSS_ENCODER_TOP_K", "20"))
SCORE_THRESHOLD = float(os.getenv("SCORE_THRESHOLD", "0.3"))
FAISS_TOP_K_RETRIEVAL = int(os.getenv("FAISS_TOP_K_RETRIEVAL", "200"))

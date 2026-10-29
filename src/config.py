import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "indexes"

# --- API ---
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- Индексация ---
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
EMBED_BATCH_SIZE = 32
FAISS_NLIST = 256


# --- LLM ---
LLM_TEMPERATURE = 0.1

# --- Эмбеддинги (локальные) ---
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"   # ← SMALL
EMBEDDING_DIM = 384                                   # ← 384 вместо 768
EMBEDDING_BATCH_SIZE = 32                             # ← можно больше для small
EMBEDDING_PREFIX_DOC = "passage: "
EMBEDDING_PREFIX_QUERY = "query: "

# --- FAISS IVFPQ ---
FAISS_NLIST = 1024
FAISS_M = 48       # ← 384 / 48 = 8 dims per subspace
FAISS_NBITS = 8        # 8 бит на код → 256 центроидов

# --- BM25 ---
BM25_TOP_K = 10
BM25_MAX_CHARS = 1000   # обрезаем текст для BM25 (экономия RAM в 5–10 раз)

# --- Retrieval ---
VECTOR_TOP_K = 10
FUSION_WEIGHTS = [0.5, 0.5]
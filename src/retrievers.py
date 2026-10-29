import pickle
import faiss
from functools import lru_cache

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from src.config import INDEX_DIR, VECTOR_TOP_K, BM25_TOP_K, FUSION_WEIGHTS
from src.llm import get_embeddings


@lru_cache(maxsize=3)
def _load_faiss_store(source: str) -> FAISS:
    """Загружает FAISS IVFPQ и восстанавливает docstore из chunks.pkl."""
    index_path = INDEX_DIR / source / "faiss.index"
    chunks_path = INDEX_DIR / source / "chunks.pkl"

    index = faiss.read_index(str(index_path))
    index.nprobe = 16

    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(chunks)})
    index_to_docstore_id = {i: str(i) for i in range(len(chunks))}

    return FAISS(
        embedding_function=get_embeddings(),
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )


@lru_cache(maxsize=3)
def _load_bm25(source: str) -> BM25Retriever:
    with open(INDEX_DIR / source / "bm25.pkl", "rb") as f:
        bm25 = pickle.load(f)
    bm25.k = BM25_TOP_K
    return bm25


@lru_cache(maxsize=3)
def get_hybrid_retriever(source: str) -> EnsembleRetriever:
    """BM25 + векторный поиск через Reciprocal Rank Fusion."""
    vector_retriever = _load_faiss_store(source).as_retriever(
        search_kwargs={"k": VECTOR_TOP_K}
    )
    bm25 = _load_bm25(source)
    return EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=FUSION_WEIGHTS,
    )
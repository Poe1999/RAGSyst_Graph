"""
Построение и сохранение индексов FAISS (IVFPQ) + BM25.
Используется в scripts/build_index.py.
"""
import gc
import pickle
import time
import faiss
import numpy as np
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever

from src.config import (
    INDEX_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP,
    EMBEDDING_BATCH_SIZE,
    FAISS_NLIST, FAISS_M, FAISS_NBITS,
    BM25_TOP_K, BM25_MAX_CHARS,
)


def ensure_dirs(sources=("habr", "lenta", "arxiv")):
    for src in sources:
        (INDEX_DIR / src).mkdir(parents=True, exist_ok=True)


def split_documents(docs: list[Document], source: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for c in chunks:
        c.metadata["source"] = source
    return chunks


def embed_in_batches(embeddings, texts: list[str], source: str) -> np.ndarray:
    """Считаем эмбеддинги батчами с прогресс-баром."""
    vectors = []
    n_batches = (len(texts) + EMBEDDING_BATCH_SIZE - 1) // EMBEDDING_BATCH_SIZE

    for i in tqdm(
        range(0, len(texts), EMBEDDING_BATCH_SIZE),
        total=n_batches,
        desc=f"[{source}] embed",
    ):
        batch = texts[i:i + EMBEDDING_BATCH_SIZE]
        vecs = embeddings.embed_documents(batch)
        vectors.extend(vecs)

        if i % (EMBEDDING_BATCH_SIZE * 50) == 0:
            gc.collect()

    return np.array(vectors, dtype="float32")


def build_faiss_ivfpq(vectors: np.ndarray) -> faiss.Index:
    """
    IVFPQ: сжатие векторов, экономия RAM.
    Для маленьких датасетов (тесты) — fallback на IndexFlatL2.
    """
    d = vectors.shape[1]
    n = vectors.shape[0]

    MIN_POINTS_FOR_IVF = 10_000

    if n < MIN_POINTS_FOR_IVF:
        print(f" {n} векторов < {MIN_POINTS_FOR_IVF}, используем IndexFlatL2 (точный поиск)")
        index = faiss.IndexFlatL2(d)
        index.add(vectors)
        return index

    nlist = min(FAISS_NLIST, max(64, n // 40))
    if n < nlist * 39:
        nlist = max(1, n // 39)

    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(quantizer, d, nlist, FAISS_M, FAISS_NBITS)

    if hasattr(faiss, "StandardGpuResources"):
        try:
            res = faiss.StandardGpuResources()
            gpu = faiss.index_cpu_to_gpu(res, 0, index)
            print(f"  → FAISS IVFPQ на GPU (nlist={nlist}, M={FAISS_M}, nbits={FAISS_NBITS})")
            gpu.train(vectors)
            gpu.add(vectors)
            index = faiss.index_gpu_to_cpu(gpu)
            del gpu, res
            gc.collect()
        except Exception as e:
            print(f"  ⚠ GPU упал ({e}), переключаемся на CPU")
            index.train(vectors)
            index.add(vectors)
    else:
        print(f"  → FAISS IVFPQ на CPU (nlist={nlist}, M={FAISS_M}, nbits={FAISS_NBITS})")
        index.train(vectors)
        index.add(vectors)

    index.nprobe = 16
    return index


def build_and_save(source: str, docs: list[Document], embeddings) -> None:
    print(f"\n{'=' * 60}")
    print(f"[{source}] {len(docs)} документов")
    print(f"{'=' * 60}")
    t0 = time.time()

    # 1) Чанкинг
    print(f"[{source}] чанкинг...")
    chunks = split_documents(docs, source)
    print(f"[{source}] {len(chunks)} чанков")
    del docs
    gc.collect()

    # 2) BM25 — по обрезанным текстам (экономия RAM)
    print(f"[{source}] BM25 (по первым {BM25_MAX_CHARS} симв.)...")
    bm25_chunks = [
        Document(
            page_content=c.page_content[:BM25_MAX_CHARS],
            metadata=c.metadata,
        )
        for c in chunks
    ]
    bm25 = BM25Retriever.from_documents(bm25_chunks, k=BM25_TOP_K)
    with open(INDEX_DIR / source / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    del bm25_chunks, bm25
    gc.collect()

    # 3) Эмбеддинги (локальные)
    print(f"[{source}] считаем эмбеддинги...")
    texts = [c.page_content for c in chunks]
    vectors = embed_in_batches(embeddings, texts, source)
    print(f"[{source}] векторы: {vectors.shape}")
    del texts
    gc.collect()

    # 4) FAISS IVFPQ
    print(f"[{source}] строим FAISS...")
    index = build_faiss_ivfpq(vectors)
    faiss.write_index(index, str(INDEX_DIR / source / "faiss.index"))
    del vectors, index
    gc.collect()

    # 5) Сохраняем чанки (для восстановления текста при поиске)
    print(f"[{source}] сохраняем chunks.pkl...")
    with open(INDEX_DIR / source / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    del chunks
    gc.collect()

    dt = time.time() - t0
    print(f"[{source}] ✅ готово за {dt / 60:.1f} мин")
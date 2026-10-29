"""
Локальные эмбеддинги через ONNX с int8-квантованием.
Модель грузится напрямую из HF-кэша (int8 уже скачан, ~120 MB).
"""
import os
CPU_COUNT = os.cpu_count() or 4
os.environ.setdefault("OMP_NUM_THREADS", str(CPU_COUNT))
os.environ.setdefault("ORT_NUM_THREADS", str(CPU_COUNT))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from src.config import (
    EMBEDDING_MODEL, EMBEDDING_PREFIX_DOC, EMBEDDING_PREFIX_QUERY,)

ONNX_INT8_FILE = "onnx/model_qint8_avx512_vnni.onnx"


class LocalEmbeddings(Embeddings):
    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        max_seq_length: int = 224,
        use_int8: bool = True,
    ):
        print(f"[embeddings] загружаем {model_name} (ONNX, int8={use_int8})...")

        if use_int8:
            self.model = SentenceTransformer(
                model_name,
                backend="onnx",
                model_kwargs={"file_name": ONNX_INT8_FILE},
            )
        else:
            try:
                self.model = SentenceTransformer(model_name, backend="onnx")
            except Exception as e:
                print(f"[embeddings] ⚠ ONNX недоступен ({e}), PyTorch")
                self.model = SentenceTransformer(model_name)

        self.model.max_seq_length = max_seq_length
        self.model.eval()

        try:
            dim = self.model.get_embedding_dimension()
        except AttributeError:
            dim = self.model.get_sentence_embedding_dimension()
        print(f"[embeddings] готово, dim={dim}, max_seq={max_seq_length}")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        prefixed = [f"{EMBEDDING_PREFIX_DOC}{t}" for t in texts]
        vecs = self.model.encode(
            prefixed,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        vec = self.model.encode(
            f"{EMBEDDING_PREFIX_QUERY}{text}",
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.tolist()
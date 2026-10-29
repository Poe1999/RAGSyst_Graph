import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import DATA_DIR
from src.indexing import ensure_dirs, build_and_save
from src.llm import get_embeddings
from src.data_loader import load_habr, load_lenta, load_arxiv


def main():
    ensure_dirs()
    emb = get_embeddings()

    habr_path = DATA_DIR / "10_000-постов-Хабр.csv"
    if habr_path.exists():
        print(f"Найден Habr: {habr_path.name}")
        build_and_save("habr", load_habr(habr_path), emb)
    else:
        print(f"Habr НЕ найден: {habr_path}")

    lenta_path = DATA_DIR / "10_000-постов-лента.csv"
    if lenta_path.exists():
        print(f"Найден Lenta: {lenta_path.name}")
        build_and_save("lenta", load_lenta(lenta_path), emb)
    else:
        print(f"Lenta НЕ найден: {lenta_path}")

    #arxiv_dir = DATA_DIR / "arxiv2"
    #if arxiv_dir.exists():
    #    print(f"Найден arXiv: {arxiv_dir}")
    #    build_and_save("arxiv", load_arxiv(arxiv_dir), emb)
    #else:
    #    print(f"arXiv НЕ найден: {arxiv_dir}")

    print("\nИндексация завершена. Индексы в:", DATA_DIR.parent / "indexes")


if __name__ == "__main__":
    main()
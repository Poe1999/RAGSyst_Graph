"""
Загрузка документов из трёх источников:
  - Habr  (CSV: post_id, post_hubs, post_tags, post_publish_date,
                post_title, post_html, post_author)
  - Lenta (CSV: url, title, text, topic, tags, date)
  - arXiv (JSON: {meta: {202: title, 81: abstract, 217: url, 236: id, ...}})
          + TXT файлы
"""
import json
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
from langchain_core.documents import Document

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

def _clean_html(html: str) -> str:
    """Удаляем HTML-теги и лишние пробелы."""
    if not isinstance(html, str):
        return ""
    soup = BeautifulSoup(html, "lxml")
    text = soup.get_text(separator=" ")
    return _WS_RE.sub(" ", text).strip()


def _safe_str(x) -> str:
    """Приводим значение к строке, чистим NaN/None."""
    if x is None:
        return ""
    if isinstance(x, float) and pd.isna(x):
        return ""
    return str(x).strip()


def load_habr(path: Path, limit: int | None = None) -> list[Document]:
    """
    Habr CSV. Основной текст — в post_html, чистим.
    Метаданные: post_id, хабы, теги, дата, автор.
    """
    print(f"[habr] читаем {path.name}...")
    df = pd.read_csv(path)
    print(f"[habr] колонки: {df.columns.tolist()}")

    if limit:
        df = df.head(limit)

    docs: list[Document] = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="habr"):
        title = _safe_str(getattr(row, "post_title", ""))
        body = _clean_html(getattr(row, "post_html", ""))
        if not title and not body:
            continue

        content = f"{title}\n\n{body}".strip()

        docs.append(Document(
            page_content=content,
            metadata={
                "source": "habr",
                "post_id": _safe_str(getattr(row, "post_id", "")),
                "title": title,
                "author": _safe_str(getattr(row, "post_author", "")),
                "date": _safe_str(getattr(row, "post_publish_date", "")),
                "hubs": _safe_str(getattr(row, "post_hubs", "")),
                "tags": _safe_str(getattr(row, "post_tags", "")),
                "url": f"https://habr.com/ru/post/{_safe_str(getattr(row, 'post_id', ''))}/",
            },
        ))
    print(f"[habr] загружено {len(docs)} документов")
    return docs


def load_lenta(path: Path, limit: int | None = None) -> list[Document]:
    """
    Lenta CSV. Колонки: url, title, text, topic, tags, date.
    """
    print(f"[lenta] читаем {path.name}...")
    df = pd.read_csv(path)
    print(f"[lenta] колонки: {df.columns.tolist()}")

    if limit:
        df = df.head(limit)

    docs: list[Document] = []
    for row in tqdm(df.itertuples(index=False), total=len(df), desc="lenta"):
        title = _safe_str(getattr(row, "title", ""))
        text = _safe_str(getattr(row, "text", ""))
        if not title and not text:
            continue

        content = f"{title}\n\n{text}".strip()

        docs.append(Document(
            page_content=content,
            metadata={
                "source": "lenta",
                "title": title,
                "url": _safe_str(getattr(row, "url", "")),
                "date": _safe_str(getattr(row, "date", "")),
                "topic": _safe_str(getattr(row, "topic", "")),
                "tags": _safe_str(getattr(row, "tags", "")),
            },
        ))
    print(f"[lenta] загружено {len(docs)} документов")
    return docs

def _parse_arxiv_json(data: dict, filename: str) -> Document | None:
    """
    Структура: {"meta": {"202": title, "81": abstract, "217": url,
                         "236": arxiv_id, "203": date, "230": category}}
    """
    meta = data.get("meta", {})
    if not isinstance(meta, dict):
        return None

    title = _safe_str(meta.get("202", ""))
    abstract = _safe_str(meta.get("81", ""))
    url = _safe_str(meta.get("217", "")) or _safe_str(meta.get("url", ""))
    arxiv_id = _safe_str(meta.get("236", ""))
    date = _safe_str(meta.get("203", ""))
    category = _safe_str(meta.get("230", ""))
    authors_raw = meta.get("200", "")
    authors = _safe_str(authors_raw)

    if not title and not abstract:
        return None

    content = f"{title}\n\n{abstract}".strip()

    return Document(
        page_content=content,
        metadata={
            "source": "arxiv",
            "title": title,
            "arxiv_id": arxiv_id,
            "url": url,
            "date": date,
            "category": category,
            "authors": authors,
            "file": filename,
        },
    )


def load_arxiv(folder: Path, limit: int | None = None) -> list[Document]:
    """
    arXiv: полный текст из TXT + метаданные из парного JSON.
    Файлы json/<stem>.json и txt/<stem>.txt соответствуют одной статье.
    """
    docs: list[Document] = []

    json_dir = folder / "json"
    txt_dir = folder / "txt"

    if not txt_dir.exists():
        print(f"[arxiv]  нет папки {txt_dir}")
        return docs

    txt_files = sorted(txt_dir.glob("*.txt"))
    if limit:
        txt_files = txt_files[:limit]

    print(f"[arxiv] TXT-файлов: {len(txt_files)}")
    for f in tqdm(txt_files, desc="arxiv"):
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"  {f.name}: {e}")
            continue

        text = _WS_RE.sub(" ", text).strip()
        MAX_ARXIV_CHARS = 10_000
        text = text[:MAX_ARXIV_CHARS]
        if not text:
            continue

        meta = {"source": "arxiv", "file": f.name, "arxiv_id": f.stem}
        json_file = json_dir / f"{f.stem}.json"
        if json_file.exists():
            try:
                data = json.loads(json_file.read_text(encoding="utf-8", errors="ignore"))
                m = data.get("meta", {})
                meta.update({
                    "title": _safe_str(m.get("202", "")),
                    "arxiv_id": _safe_str(m.get("236", f.stem)),
                    "url": _safe_str(m.get("217", "")),
                    "date": _safe_str(m.get("203", "")),
                    "category": _safe_str(m.get("230", "")),
                    "authors": _safe_str(m.get("200", "")),
                })
            except Exception as e:
                print(f"  ⚠ meta {json_file.name}: {e}")

        title = meta.get("title", "")
        content = f"{title}\n\n{text}".strip() if title else text

        docs.append(Document(page_content=content, metadata=meta))

    print(f"[arxiv] загружено {len(docs)} документов")
    return docs

def load_all(data_dir: Path, limit_per_source: int | None = None) -> dict[str, list[Document]]:
    return {
        "habr":  load_habr(data_dir / "10_000-постов-Хабр.csv",  limit=limit_per_source),
        "lenta": load_lenta(data_dir / "10_000-постов-лента.csv", limit=limit_per_source),
        "arxiv": load_arxiv(data_dir / "arxiv2",    limit=limit_per_source),
    }
# ingest.py
# ================================
# Индексация docs/ в локальный ChromaDB с embedding_function OpenAI
# ================================

# --- SQLite shim (нужен на Streamlit Cloud) ---
# Должен идти ПЕРЕД "import chromadb"
try:
    import sqlite3
    from sqlite3 import sqlite_version
    def _ver(t): return tuple(int(x) for x in t.split("."))
    NEEDS_SHIM = _ver(sqlite_version) < (3, 35, 0)
except Exception:
    NEEDS_SHIM = True

if NEEDS_SHIM:
    import sys
    import pysqlite3  # noqa: F401
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
# ----------------------------------------------
import os
import glob
import os.path
from typing import List, Tuple

import chromadb
from chromadb.utils import embedding_functions

from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# ------------------------
# Параметры по умолчанию
# ------------------------
DEFAULT_DOC_DIR = "docs"
DEFAULT_CHROMA_PATH = os.getenv("KB_CHROMA_PATH", "data/chroma")
DEFAULT_COLLECTION = os.getenv("KB_COLLECTION_NAME", "kb_docs")
DEFAULT_MODEL = "text-embedding-3-small"


def _read_pdf_text(fp: str) -> str:
    """Прочитать текстовый PDF (без OCR) постранично."""
    txt = []
    try:
        r = PdfReader(fp)
        for p in r.pages:
            page_text = p.extract_text() or ""
            if page_text:
                txt.append(page_text)
    except Exception:
        # пропускаем проблемные файлы, не валим весь ingest
        return ""
    return "\n".join(txt)


def _load_docs(doc_dir: str) -> List[Tuple[str, str, str]]:
    """
    Собираем (typ, path, text) из docs/.
    typ ∈ {"md", "pdf"}.
    """
    docs = []

    # PDF
    for fp in glob.glob(f"{doc_dir}/**/*.pdf", recursive=True):
        text = _read_pdf_text(fp)
        if text.strip():
            docs.append(("pdf", fp, text))

    # MD
    for fp in glob.glob(f"{doc_dir}/**/*.md", recursive=True):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                text = f.read()
            if text.strip():
                docs.append(("md", fp, text))
        except Exception:
            continue

    return docs


def _chunk(title: str, text: str):
    """Нарезка на перекрывающиеся чанки."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900, chunk_overlap=120, separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)
    return [{"id": f"{title}::{i}", "text": c} for i, c in enumerate(chunks)]


def run_ingest(
    doc_dir: str = DEFAULT_DOC_DIR,
    chroma_path: str = DEFAULT_CHROMA_PATH,
    collection_name: str = DEFAULT_COLLECTION,
    model: str = DEFAULT_MODEL,
):
    """
    Основная функция индексации.
    Возвращает словарь со статистикой: files, chunks, added.
    """

    # подготовка окружения
    os.makedirs(chroma_path, exist_ok=True)
    os.makedirs(doc_dir, exist_ok=True)

    # ключ для OpenAI (нужен embedding_function)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY не задан")

    # подключение к Chroma + задание embedding_function
    chroma = chromadb.PersistentClient(path=chroma_path)
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=model,
    )
    # важно: коллекцию создаём/получаем С embedding_function
    collection = chroma.get_or_create_collection(
        collection_name,
        embedding_function=ef,
    )

    # загружаем исходники
    docs = _load_docs(doc_dir)
    payload = []
    for typ, path, txt in docs:
        title = os.path.basename(path)
        for ch in _chunk(title, txt):
            payload.append({
                "id": ch["id"],
                "text": ch["text"],
                "source": title,
                "path": path,
            })

    if not payload:
        return {"files": len(docs), "chunks": 0, "added": 0}

    # добавляем документы БЕЗ явных embeddings — Chroma посчитает сама
    collection.upsert(
        ids=[x["id"] for x in payload],
        documents=[x["text"] for x in payload],
        metadatas=[{"source": x["source"], "path": x["path"]} for x in payload],
    )

    return {"files": len(docs), "chunks": len(payload), "added": len(payload)}


# ------------------------
# standalone-запуск
# ------------------------
if __name__ == "__main__":
    try:
        stats = run_ingest()
        print("Ingested:", stats)
    except Exception as e:
        # печатаем явную ошибку, чтобы она попала в stdout Streamlit
        print("INGEST ERROR:", repr(e))
        raise

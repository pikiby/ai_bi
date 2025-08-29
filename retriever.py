# retriever.py

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
# ---------------------------------------------
import chromadb

# БЛОК: импортов и утилит
import os
import re
from typing import List, Dict
from openai import OpenAI


# --- нормализация/валидация имени коллекции ---
_NAME_RE = re.compile(r"^[a-z0-9_]{3,63}$")
def _norm_name(name: str) -> str:
    n = (name or "").strip().lower()
    if not _NAME_RE.fullmatch(n):
        raise ValueError(
            f"Invalid collection name {name!r}. Use 3–63 chars from [a-z0-9_]."
        )
    return n


# БЛОК: ретривер top-k чанков из Chroma по эмбеддингу вопроса
def retrieve(
    query: str,
    k: int = 5,
    chroma_path: str = "data/chroma",
    collection_name: str = "kb_docs",
    model: str = "text-embedding-3-small",
) -> List[Dict]:
    # нормализуем и валидируем имя коллекции
    collection_name = _norm_name(collection_name)

    # быстрые проверки
    q = (query or "").strip()
    if not q:
        return []

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    # инициализация клиентов
    client = OpenAI(api_key=api_key)
    chroma = chromadb.PersistentClient(path=chroma_path)
    col = chroma.get_or_create_collection(collection_name)

    # эмбеддинг вопроса
    q_emb = client.embeddings.create(model=model, input=[q]).data[0].embedding

    # ANN-поиск
    res = col.query(
        query_embeddings=[q_emb],
        n_results=max(1, int(k)),
        include=["documents", "metadatas", "ids", "distances"],
    )

    hits: List[Dict] = []
    # защита от пустого ответа
    if not res or not res.get("ids") or not res["ids"] or not res["ids"][0]:
        return hits

    n = len(res["ids"][0])
    for i in range(n):
        hits.append({
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "source": (res["metadatas"][0][i] or {}).get("source"),
            "path": (res["metadatas"][0][i] or {}).get("path"),
            "score": res["distances"][0][i],
        })
    return hits

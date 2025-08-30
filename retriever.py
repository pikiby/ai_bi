# retriever.py
# ===========================================
# Поиск top-k чанков из Chroma по текстовому запросу.
# Использует embedding_function OpenAI (как в ingest.py)
# и query_texts=[...] вместо ручных эмбеддингов.
# ===========================================

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

import os
import re
from typing import List, Dict

import chromadb
from chromadb.utils import embedding_functions


# --- нормализация/валидация имени коллекции ---
_NAME_RE = re.compile(r"^[a-z0-9_]{3,63}$")
def _norm_name(name: str) -> str:
    n = (name or "").strip().lower()
    if not _NAME_RE.fullmatch(n):
        raise ValueError("Invalid collection name. Use 3–63 chars from [a-z0-9_].")
    return n


# --- резервный поиск по ключевым словам (на случай пустого ANN) ---
def _keyword_fallback(col, query: str, k: int = 5) -> List[Dict]:
    q = (query or "").strip()
    if not q:
        return []

    # 1) Попытка where_document:$contains (если поддерживается вашей версией Chroma)
    try:
        res = col.get(where_document={"$contains": q.lower()}, limit=k)
        docs = res.get("documents") or []
        ids = res.get("ids") or []
        metas = res.get("metadatas") or []
        hits = []
        for i in range(min(len(docs), k)):
            meta_i = metas[i] or {}
            hits.append({
                "id": ids[i] if i < len(ids) else None,
                "text": docs[i],
                "source": meta_i.get("source"),
                "path": meta_i.get("path"),
                "score": 0.0,
            })
        if hits:
            return hits
    except Exception:
        pass

    # 2) Фолбэк через peek + простая фильтрация подстроки
    try:
        peek = col.peek(limit=200)
        docs = peek.get("documents") or []
        ids = peek.get("ids") or []
        metas = peek.get("metadatas") or []
        ql = q.lower()

        candidates = []
        for i, d in enumerate(docs):
            if not d:
                continue
            pos = d.lower().find(ql)
            if pos >= 0:
                candidates.append((pos, i))
        candidates.sort(key=lambda t: t[0])

        hits = []
        for _, i in candidates[:k]:
            meta_i = metas[i] or {}
            hits.append({
                "id": ids[i] if i < len(ids) else None,
                "text": docs[i],
                "source": meta_i.get("source"),
                "path": meta_i.get("path"),
                "score": 0.0,
            })
        return hits
    except Exception:
        return []


# --- основной ретривер ---
def retrieve(
    query: str,
    k: int = 5,
    chroma_path: str = "data/chroma",
    collection_name: str = "kb_docs",
    model: str = "text-embedding-3-small",
) -> List[Dict]:
    """
    Возвращает список документов-чанков с метаданными:
    [{id, text, source, path, score}, ...]
    """
    collection_name = _norm_name(collection_name)

    q = (query or "").strip()
    if not q:
        return []

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    # 1) Подключаемся к Chroma и задаём embedding_function OpenAI
    chroma = chromadb.PersistentClient(path=chroma_path)
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name=model,  # тот же, что в ingest.py
    )
    col = chroma.get_or_create_collection(
        collection_name,
        embedding_function=ef,
    )

    # 2) Запрос: Chroma сама эмбеддит текст и ищет ближайшие документы
    res = col.query(
        query_texts=[q],
        n_results=max(1, int(k)),
        include=["documents", "metadatas", "distances"],  # без "ids"
    )

    hits: List[Dict] = []

    # 3) Если ANN ничего не дал — пробуем keyword-fallback
    if (not res or
        not res.get("documents") or not res["documents"] or not res["documents"][0]):
        return _keyword_fallback(col, q, k=k)

    docs0 = res["documents"][0]
    metas0 = (res.get("metadatas") or [[]])[0]
    dists0 = (res.get("distances") or [[]])[0]

    n = len(docs0)
    for i in range(n):
        meta_i = metas0[i] if i < len(metas0) else {}
        hits.append({
            "id": None,  # ids не запрашивали; при желании можно добавить
            "text": docs0[i],
            "source": (meta_i or {}).get("source"),
            "path": (meta_i or {}).get("path"),
            "score": dists0[i] if i < len(dists0) else None,
        })

    # На всякий случай — если пусто, ещё раз keyword-fallback
    if not hits:
        hits = _keyword_fallback(col, q, k=k)

    return hits

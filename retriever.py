
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

# retriever.py — ДОБАВЬ эти вспомогательные функции рядом с другими

def _keyword_fallback(col, query: str, k: int = 5):
    """
    Простой резервный поиск: фильтруем документы, где встречается подстрока query (case-insensitive),
    потом ранжируем по длине совпадения/положению. Работает быстро на малых коллекциях.
    """
    q = (query or "").strip()
    if not q:
        return []

    # Попробуем сначала через where_document (если поддерживается версией Chroma)
    try:
        res = col.get(where_document={"$contains": q.lower()}, limit=k)
        docs = res.get("documents") or []
        ids = res.get("ids") or []
        metas = res.get("metadatas") or []
        hits = []
        for i in range(len(docs)):
            hits.append({
                "id": ids[i] if i < len(ids) else None,
                "text": docs[i],
                "source": (metas[i] or {}).get("source"),
                "path": (metas[i] or {}).get("path"),
                "score": 0.0,  # без метрики
            })
        return hits[:k]
    except Exception:
        pass

    # Если where_document нет — берём маленький сэмпл и фильтруем вручную
    try:
        peek = col.peek(limit=200)  # достаточно для POC
        docs = peek.get("documents") or []
        ids = peek.get("ids") or []
        metas = peek.get("metadatas") or []
        ql = q.lower()
        cand = []
        for i, d in enumerate(docs):
            if not d:
                continue
            pos = (d.lower()).find(ql)
            if pos >= 0:
                cand.append((pos, i))
        cand.sort(key=lambda x: x[0])
        hits = []
        for _, i in cand[:k]:
            hits.append({
                "id": ids[i] if i < len(ids) else None,
                "text": docs[i],
                "source": (metas[i] or {}).get("source"),
                "path": (metas[i] or {}).get("path"),
                "score": 0.0,
            })
        return hits
    except Exception:
        return []


# БЛОК: ретривер top-k чанков из Chroma по эмбеддингу вопроса
def retrieve(
    query: str,
    k: int = 5,
    chroma_path: str = "data/chroma",
    collection_name: str = "kb_docs",
    model: str = "text-embedding-3-small",
) -> List[Dict]:
    print(f"[retriever] chroma_path={chroma_path} collection={collection_name}")

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
        include=["documents", "metadatas", "distances"],  # без "ids"
    )

    hits: List[Dict] = []
    # защита от пустого ответа
    if (not res or
        not res.get("documents") or not res["documents"] or not res["documents"][0]):
        return hits

    docs0 = res["documents"][0]
    metas0 = (res.get("metadatas") or [[]])[0]
    ids0   = (res.get("ids") or [[]])[0]           # ids может отсутствовать, но часто приходит
    dists0 = (res.get("distances") or [[]])[0]     # distances может отсутствовать, подстрахуемся

    n = len(docs0)
    for i in range(n):
        meta_i = metas0[i] if i < len(metas0) else {}
        hit = {
            "id":    ids0[i] if i < len(ids0) else None,
            "text":  docs0[i],
            "source": (meta_i or {}).get("source"),
            "path":   (meta_i or {}).get("path"),
            "score":  dists0[i] if i < len(dists0) else None,
        }
        hits.append(hit)
    
    if not hits:
        hits = _keyword_fallback(col, query, k=k)

    return hits

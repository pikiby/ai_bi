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

# retriever.py
# БЛОК: импортов и клиентов
import os
from openai import OpenAI

# БЛОК: ретривер top-k чанков из Chroma по эмбеддингу вопроса
def retrieve(query: str, k: int = 5, chroma_path="data/chroma", collection_name="kb_docs", model="text-embedding-3-small"):
    import re
    NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{2,62}$")
    if not NAME_RE.match(collection_name):
        raise ValueError(f"Invalid collection name '{collection_name}'. Use 3–63 chars: [a-z0-9_-].")
    
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chroma = chromadb.PersistentClient(path=chroma_path)
    col = chroma.get_or_create_collection(collection_name)

    q_emb = client.embeddings.create(model=model, input=[query]).data[0].embedding

    res = col.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas", "ids", "distances"],
    )

    hits = []
    if res["ids"]:
        n = len(res["ids"][0])
        for i in range(n):
            hits.append({
                "id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "source": res["metadatas"][0][i].get("source"),
                "path": res["metadatas"][0][i].get("path"),
                "score": res["distances"][0][i],
            })
    return hits

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

# ingest.py
# БЛОК: импортов и клиентов
import os, glob, os.path
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# БЛОК: запуск индексации (как функция — удобно вызывать из app по кнопке)
def run_ingest(doc_dir="docs", chroma_path="data/chroma", collection_name="kb", model="text-embedding-3-small"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    chroma = chromadb.PersistentClient(path=chroma_path)
    collection = chroma.get_or_create_collection(collection_name)

    # эмбеддинги батчом
    def embed(texts):
        resp = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]

    # загрузка исходников из docs/
    def load_docs():
        docs = []
        for fp in glob.glob(f"{doc_dir}/**/*.pdf", recursive=True):
            txt = ""
            try:
                r = PdfReader(fp)
                for p in r.pages:
                    txt += (p.extract_text() or "") + "\n"
            except:
                continue
            docs.append(("pdf", fp, txt))

        for fp in glob.glob(f"{doc_dir}/**/*.md", recursive=True):
            docs.append(("md", fp, open(fp, "r", encoding="utf-8").read()))
        return docs

    # разбиение на чанки
    def chunk(title, text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900, chunk_overlap=120, separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_text(text)
        return [{"id": f"{title}::{i}", "text": c} for i, c in enumerate(chunks)]

    docs = load_docs()
    payload = []
    for typ, path, txt in docs:
        title = os.path.basename(path)
        for ch in chunk(title, txt):
            payload.append({"id": ch["id"], "text": ch["text"], "source": title, "path": path})

    if not payload:
        return {"files": len(docs), "chunks": 0, "added": 0}

    embeddings = embed([x["text"] for x in payload])
    collection.add(
        ids=[x["id"] for x in payload],
        documents=[x["text"] for x in payload],
        embeddings=embeddings,
        metadatas=[{"source": x["source"], "path": x["path"]} for x in payload],
    )
    return {"files": len(docs), "chunks": len(payload), "added": len(payload)}

# БЛОК: standalone-режим (на случай ручного запуска: python ingest.py)
if __name__ == "__main__":
    stats = run_ingest()
    print("Ingested:", stats)

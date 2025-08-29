# ingest.py

# --- SQLite shim (нужен на Streamlit Cloud) ---
# ДОЛЖЕН идти ПЕРЕД "import chromadb"
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

# ---- импортов и утилит ----
import os
import glob
from typing import List, Tuple
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


# ---- нормализация/валидация имени коллекции ----
import re
_NAME_RE = re.compile(r"[a-z0-9_]{3,63}$")
def _norm_name(name: str) -> str:
    n = (name or "").strip().lower()
    if not _NAME_RE.fullmatch(n):
        raise ValueError(
            f"Invalid collection name {name!r}. Use 3–63 chars from [a-z0-9_]."
        )
    return n


def run_ingest(
    doc_dir: str = "docs",
    chroma_path: str = "data/chroma",
    collection_name: str = "kb_docs",
    model: str = "text-embedding-3-small",
) -> dict:
    """Индексация: читает файлы из docs/, режет на чанки, считает эмбеддинги, пишет в Chroma."""
    collection_name = _norm_name(collection_name)

    # пути и директории
    os.makedirs(doc_dir, exist_ok=True)
    os.makedirs(chroma_path, exist_ok=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    chroma = chromadb.PersistentClient(path=chroma_path)
    collection = chroma.get_or_create_collection(collection_name)

    # --- helpers ---
    def _embed(texts: List[str]):
        # простой батч (при больших объёмах можно дробить по N)
        resp = client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]

    def _load_docs() -> List[Tuple[str, str, str]]:
        """Возвращает список (тип, путь, текст)."""
        docs: List[Tuple[str, str, str]] = []

        # PDF
        for fp in glob.glob(os.path.join(doc_dir, "**", "*.pdf"), recursive=True):
            txt = ""
            try:
                r = PdfReader(fp)
                for p in r.pages:
                    txt += (p.extract_text() or "") + "\n"
            except Exception:
                # пропускаем битые/сканы без OCR
                continue
            if txt.strip():
                docs.append(("pdf", fp, txt))

        # Markdown
        for fp in glob.glob(os.path.join(doc_dir, "**", "*.md"), recursive=True):
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
            except Exception:
                continue
            if txt.strip():
                docs.append(("md", fp, txt))

        return docs

    def _chunk(text: str) -> List[str]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=900, chunk_overlap=120, separators=["\n\n", "\n", " ", ""]
        )
        return [c for c in splitter.split_text(text) if c.strip()]

    # --- сбор данных ---
    docs = _load_docs()
    payload = []
    for _, path, txt in docs:
        title = os.path.basename(path)
        for i, ch in enumerate(_chunk(txt)):
            payload.append({
                "id": f"{title}::{i}",
                "text": ch,
                "source": title,
                "path": path,
            })

    if not payload:
        return {"files": len(docs), "chunks": 0, "added": 0}

    # --- эмбеддинги + запись ---
    texts = [x["text"] for x in payload]
    embeddings = _embed(texts)

    collection.add(
        ids=[x["id"] for x in payload],
        documents=texts,
        embeddings=embeddings,
        metadatas=[{"source": x["source"], "path": x["path"]} for x in payload],
    )
    return {"files": len(docs), "chunks": len(payload), "added": len(payload)}


if __name__ == "__main__":
    # читаем параметры из окружения, которые передаёт app.py при subprocess.run(...)
    CHROMA_PATH = os.getenv("KB_CHROMA_PATH", "data/chroma")
    COLLECTION_NAME = os.getenv("KB_COLLECTION_NAME", "kb_docs")
    stats = run_ingest(
        doc_dir="docs",
        chroma_path=CHROMA_PATH,
        collection_name=COLLECTION_NAME,
    )
    print("Ingested:", stats)

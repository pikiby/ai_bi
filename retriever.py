# retriever.py
# БЛОК: импортов и клиентов
import os
import chromadb
from openai import OpenAI

# БЛОК: ретривер top-k чанков из Chroma по эмбеддингу вопроса
def retrieve(query: str, k: int = 5, chroma_path="data/chroma", collection_name="kb", model="text-embedding-3-small"):
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

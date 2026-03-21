# app/main.py

import sys
from pathlib import Path

from fastapi import FastAPI

if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from app.services.embedding import embed_texts
from app.services.retriever import VectorStore
from app.services.generator import generate_answer

app = FastAPI()

vector_store = VectorStore(dim=384)

@app.post("/query")
def query_rag(query: str):
    query_embedding = embed_texts([query])[0]
    contexts = vector_store.search(query_embedding)
    if not contexts:
        return {"answer": "No indexed documents found yet. Add embeddings to the vector store first."}

    context_text = "\n".join(contexts)
    answer = generate_answer(query, context_text)

    return {"answer": answer}
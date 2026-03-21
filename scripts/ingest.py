# scripts/ingest.py

from app.utils.chunking import chunk_text
from app.services.embedding import embed_texts
from app.services.retriever import VectorStore

def ingest(document_text):
    chunks = chunk_text(document_text)
    embeddings = embed_texts(chunks)

    store = VectorStore(dim=384)
    store.add(embeddings, chunks)

    return store
import os
import requests
import json
import numpy as np
from pathlib import Path
import sys
from typing import Any

# Add repo root to path for imports
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ingestion.chunking.chunk_config import ChunkConfig, build_chunker
from ingestion.loader import load_documents

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


def generate_embedding(text: str) -> list[float]:
    resp = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    embeddings = []
    for text in texts:
        embedding = generate_embedding(text)
        embeddings.append(embedding)
    return embeddings


def embed_texts(texts: list[str]) -> np.ndarray:
    """Create embeddings using Ollama batch -> float32 matrix."""
    vectors = generate_embeddings(texts)
    return np.array(vectors, dtype="float32")


def embed_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not chunks:
        return []
    
    texts = [chunk["text"] for chunk in chunks]
    embeddings = generate_embeddings(texts)
    
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
    
    return chunks


def chunk_documents(
    data_path: str,
    strategy: str = "fixed",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_chunk_chars: int = 20,
) -> list[dict[str, Any]]:
    from ingestion.chunking.chunk_config import ChunkConfig, build_chunker
    from ingestion.loader import load_documents
    
    docs = load_documents(data_path)
    chunker = build_chunker(
        ChunkConfig(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_chars=min_chunk_chars,
        )
    )

    chunks: list[dict[str, Any]] = []
    for doc in docs:
        for chunk in chunker.chunk_document(doc):
            chunks.append(
                {
                    "id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
            )

    return chunks
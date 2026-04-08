from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

if __package__ in (None, ""):
    # Supports running as: python ingestion/embed.py
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from ingestion.chunking.chunk_config import ChunkConfig, build_chunker
from ingestion.loader import load_documents


# Use all-MiniLM-L6-v2 for fast embeddings or configure via EMBED_MODEL env var
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
EXPECTED_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 dim



def configure_embedder() -> None:
    """Initialize sentence-transformers embedder."""
    load_dotenv()
    if SentenceTransformer is None:
        raise RuntimeError(
            "Missing sentence-transformers. Install it with: pip install sentence-transformers"
        )


_model_cache = None


def get_embedder() -> Any:
    """Get or initialize the cached embeddings model."""
    global _model_cache
    if _model_cache is None:
        if SentenceTransformer is None:
            raise RuntimeError("sentence-transformers not installed")
        _model_cache = SentenceTransformer(EMBED_MODEL)
    return _model_cache


def embed_text(text: str) -> list[float]:
    """Embed a single text string using sentence-transformers."""
    configure_embedder()
    model = get_embedder()
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()


def embed_batch(text_list: list[str]) -> list[list[float]]:
    """Embed a batch of texts using sentence-transformers."""
    configure_embedder()
    model = get_embedder()
    embeddings = model.encode(text_list, convert_to_numpy=True)
    return embeddings.tolist()


def _extract_legacy_embeddings(response: Any) -> list[list[float]]:
    """Legacy compatibility helper retained for older call sites."""
    raise RuntimeError("Legacy embedding responses are no longer supported. Use sentence-transformers instead.")


def embed_texts(texts: list[str]) -> np.ndarray:
    """Create embeddings using sentence-transformers (all-MiniLM-L6-v2) batch -> float32 matrix."""
    vectors = embed_batch(texts)
    return np.array(vectors, dtype="float32")


def create_embeddings_for_documents(documents: list[dict[str, Any]]) -> np.ndarray:
    """Create embeddings for chunk-like documents with a `text` field."""
    texts = [doc["text"] for doc in documents]
    embeddings = embed_texts(texts)

    if embeddings.ndim != 2:
        raise RuntimeError(f"Unexpected embedding shape: {embeddings.shape}")

    if embeddings.shape[1] != EXPECTED_EMBEDDING_DIM:
        raise RuntimeError(
            f"Unexpected embedding dimension: {embeddings.shape[1]} (expected {EXPECTED_EMBEDDING_DIM})"
        )

    return embeddings


def chunk_documents(
    data_path: str,
    strategy: str = "fixed",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_chunk_chars: int = 20,
) -> list[dict[str, Any]]:
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


def embed_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not chunks:
        return []

    embedding_matrix = create_embeddings_for_documents(chunks)
    embeddings = embedding_matrix.tolist()

    if len(embeddings) != len(chunks):
        raise RuntimeError("Embedding result size mismatch.")

    for idx, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[idx]

    return chunks


def save_embeddings(chunks: list[dict[str, Any]], output_path: str = "data/embeddings.json") -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk documents and create embeddings using sentence-transformers.")
    parser.add_argument("--data-path", default="data/sample")
    parser.add_argument("--output", default="data/embeddings.json")
    parser.add_argument("--strategy", default="fixed", choices=["fixed", "recursive"])
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--min-chunk-chars", type=int, default=20)
    args = parser.parse_args()

    configure_embedder()
    chunks = chunk_documents(
        data_path=args.data_path,
        strategy=args.strategy,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        min_chunk_chars=args.min_chunk_chars,
    )
    embedded_chunks = embed_chunks(chunks)
    save_embeddings(embedded_chunks, output_path=args.output)

    # Explicit shape output for STEP 2 verification.
    shape = (len(embedded_chunks), len(embedded_chunks[0]["embedding"]) if embedded_chunks else 0)
    print(shape)

    print(f"Embedded chunks: {len(embedded_chunks)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
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
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    from google import genai as google_genai
except ImportError:
    google_genai = None

if __package__ in (None, ""):
    # Supports running as: python ingestion/embed.py
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from ingestion.chunking.chunk_config import ChunkConfig, build_chunker
from ingestion.loader import load_documents


EMBED_MODELS = [
    os.getenv("EMBED_MODEL", "models/text-embedding-004"),
    "models/embedding-001",
    "embedding-001",
    "gemini-embedding-001",
]

EXPECTED_EMBEDDING_DIM = 768


def _extract_legacy_embeddings(response: Any) -> list[list[float]]:
    """Normalize legacy google.generativeai embed responses to list[list[float]]."""
    if hasattr(response, "to_dict"):
        try:
            response = response.to_dict()
        except Exception:  # noqa: BLE001
            pass

    if hasattr(response, "embeddings"):
        values = []
        for item in getattr(response, "embeddings"):
            if hasattr(item, "values"):
                values.append(list(item.values))
            elif hasattr(item, "embedding"):
                values.append(list(item.embedding))
            elif isinstance(item, dict):
                payload = item.get("values") or item.get("embedding")
                if payload is not None:
                    values.append(list(payload))
        if values:
            return values

    if hasattr(response, "embedding"):
        return [list(getattr(response, "embedding"))]

    if isinstance(response, dict):
        payload = response.get("embedding") or response.get("embeddings")
        if isinstance(payload, list) and payload and isinstance(payload[0], dict):
            if "embedding" in payload[0]:
                return [item["embedding"] for item in payload]
            if "values" in payload[0]:
                return [item["values"] for item in payload]
        if isinstance(payload, list) and payload and isinstance(payload[0], (int, float)):
            return [payload]

    if isinstance(response, list) and response:
        if isinstance(response[0], (int, float)):
            return [response]
        if hasattr(response[0], "values"):
            return [list(item.values) for item in response]
        if isinstance(response[0], dict):
            values = []
            for item in response:
                payload = item.get("values") or item.get("embedding")
                if payload is not None:
                    values.append(list(payload))
            if values:
                return values

    raise RuntimeError("Unexpected legacy embedding response format.")


def configure_genai() -> None:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) in environment.")
    if genai is None and google_genai is None:
        raise RuntimeError(
            "Missing Gemini SDK. Install `google-generativeai` (legacy) or `google-genai` in your active interpreter."
        )
    if genai is not None:
        genai.configure(api_key=api_key)


def embed_text(text: str) -> list[float]:
    configure_genai()

    if google_genai is not None:
        client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        last_error = None
        for model_name in EMBED_MODELS:
            try:
                response = client.models.embed_content(model=model_name, contents=[text])
                return response.embeddings[0].values
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if "not found" not in str(exc).lower():
                    raise
        raise RuntimeError(f"No usable embedding model found. Tried: {EMBED_MODELS}") from last_error

    if genai is not None:
        last_error: Exception | None = None
        for model_name in EMBED_MODELS:
            try:
                response = genai.embed_content(model=model_name, content=text)
                return response["embedding"]
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if "not found" not in str(exc).lower():
                    raise
        raise RuntimeError(f"No usable embedding model found. Tried: {EMBED_MODELS}") from last_error

    raise RuntimeError("No Gemini SDK client available.")


def embed_batch(text_list: list[str]) -> list[list[float]]:
    configure_genai()

    if google_genai is not None:
        client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        last_error = None
        for model_name in EMBED_MODELS:
            try:
                response = client.models.embed_content(model=model_name, contents=text_list)
                return [item.values for item in response.embeddings]
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if "not found" not in str(exc).lower():
                    raise
        raise RuntimeError(f"No usable embedding model found. Tried: {EMBED_MODELS}") from last_error

    if genai is not None:
        last_error: Exception | None = None
        for model_name in EMBED_MODELS:
            try:
                response = genai.embed_content(model=model_name, content=text_list)
                return _extract_legacy_embeddings(response)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if "not found" not in str(exc).lower():
                    raise
        raise RuntimeError(f"No usable embedding model found. Tried: {EMBED_MODELS}") from last_error

    raise RuntimeError("No Gemini SDK client available.")


def embed_texts(texts: list[str]) -> np.ndarray:
    """STEP 2 helper: Gemini text-embedding-004 batch -> float32 matrix."""
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
    parser = argparse.ArgumentParser(description="Chunk documents and create Gemini embeddings.")
    parser.add_argument("--data-path", default="data/sample")
    parser.add_argument("--output", default="data/embeddings.json")
    parser.add_argument("--strategy", default="fixed", choices=["fixed", "recursive"])
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--min-chunk-chars", type=int, default=20)
    args = parser.parse_args()

    configure_genai()
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
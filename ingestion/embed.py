from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

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


def embed_batch(text_list: list[str]) -> list[list[float]]:
    if genai is not None:
        last_error: Exception | None = None
        for model_name in EMBED_MODELS:
            try:
                response = genai.embed_content(model=model_name, content=text_list)
                return response["embedding"]
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if "not found" not in str(exc).lower():
                    raise
        raise RuntimeError(f"No usable embedding model found. Tried: {EMBED_MODELS}") from last_error

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

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_batch(texts)

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

    print(f"Embedded chunks: {len(embedded_chunks)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
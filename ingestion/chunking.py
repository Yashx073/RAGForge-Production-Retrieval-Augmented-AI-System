from __future__ import annotations

import nltk

from .loader import load_documents
from .chunking.chunk_config import ChunkConfig, ExperimentChunkConfig, build_chunker
from .chunking.experiment_chunking import default_experiment_config, run_chunking_experiment
from .chunking.fixed_chunker import FixedChunker
from .chunking.recursive_chunker import RecursiveChunker


__all__ = [
    "ChunkConfig",
    "ExperimentChunkConfig",
    "FixedChunker",
    "RecursiveChunker",
    "build_chunker",
    "default_experiment_config",
    "run_chunking_experiment",
    "fixed_chunk",
    "recursive_chunk",
    "sentence_chunk",
    "chunk_documents",
    "build_bm25_from_chunking",
]


def _get_text(obj) -> str:
    if isinstance(obj, str):
        return obj
    return getattr(obj, "text", str(obj))


def fixed_chunk(text_or_doc, size: int = 512, overlap: int = 50):
    chunker = FixedChunker(chunk_size=size, chunk_overlap=overlap)
    return [chunk.text for chunk in chunker.chunk_document(text_or_doc)]


def recursive_chunk(text_or_doc, chunk_size: int = 512, chunk_overlap: int = 50):
    chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [chunk.text for chunk in chunker.chunk_document(text_or_doc)]


def sentence_chunk(text_or_doc, max_sentences: int = 5):
    text = _get_text(text_or_doc)
    if not text:
        return []

    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        sentences = nltk.sent_tokenize(text)

    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunks.append(" ".join(sentences[i : i + max_sentences]))

    return chunks


def chunk_documents(
    data_path: str,
    strategy: str = "fixed",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_chunk_chars: int = 20,
):
    docs = load_documents(data_path)
    chunker = build_chunker(
        ChunkConfig(
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_chars=min_chunk_chars,
        )
    )

    chunks = []
    for doc in docs:
        for chunk in chunker.chunk_document(doc):
            chunks.append(chunk.text)

    return chunks


def build_bm25_from_chunking(
    data_path: str,
    strategy: str = "fixed",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    min_chunk_chars: int = 20,
):
    from retrieval.bm25 import BM25Retriever

    chunks = chunk_documents(
        data_path=data_path,
        strategy=strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        min_chunk_chars=min_chunk_chars,
    )

    return BM25Retriever.from_chunks(chunks)

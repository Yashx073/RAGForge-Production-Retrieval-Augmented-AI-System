from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from hashlib import sha1
from typing import Any


@dataclass(frozen=True)
class TextSpan:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    metadata: dict[str, Any]


class BaseChunker(ABC):
    """Base class for deterministic chunking strategies."""

    def __init__(
        self,
        strategy_name: str,
        chunk_size: int,
        chunk_overlap: int,
        min_chunk_chars: int = 1,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be >= 0")
        if chunk_overlap >= chunk_size:
            chunk_overlap = chunk_size - 1

        self.strategy_name = strategy_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_chars = min_chunk_chars

    @abstractmethod
    def split_text(self, text: str) -> list[TextSpan]:
        """Return text spans for chunk generation."""

    def chunk_document(self, document: Any) -> list[Chunk]:
        doc_id = getattr(document, "doc_id", None) or "ad_hoc_doc"
        doc_text = getattr(document, "text", document)
        doc_metadata = dict(getattr(document, "metadata", {}) or {})

        if not isinstance(doc_text, str):
            doc_text = str(doc_text)

        if not doc_text.strip():
            return []

        spans = self.split_text(doc_text)
        chunks: list[Chunk] = []

        for index, span in enumerate(spans):
            chunk_text = span.text.strip()
            if len(chunk_text) < self.min_chunk_chars:
                continue

            chunk_id = self._make_chunk_id(doc_id, index, span.start, span.end, chunk_text)
            metadata = {
                "doc_id": doc_id,
                "chunk_id": chunk_id,
                "chunk_index": index,
                "start_char": span.start,
                "end_char": span.end,
                "strategy": self.strategy_name,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                **doc_metadata,
            }

            chunks.append(Chunk(chunk_id=chunk_id, doc_id=doc_id, text=chunk_text, metadata=metadata))

        return chunks

    def _make_chunk_id(
        self,
        doc_id: str,
        chunk_index: int,
        start_char: int,
        end_char: int,
        chunk_text: str,
    ) -> str:
        payload = f"{doc_id}|{self.strategy_name}|{chunk_index}|{start_char}|{end_char}|{chunk_text}"
        digest = sha1(payload.encode("utf-8")).hexdigest()[:12]
        return f"{doc_id}:{self.strategy_name}:{chunk_index}:{digest}"

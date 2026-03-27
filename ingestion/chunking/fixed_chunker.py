from __future__ import annotations

try:
    from .base_chunker import BaseChunker, TextSpan
except ImportError:
    from base_chunker import BaseChunker, TextSpan


class FixedChunker(BaseChunker):
    """Character window chunker with fixed size and overlap."""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, min_chunk_chars: int = 1) -> None:
        super().__init__(
            strategy_name="fixed",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_chars=min_chunk_chars,
        )

    def split_text(self, text: str) -> list[TextSpan]:
        if not text:
            return []

        spans: list[TextSpan] = []
        step = self.chunk_size - self.chunk_overlap
        start = 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            spans.append(TextSpan(text=text[start:end], start=start, end=end))
            start += step

        return spans

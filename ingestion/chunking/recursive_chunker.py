from __future__ import annotations

try:
    from .base_chunker import BaseChunker, TextSpan
except ImportError:
    from base_chunker import BaseChunker, TextSpan


class RecursiveChunker(BaseChunker):
    """Hierarchy-aware chunker using recursive separators."""

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_chars: int = 1,
        separators: list[str] | None = None,
    ) -> None:
        super().__init__(
            strategy_name="recursive",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_chars=min_chunk_chars,
        )
        self.separators = separators or ["\n\n", "\n", ". ", " "]

    def split_text(self, text: str) -> list[TextSpan]:
        if not text:
            return []

        parts = self._recursive_split(text, self.separators)
        base_spans = self._parts_to_spans(text, parts)

        if self.chunk_overlap <= 0:
            return base_spans

        overlapped_spans: list[TextSpan] = []
        for span in base_spans:
            start = max(0, span.start - self.chunk_overlap)
            end = span.end
            overlapped_spans.append(TextSpan(text=text[start:end], start=start, end=end))

        return overlapped_spans

    def _recursive_split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]

        if not separators:
            return self._fixed_split(text)

        separator = separators[0]
        if separator == "" or separator not in text:
            return self._recursive_split(text, separators[1:])

        pieces = text.split(separator)
        tokens = [
            piece if idx == len(pieces) - 1 else piece + separator
            for idx, piece in enumerate(pieces)
        ]

        merged: list[str] = []
        current = ""

        for token in tokens:
            if len(current) + len(token) <= self.chunk_size:
                current += token
                continue

            if current:
                merged.append(current)

            if len(token) <= self.chunk_size:
                current = token
            else:
                merged.extend(self._recursive_split(token, separators[1:]))
                current = ""

        if current:
            merged.append(current)

        normalized: list[str] = []
        for item in merged:
            if len(item) <= self.chunk_size:
                normalized.append(item)
            else:
                normalized.extend(self._recursive_split(item, separators[1:]))

        return normalized

    def _fixed_split(self, text: str) -> list[str]:
        chunks = []
        for start in range(0, len(text), self.chunk_size):
            chunks.append(text[start : start + self.chunk_size])
        return chunks

    def _parts_to_spans(self, original_text: str, parts: list[str]) -> list[TextSpan]:
        spans: list[TextSpan] = []
        cursor = 0

        for part in parts:
            if not part:
                continue

            start = original_text.find(part, cursor)
            if start < 0:
                start = cursor

            end = min(start + len(part), len(original_text))
            spans.append(TextSpan(text=original_text[start:end], start=start, end=end))
            cursor = end

        return spans

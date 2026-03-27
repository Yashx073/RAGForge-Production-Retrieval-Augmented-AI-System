from __future__ import annotations

from dataclasses import dataclass, field

try:
    from .base_chunker import BaseChunker
    from .fixed_chunker import FixedChunker
    from .recursive_chunker import RecursiveChunker
except ImportError:
    from base_chunker import BaseChunker
    from fixed_chunker import FixedChunker
    from recursive_chunker import RecursiveChunker


@dataclass(frozen=True)
class ChunkConfig:
    strategy: str
    chunk_size: int = 512
    chunk_overlap: int = 50
    min_chunk_chars: int = 20
    separators: list[str] | None = None


@dataclass(frozen=True)
class ExperimentChunkConfig:
    strategies: list[ChunkConfig] = field(
        default_factory=lambda: [
            ChunkConfig(strategy="fixed", chunk_size=512, chunk_overlap=50, min_chunk_chars=20),
            ChunkConfig(strategy="recursive", chunk_size=512, chunk_overlap=50, min_chunk_chars=20),
        ]
    )


def build_chunker(config: ChunkConfig) -> BaseChunker:
    strategy = config.strategy.lower().strip()

    if strategy == "fixed":
        return FixedChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            min_chunk_chars=config.min_chunk_chars,
        )

    if strategy == "recursive":
        return RecursiveChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            min_chunk_chars=config.min_chunk_chars,
            separators=config.separators,
        )

    raise ValueError(f"Unsupported chunking strategy: {config.strategy}")

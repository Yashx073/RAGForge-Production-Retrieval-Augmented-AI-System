from typing import Any
from ingestion.chunking.chunk_config import ChunkConfig, build_chunker


def chunk_pages(pages: list[dict[str, Any]], config: ChunkConfig = None) -> list[dict[str, Any]]:
    if config is None:
        config = ChunkConfig(strategy="fixed", chunk_size=512, chunk_overlap=50, min_chunk_chars=20)
    
    chunker = build_chunker(config)
    chunks = []
    
    for page in pages:
        doc = {"text": page["text"], "metadata": {"page": page["page"]}}
        for chunk in chunker.chunk_document(doc):
            chunks.append({
                "text": chunk.text,
                "metadata": {
                    "page": page["page"],
                    "chunk_id": chunk.chunk_id,
                    **chunk.metadata
                }
            })
    
    return chunks
from __future__ import annotations
from typing import Any
import faiss
import numpy as np
from ingestion.embed import embed_texts


class DenseRetriever:
    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.metadata: dict[int, dict[str, Any]] = {}

    def build_index(self, documents: list[dict[str, Any]]) -> None:
        if not documents:
            raise ValueError("documents must not be empty")

        texts = [doc["text"] for doc in documents]
        embeddings = np.asarray(embed_texts(texts), dtype="float32")
        faiss.normalize_L2(embeddings)

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)

        # Required step: add embeddings to FAISS index.
        self.index.add(embeddings)
        print("vectors in index:", self.index.ntotal)

        self.metadata = {i: documents[i] for i in range(len(documents))}

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        if self.index is None:
            raise RuntimeError("index is not built. Call build_index(documents) first.")

        query_embedding = np.asarray(embed_texts([query]), dtype="float32")
        faiss.normalize_L2(query_embedding)

        limit = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, limit)

        results: list[dict[str, Any]] = []
        for rank, doc_index in enumerate(indices[0]):
            if doc_index < 0:
                continue
            # Handle both int and string keys
            key = int(doc_index)
            if key not in self.metadata:
                key = str(key)
            if key not in self.metadata:
                continue
            item = dict(self.metadata[key])
            item["id"] = key
            item["score"] = float(scores[0][rank])
            results.append(item)

        return results
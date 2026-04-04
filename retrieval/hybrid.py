from __future__ import annotations

from typing import Any

from ingestion.embed import chunk_documents
from retrieval.bm25 import BM25Retriever, build_documents_from_chunks
from retrieval.dense_faiss import DenseRetriever


def _normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_s = min(scores)
    max_s = max(scores)
    denom = max_s - min_s
    if denom == 0:
        return [1.0 for _ in scores]
    return [(s - min_s) / denom for s in scores]


class HybridRetriever:
    def __init__(self, dense_weight: float = 0.6, bm25_weight: float = 0.4) -> None:
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.dense = DenseRetriever()
        self.bm25: BM25Retriever | None = None
        self.documents: list[dict[str, Any]] = []

    def build(self, chunks: list[str]) -> None:
        self.documents = build_documents_from_chunks(chunks)
        self.bm25 = BM25Retriever(self.documents)
        self.dense.build_index(self.documents)

    def build_from_data_path(
        self,
        data_path: str,
        strategy: str = "fixed",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        min_chunk_chars: int = 20,
    ) -> None:
        chunks = chunk_documents(
            data_path=data_path,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_chars=min_chunk_chars,
        )
        chunk_texts = [chunk["text"] for chunk in chunks]
        self.build(chunk_texts)

    def search(self, query: str, k: int = 5, candidate_k: int = 20) -> list[dict[str, Any]]:
        if self.bm25 is None:
            raise RuntimeError("HybridRetriever is not initialized. Call build or build_from_data_path first.")

        dense_results = self.dense.search(query, k=candidate_k)
        bm25_results = self.bm25.search(query, k=candidate_k)

        dense_scores = _normalize([float(r["score"]) for r in dense_results])
        bm25_scores = _normalize([float(r["bm25_score"]) for r in bm25_results])

        combined: dict[int, dict[str, float]] = {}

        for idx, result in enumerate(dense_results):
            doc_id = int(result["id"])
            combined.setdefault(doc_id, {"dense": 0.0, "bm25": 0.0})
            combined[doc_id]["dense"] = dense_scores[idx]

        for idx, result in enumerate(bm25_results):
            doc_id = int(result["id"])
            combined.setdefault(doc_id, {"dense": 0.0, "bm25": 0.0})
            combined[doc_id]["bm25"] = bm25_scores[idx]

        id_to_doc = {int(doc["id"]): doc for doc in self.documents}
        merged: list[dict[str, Any]] = []

        for doc_id, score_parts in combined.items():
            hybrid_score = (
                self.dense_weight * score_parts["dense"]
                + self.bm25_weight * score_parts["bm25"]
            )
            doc = id_to_doc.get(doc_id, {"id": doc_id, "text": ""})
            merged.append(
                {
                    "id": doc_id,
                    "score": float(hybrid_score),
                    "text": doc.get("text", ""),
                }
            )

        merged.sort(key=lambda item: item["score"], reverse=True)
        return merged[:k]

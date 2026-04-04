from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sentence_transformers import CrossEncoder

if TYPE_CHECKING:
    from retrieval.hybrid import HybridRetriever

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def _resolve_device() -> str:
    if torch is not None and torch.cuda.is_available():
        return "cuda"
    return "cpu"


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device or _resolve_device()
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    def rerank_documents(
        self,
        query: str,
        docs: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not docs:
            return []

        pairs = [(query, str(doc.get("text", ""))) for doc in docs]
        scores = self._get_model().predict(pairs)

        reranked: list[dict[str, Any]] = []
        for doc, score in zip(docs, scores):
            item = dict(doc)
            item["score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda item: item["score"], reverse=True)
        return reranked[:top_k]


def rerank_documents(
    query: str,
    docs: list[dict[str, Any]],
    top_k: int = 5,
    reranker: CrossEncoderReranker | None = None,
) -> list[dict[str, Any]]:
    model = reranker or CrossEncoderReranker()
    return model.rerank_documents(query=query, docs=docs, top_k=top_k)


def retrieve_with_rerank(
    query: str,
    hybrid_retriever: "HybridRetriever",
    top_k: int = 5,
    candidate_k: int = 20,
    reranker: CrossEncoderReranker | None = None,
) -> list[dict[str, Any]]:
    candidates = hybrid_retriever.search(query, k=candidate_k, candidate_k=candidate_k)
    return rerank_documents(query=query, docs=candidates, top_k=top_k, reranker=reranker)

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sentence_transformers import CrossEncoder

if TYPE_CHECKING:
    from retrieval.hybrid import HybridRetriever

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def standardize_document_format(doc: dict[str, Any]) -> dict[str, Any]:
    """STEP 8: Standardize document format.
    
    Ensures documents have required fields: text, source, dense_score, sparse_score.
    """
    standardized = {
        "text": doc.get("text", ""),
        "source": doc.get("source", "unknown"),
        "dense_score": doc.get("dense_score", 0.0),
        "sparse_score": doc.get("sparse_score", 0.0),
    }
    # Preserve any additional fields
    for key, value in doc.items():
        if key not in standardized:
            standardized[key] = value
    return standardized


def _resolve_device() -> str:
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
        """STEP 9-10: Rerank documents using cross-encoder with batch prediction.
        
        Args:
            query: The search query
            docs: List of documents with 'text' field
            top_k: Number of top documents to return
            
        Returns:
            Reranked documents with 'rerank_score' field added
        """
        if not docs:
            return []

        # Standardize all documents to ensure consistent format
        docs = [standardize_document_format(doc) for doc in docs]

        # Create query-document pairs for batch prediction
        pairs = [(query, str(doc.get("text", ""))) for doc in docs]
        
        # STEP 10: Batch prediction with device optimization
        scores = self._get_model().predict(pairs, batch_size=16)

        # Add rerank_score to documents and sort
        reranked: list[dict[str, Any]] = []
        for doc, score in zip(docs, scores):
            item = dict(doc)
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return reranked[:top_k]


def rerank_documents(
    query: str,
    docs: list[dict[str, Any]],
    top_k: int = 5,
    reranker: CrossEncoderReranker | None = None,
) -> list[dict[str, Any]]:
    """Rerank documents using cross-encoder model.
    
    Args:
        query: The search query
        docs: List of documents to rerank
        top_k: Number of top results to return
        reranker: Optional CrossEncoderReranker instance
        
    Returns:
        Top-k reranked documents with rerank_score
    """
    model = reranker or CrossEncoderReranker()
    return model.rerank_documents(query=query, docs=docs, top_k=top_k)


def retrieve_with_rerank(
    query: str,
    hybrid_retriever: "HybridRetriever",
    top_k: int = 5,
    candidate_k: int = 20,
    reranker: CrossEncoderReranker | None = None,
) -> list[dict[str, Any]]:
    """STEP 10: Reduce latency by reranking only top candidates.
    
    This is the correct approach: retrieve top_20 candidates, then rerank to top_5.
    NOT reranking all 100 docs (common mistake from STEP 12).
    """
    # Get top candidates from retriever
    candidates = hybrid_retriever.search(query, k=candidate_k, candidate_k=candidate_k)
    
    # Rerank only the smaller pool for latency efficiency
    return rerank_documents(query=query, docs=candidates, top_k=top_k, reranker=reranker)


def test_reranker() -> None:
    """STEP 9: Test reranker independently.
    
    Expected: Relevant ML docs are ranked higher.
    """
    query = "What is gradient descent?"
    docs = [
        {"text": "Gradient descent is an optimization algorithm"},
        {"text": "Dogs are mammals"},
        {"text": "Backpropagation uses gradients"},
    ]
    
    reranker = CrossEncoderReranker()
    results = reranker.rerank_documents(query, docs, top_k=3)
    
    print("\n=== Reranker Test (STEP 9) ===")
    print(f"Query: {query}\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. {r['text'][:50]}... (score: {r['rerank_score']:.3f})")
    print()


if __name__ == "__main__":
    test_reranker()

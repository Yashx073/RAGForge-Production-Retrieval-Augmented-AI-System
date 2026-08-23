import time
from typing import Any

from retrieval.hybrid import HybridRetriever
from retrieval.rerank import CrossEncoderReranker, retrieve_with_rerank
from generation.llm import generate_answer
from ingestion.embed import chunk_documents


class RAGService:
    def __init__(
        self,
        data_path: str = "data",
        dense_weight: float = 0.6,
        bm25_weight: float = 0.4,
        reranker_model: str = "BAAI/bge-reranker-large",
        candidate_k: int = 20,
    ):
        self.data_path = data_path
        self.dense_weight = dense_weight
        self.bm25_weight = bm25_weight
        self.reranker_model = reranker_model
        self.candidate_k = candidate_k
        self.retriever: HybridRetriever | None = None
        self.reranker: CrossEncoderReranker | None = None
        self._initialized = False

    def initialize(self) -> None:
        """Load or build the RAG index."""
        if self._initialized:
            return

        print("Initializing RAG service...")
        self.retriever = HybridRetriever(
            dense_weight=self.dense_weight,
            bm25_weight=self.bm25_weight,
        )

        try:
            chunks = chunk_documents(self.data_path)
            self.retriever.build_from_data_path(self.data_path)
            print(f"Built index with {len(chunks)} chunks from {self.data_path}")
        except Exception as e:
            print(f"Warning: Could not build index from {self.data_path}: {e}")
            print("Run ingestion pipeline first to create index")

        self.reranker = CrossEncoderReranker(model_name=self.reranker_model)
        self._initialized = True
        print("RAG service initialized")

    def rebuild(self) -> None:
        """Force a full index rebuild (e.g. after documents change)."""
        self._initialized = False
        self.retriever = None
        self.initialize()

    def query(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[str, list[dict[str, Any]], dict[str, float]]:
        """
        Execute the full RAG pipeline.
        
        Returns:
            answer: The generated answer
            sources: List of source documents with metadata
            latencies: Dict with timing for each stage
        """
        if not self._initialized:
            self.initialize()

        latencies = {}

        # Retrieval
        start = time.perf_counter()
        if self.retriever is None:
            raise RuntimeError("Retriever not initialized. Run ingestion first.")
        
        # Use retrieve_with_rerank for hybrid + rerank
        results = retrieve_with_rerank(
            query=query,
            hybrid_retriever=self.retriever,
            top_k=top_k,
            candidate_k=self.candidate_k,
            reranker=self.reranker,
        )
        latencies["retrieval_ms"] = (time.perf_counter() - start) * 1000

        # Extract context for generation
        contexts = [result.get("text", "") for result in results]

        # Generation
        start = time.perf_counter()
        answer, token_counts = generate_answer(
            query=query,
            retrieved_chunks=contexts,
            prompt_type="citation",
            timeout_seconds=120.0,
        )
        latencies["generation_ms"] = (time.perf_counter() - start) * 1000

        latencies["total_ms"] = sum(latencies.values())

        # Format sources
        sources = []
        for i, result in enumerate(results):
            sources.append({
                "document_id": result.get("metadata", {}).get("source", "unknown"),
                "chunk_id": str(result.get("id", i)),
                "text": result.get("text", "")[:500],
                "score": result.get("rerank_score", result.get("score", 0.0)),
            })

        return answer, sources, latencies


# Global instance
rag_service = RAGService()
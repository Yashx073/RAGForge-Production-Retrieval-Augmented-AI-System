import json
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from generation.llm import generate_answer, generate_with_prompt
from evaluation.metrics import faithfulness_score, hallucination_flag, mrr, precision_at_k
from retrieval.hybrid import HybridRetriever
from retrieval.rerank import rerank_documents
from retrieval.dense_faiss import DenseRetriever
from retrieval.bm25 import BM25Retriever
from observability.latency import LatencyTracker, BaselineCollector, calculate_percentiles
from observability.cost import calculate_cost, estimate_tokens, PricingConfig
from ingestion.embed import embed_texts, chunk_documents
import numpy as np
import faiss


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "sample"
DATASET_PATH = BASE_DIR / "evaluation" / "dataset.json"
BASELINE_PATH = BASE_DIR / "evaluation" / "baseline.csv"
BASELINE_JSON_PATH = BASE_DIR / "evaluation" / "baseline.json"
PRICING_PATH = BASE_DIR / "config" / "pricing.yaml"


def load_dataset():
    with open(DATASET_PATH, encoding="utf-8") as f:
        return json.load(f)


class InstrumentedRetriever:
    """Wrapper around HybridRetriever that instruments each stage."""
    
    def __init__(self, data_path: str, strategy: str = "fixed", chunk_size: int = 512):
        self.tracker = LatencyTracker()
        
        # Build index (not timed - one-time cost)
        self.dense = DenseRetriever()
        self.bm25: BM25Retriever | None = None
        self.documents: list[dict[str, Any]] = []
        self.dense_weight = 0.6
        self.bm25_weight = 0.4
        
        chunks = chunk_documents(
            data_path=data_path,
            strategy=strategy,
            chunk_size=chunk_size,
            chunk_overlap=50,
            min_chunk_chars=20,
        )
        chunk_texts = [chunk["text"] for chunk in chunks]
        
        # Build BM25
        from retrieval.bm25 import build_documents_from_chunks
        self.documents = build_documents_from_chunks(chunk_texts)
        self.bm25 = BM25Retriever(self.documents)
        
        # Build dense index
        self.dense.build_index(self.documents)
        self.id_to_doc = {int(doc["id"]): doc for doc in self.documents}

    def _normalize(self, scores: list[float]) -> list[float]:
        if not scores:
            return []
        min_s = min(scores)
        max_s = max(scores)
        denom = max_s - min_s
        if denom == 0:
            return [1.0 for _ in scores]
        return [(s - min_s) / denom for s in scores]

    def search(self, query: str, k: int = 5, candidate_k: int = 20) -> list[dict[str, Any]]:
        self.tracker = LatencyTracker()
        
        # Stage 1: Query embedding
        self.tracker.start("embedding")
        query_embedding = np.asarray(embed_texts([query]), dtype="float32")
        faiss.normalize_L2(query_embedding)
        self.tracker.stop("embedding")
        
        # Stage 2: FAISS search
        self.tracker.start("faiss_search")
        limit = min(candidate_k, self.dense.index.ntotal)
        scores, indices = self.dense.index.search(query_embedding, limit)
        self.tracker.stop("faiss_search")
        
        dense_results = []
        for rank, doc_index in enumerate(indices[0]):
            if doc_index < 0:
                continue
            item = dict(self.dense.metadata[doc_index])
            item["score"] = float(scores[0][rank])
            dense_results.append(item)
        
        # Stage 3: BM25 search
        self.tracker.start("bm25_search")
        bm25_results = self.bm25.search(query, k=candidate_k)
        self.tracker.stop("bm25_search")
        
        # Stage 4: Score fusion
        self.tracker.start("fusion")
        dense_scores = self._normalize([float(r["score"]) for r in dense_results])
        bm25_scores = self._normalize([float(r["bm25_score"]) for r in bm25_results])
        
        combined: dict[int, dict[str, float]] = {}
        for idx, result in enumerate(dense_results):
            doc_id = int(result["id"])
            combined.setdefault(doc_id, {"dense": 0.0, "bm25": 0.0})
            combined[doc_id]["dense"] = dense_scores[idx]
        for idx, result in enumerate(bm25_results):
            doc_id = int(result["id"])
            combined.setdefault(doc_id, {"dense": 0.0, "bm25": 0.0})
            combined[doc_id]["bm25"] = bm25_scores[idx]
        
        merged: list[dict[str, Any]] = []
        for doc_id, score_parts in combined.items():
            hybrid_score = (
                self.dense_weight * score_parts["dense"]
                + self.bm25_weight * score_parts["bm25"]
            )
            doc = self.id_to_doc.get(doc_id, {"id": doc_id, "text": ""})
            merged.append({
                "id": doc_id,
                "score": float(hybrid_score),
                "text": doc.get("text", ""),
            })
        
        merged.sort(key=lambda item: item["score"], reverse=True)
        self.tracker.stop("fusion")
        
        return merged[:k]


def run_baseline(num_queries: int = 50) -> BaselineCollector:
    print(f"=== Running Baseline Measurement ({num_queries} queries) ===")

    retriever = InstrumentedRetriever(str(DATA_PATH), strategy="fixed", chunk_size=512)
    dataset = load_dataset()

    if num_queries and num_queries < len(dataset):
        dataset = dataset[:num_queries]

    collector = BaselineCollector()
    pricing = PricingConfig.from_yaml(str(PRICING_PATH))

    for i, item in enumerate(dataset):
        question = item["question"]
        print(f"\n[{i+1}/{len(dataset)}] Evaluating: {question}")

        # Retrieval (includes embedding, FAISS, BM25, fusion)
        retrieved_docs = retriever.search(question, k=5, candidate_k=20)
        retrieval_latency = retriever.tracker.get_results()
        
        retrieved_texts = [doc["text"] for doc in retrieved_docs]

        # Stage: Reranking
        tracker = LatencyTracker()
        tracker.start("reranking")
        reranked_docs = rerank_documents(question, retrieved_docs, top_k=5)
        tracker.stop("reranking")
        rerank_latency = tracker.get_results()

        reranked_texts = [doc["text"] for doc in reranked_docs]

        # Stage: Generation
        tracker = LatencyTracker()
        tracker.start("generation")
        answer, token_info = generate_answer(
            question,
            reranked_texts,
            prompt_type="citation",
            timeout_seconds=60.0,
        )
        tracker.stop("generation")
        gen_latency = tracker.get_results()

        context = "\n".join(reranked_texts)

        # Evaluation metrics
        source_doc = item.get("source_doc", "")
        precision = precision_at_k(reranked_docs, source_doc, k=5)
        reciprocal_rank = mrr(reranked_docs, source_doc)

        try:
            faithfulness = faithfulness_score(question, answer, context, timeout_seconds=60.0)
        except Exception as exc:
            print(f"  Faithfulness call failed: {exc}, using default score 3")
            faithfulness = 3

        hallucination = hallucination_flag(faithfulness)

        # Token counting - use actual tokens from Ollama
        input_tokens = token_info["input"]
        output_tokens = token_info["output"]

        # Cost calculation
        costs = calculate_cost(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            pricing=pricing,
        )

        # Combine all latencies
        total_latency = {
            "embedding_ms": retrieval_latency.get("embedding", 0),
            "faiss_search_ms": retrieval_latency.get("faiss_search", 0),
            "bm25_search_ms": retrieval_latency.get("bm25_search", 0),
            "fusion_ms": retrieval_latency.get("fusion", 0),
            "retrieval_ms": sum([
                retrieval_latency.get("embedding", 0),
                retrieval_latency.get("faiss_search", 0),
                retrieval_latency.get("bm25_search", 0),
                retrieval_latency.get("fusion", 0),
            ]),
            "reranking_ms": rerank_latency.get("reranking", 0),
            "generation_ms": gen_latency.get("generation", 0),
            "total_ms": 0,  # calculated below
        }
        total_latency["total_ms"] = sum([
            total_latency["retrieval_ms"],
            total_latency["reranking_ms"],
            total_latency["generation_ms"],
        ])

        record = {
            "query": question,
            "answer": answer,
            "latency": total_latency,
            "tokens": {
                "input": input_tokens,
                "output": output_tokens,
                "total": input_tokens + output_tokens,
            },
            "cost": costs,
            "retrieved_chunks": len(reranked_docs),
            "precision@5": precision,
            "mrr": reciprocal_rank,
            "faithfulness": faithfulness,
            "hallucination": hallucination,
        }

        collector.add_record(record)

        # Print progress
        print(f"  Total: {total_latency['total_ms']:.1f} ms")
        print(f"  Embedding: {total_latency['embedding_ms']:.1f} ms")
        print(f"  FAISS: {total_latency['faiss_search_ms']:.1f} ms")
        print(f"  BM25: {total_latency['bm25_search_ms']:.1f} ms")
        print(f"  Fusion: {total_latency['fusion_ms']:.1f} ms")
        print(f"  Retrieval total: {total_latency['retrieval_ms']:.1f} ms")
        print(f"  Reranking: {total_latency['reranking_ms']:.1f} ms")
        print(f"  Generation: {total_latency['generation_ms']:.1f} ms")
        print(f"  Tokens: in={input_tokens} out={output_tokens}")
        print(f"  Cost: ${costs['total_cost']:.6f}")

    return collector


def save_baseline(collector: BaselineCollector) -> None:
    collector.to_csv(str(BASELINE_PATH))
    print(f"\nSaved baseline CSV to {BASELINE_PATH}")

    summary = collector.summarize()
    with open(BASELINE_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "records": collector.records,
        }, f, indent=2, ensure_ascii=False)
    print(f"Saved baseline JSON to {BASELINE_JSON_PATH}")


def print_summary(collector: BaselineCollector) -> None:
    summary = collector.summarize()
    print("\n" + "=" * 60)
    print("BASELINE SUMMARY")
    print("=" * 60)
    print(f"Queries: {summary.get('count', 0)}")

    latency = summary.get("latency", {})
    print(f"\nLatency (ms):")
    for stage, stats in latency.items():
        if isinstance(stats, dict) and "avg" in stats:
            print(f"  {stage}: avg={stats['avg']:.1f} p50={stats['p50']:.1f} "
                  f"p95={stats['p95']:.1f} p99={stats['p99']:.1f} "
                  f"min={stats['min']:.1f} max={stats['max']:.1f}")

    tokens = summary.get("tokens", {})
    print(f"\nTokens:")
    print(f"  Avg input: {tokens.get('avg_input', 0):.0f}")
    print(f"  Avg output: {tokens.get('avg_output', 0):.0f}")
    print(f"  Total input: {tokens.get('total_input', 0)}")
    print(f"  Total output: {tokens.get('total_output', 0)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run baseline measurement")
    parser.add_argument("--num-queries", type=int, default=50,
                        help="Number of queries to run (default: 50)")
    args = parser.parse_args()

    collector = run_baseline(num_queries=args.num_queries)
    save_baseline(collector)
    print_summary(collector)
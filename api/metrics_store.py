import json
import time
from pathlib import Path

METRICS_FILE = Path("data/processed/metrics.json")
METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

# Reference cloud-equivalent pricing (USD per 1M tokens), used to estimate
# inference-equivalent cost of local models.
INPUT_COST_PER_MTOK = 0.15
OUTPUT_COST_PER_MTOK = 0.60
EMBEDDING_COST_PER_MTOK = 0.02
RERANK_COST_PER_QUERY = 0.0008


def _load() -> list[dict]:
    if METRICS_FILE.exists():
        try:
            return json.loads(METRICS_FILE.read_text())
        except Exception:
            return []
    return []


def _save(records: list[dict]) -> None:
    METRICS_FILE.write_text(json.dumps(records[-10000:], indent=2))


def record_query(
    query: str,
    latencies: dict[str, float],
    tokens: dict[str, int],
    num_sources: int,
) -> None:
    records = _load()
    input_tokens = tokens.get("input", 0)
    output_tokens = tokens.get("output", 0)
    embedding_tokens = int(latencies.get("retrieval_ms", 0) > 0) * (len(query) // 4)

    cost_input = input_tokens / 1_000_000 * INPUT_COST_PER_MTOK
    cost_output = output_tokens / 1_000_000 * OUTPUT_COST_PER_MTOK
    cost_embedding = embedding_tokens / 1_000_000 * EMBEDDING_COST_PER_MTOK
    cost_rerank = RERANK_COST_PER_QUERY
    cost_llm = cost_input + cost_output

    records.append({
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "query": query,
        "latency_ms": latencies.get("total_ms", 0.0),
        "retrieval_ms": latencies.get("retrieval_ms", 0.0),
        "generation_ms": latencies.get("generation_ms", 0.0),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": tokens.get("total", 0),
        "num_sources": num_sources,
        "cost_llm": round(cost_llm, 6),
        "cost_embedding": round(cost_embedding, 6),
        "cost_rerank": round(cost_rerank, 6),
        "cost_total": round(cost_llm + cost_embedding + cost_rerank, 6),
    })
    _save(records)


def get_records() -> list[dict]:
    return _load()


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(int(len(values) * p / 100), len(values) - 1)
    return values[idx]


def performance_summary() -> dict:
    records = get_records()
    if not records:
        return {"total_requests": 0}

    totals = [r["latency_ms"] for r in records]
    retrievals = [r["retrieval_ms"] for r in records]
    generations = [r["generation_ms"] for r in records]

    return {
        "total_requests": len(records),
        "avg_latency_ms": sum(totals) / len(totals),
        "percentiles": {
            "p50": percentile(totals, 50),
            "p95": percentile(totals, 95),
            "p99": percentile(totals, 99),
        },
        "avg_retrieval_ms": sum(retrievals) / len(retrievals),
        "avg_generation_ms": sum(generations) / len(generations),
    }


def cost_summary() -> dict:
    records = get_records()
    if not records:
        return {"total_queries": 0}

    total_cost = sum(r["cost_total"] for r in records)
    total_tokens = sum(r["total_tokens"] for r in records)
    llm = sum(r["cost_llm"] for r in records)
    emb = sum(r["cost_embedding"] for r in records)
    rer = sum(r["cost_rerank"] for r in records)

    return {
        "total_queries": len(records),
        "avg_cost_per_query": total_cost / len(records),
        "monthly_equivalent": total_cost / len(records) * 38_500,
        "total_tokens": total_tokens,
        "cost_breakdown": {
            "llm": llm,
            "embeddings": emb,
            "reranking": rer,
        },
    }

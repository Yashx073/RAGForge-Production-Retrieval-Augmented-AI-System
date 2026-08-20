import time
from typing import Any
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class StageTiming:
    name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0

    def start(self) -> None:
        self.start_time = time.perf_counter()

    def stop(self) -> None:
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000


class LatencyTracker:
    def __init__(self) -> None:
        self.stages: dict[str, StageTiming] = {}
        self._order: list[str] = []

    def start(self, name: str) -> None:
        if name not in self.stages:
            self.stages[name] = StageTiming(name=name)
            self._order.append(name)
        self.stages[name].start()

    def stop(self, name: str) -> float:
        if name in self.stages:
            self.stages[name].stop()
            return self.stages[name].duration_ms
        return 0.0

    def get_duration(self, name: str) -> float:
        if name in self.stages:
            return self.stages[name].duration_ms
        return 0.0

    def get_results(self) -> dict[str, float]:
        return {
            name: self.stages[name].duration_ms
            for name in self._order
            if self.stages[name].duration_ms > 0
        }

    def get_total_ms(self) -> float:
        return sum(self.get_results().values())

    def reset(self) -> None:
        self.stages.clear()
        self._order.clear()

    def summary(self) -> dict[str, Any]:
        results = self.get_results()
        return {
            "stages": results,
            "total_ms": self.get_total_ms(),
            "stage_order": self._order,
        }


def percentile(latencies: list[float], p: float) -> float:
    if not latencies:
        return 0.0
    sorted_latencies = sorted(latencies)
    index = int(len(sorted_latencies) * p / 100)
    index = min(index, len(sorted_latencies) - 1)
    return sorted_latencies[index]


def calculate_percentiles(latencies: list[float]) -> dict[str, float]:
    return {
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "p99": percentile(latencies, 99),
        "avg": sum(latencies) / len(latencies) if latencies else 0.0,
        "min": min(latencies) if latencies else 0.0,
        "max": max(latencies) if latencies else 0.0,
    }


class BaselineCollector:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def add_record(self, record: dict[str, Any]) -> None:
        self.records.append(record)

    def get_latencies(self, stage: str) -> list[float]:
        return [r["latency"].get(stage, 0.0) for r in self.records if "latency" in r]

    def get_totals(self) -> list[float]:
        return [r["latency"].get("total_ms", 0.0) for r in self.records if "latency" in r]

    def get_token_totals(self) -> list[tuple[int, int]]:
        return [
            (r["tokens"].get("input", 0), r["tokens"].get("output", 0))
            for r in self.records
            if "tokens" in r
        ]

    def summarize(self) -> dict[str, Any]:
        if not self.records:
            return {}

        total_latencies = self.get_totals()
        input_tokens, output_tokens = zip(*self.get_token_totals()) if self.get_token_totals() else ([], [])

        return {
            "count": len(self.records),
            "latency": {
                "total": calculate_percentiles(total_latencies),
                "embedding": calculate_percentiles(self.get_latencies("embedding_ms")),
                "retrieval": calculate_percentiles(self.get_latencies("retrieval_ms")),
                "reranking": calculate_percentiles(self.get_latencies("reranking_ms")),
                "generation": calculate_percentiles(self.get_latencies("generation_ms")),
            },
            "tokens": {
                "avg_input": sum(input_tokens) / len(input_tokens) if input_tokens else 0,
                "avg_output": sum(output_tokens) / len(output_tokens) if output_tokens else 0,
                "total_input": sum(input_tokens),
                "total_output": sum(output_tokens),
            },
        }

    def to_csv(self, path: str) -> None:
        import csv
        if not self.records:
            return

        fieldnames = [
            "query", "answer",
            "embedding_ms", "retrieval_ms", "reranking_ms", "generation_ms", "total_ms",
            "input_tokens", "output_tokens",
            "retrieved_chunks",
            "precision@5", "mrr", "faithfulness", "hallucination"
        ]

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.records:
                row = {
                    "query": r.get("query", ""),
                    "answer": r.get("answer", "")[:200],
                    "embedding_ms": r.get("latency", {}).get("embedding_ms", 0),
                    "retrieval_ms": r.get("latency", {}).get("retrieval_ms", 0),
                    "reranking_ms": r.get("latency", {}).get("reranking_ms", 0),
                    "generation_ms": r.get("latency", {}).get("generation_ms", 0),
                    "total_ms": r.get("latency", {}).get("total_ms", 0),
                    "input_tokens": r.get("tokens", {}).get("input", 0),
                    "output_tokens": r.get("tokens", {}).get("output", 0),
                    "retrieved_chunks": r.get("retrieved_chunks", 0),
                    "precision@5": r.get("precision@5", 0),
                    "mrr": r.get("mrr", 0),
                    "faithfulness": r.get("faithfulness", 0),
                    "hallucination": r.get("hallucination", 0),
                }
                writer.writerow(row)
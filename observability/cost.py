from typing import Any
from dataclasses import dataclass
import yaml
from pathlib import Path


@dataclass
class PricingConfig:
    llm_input_cost_per_million: float = 0.0
    llm_output_cost_per_million: float = 0.0
    embedding_cost_per_million: float = 0.0
    reranker_cost_per_million: float = 0.0

    @classmethod
    def from_yaml(cls, path: str) -> "PricingConfig":
        p = Path(path)
        if p.exists():
            with open(p) as f:
                data = yaml.safe_load(f) or {}
                llm = data.get("llm", {})
                embedding = data.get("embedding", {})
                reranker = data.get("reranker", {})
                return cls(
                    llm_input_cost_per_million=llm.get("input_cost_per_million", 0.0),
                    llm_output_cost_per_million=llm.get("output_cost_per_million", 0.0),
                    embedding_cost_per_million=embedding.get("cost_per_million", 0.0),
                    reranker_cost_per_million=reranker.get("cost_per_million", 0.0),
                )
        return cls()


DEFAULT_PRICING = PricingConfig(
    llm_input_cost_per_million=0.0,       # Local model - no cost
    llm_output_cost_per_million=0.0,
    embedding_cost_per_million=0.0,       # Local model - no cost
    reranker_cost_per_million=0.0,        # Local model - no cost
)


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    embedding_tokens: int = 0,
    reranker_tokens: int = 0,
    pricing: PricingConfig = DEFAULT_PRICING,
) -> dict[str, float]:
    llm_input_cost = (input_tokens / 1_000_000) * pricing.llm_input_cost_per_million
    llm_output_cost = (output_tokens / 1_000_000) * pricing.llm_output_cost_per_million
    embedding_cost = (embedding_tokens / 1_000_000) * pricing.embedding_cost_per_million
    reranker_cost = (reranker_tokens / 1_000_000) * pricing.reranker_cost_per_million

    total = llm_input_cost + llm_output_cost + embedding_cost + reranker_cost

    return {
        "llm_input_cost": llm_input_cost,
        "llm_output_cost": llm_output_cost,
        "embedding_cost": embedding_cost,
        "reranker_cost": reranker_cost,
        "total_cost": total,
    }


def estimate_tokens(text: str) -> int:
    """Rough token estimation: ~4 chars per token for English"""
    return max(1, len(text) // 4)


def count_tokens_from_messages(messages: list[dict[str, str]]) -> tuple[int, int]:
    """Count input/output tokens from message format"""
    total_input = 0
    total_output = 0
    for msg in messages:
        tokens = estimate_tokens(msg.get("content", ""))
        if msg.get("role") in ("system", "user"):
            total_input += tokens
        elif msg.get("role") == "assistant":
            total_output += tokens
    return total_input, total_output


def count_tokens_from_prompt(prompt: str, response: str) -> tuple[int, int]:
    """Count tokens from prompt and response strings"""
    return estimate_tokens(prompt), estimate_tokens(response)
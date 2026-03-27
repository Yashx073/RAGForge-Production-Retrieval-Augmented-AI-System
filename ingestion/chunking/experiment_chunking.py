from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

try:
    from .chunk_config import ChunkConfig, ExperimentChunkConfig, build_chunker
    from ..loader import load_documents
except ImportError:
    try:
        from ingestion.chunking.chunk_config import (
            ChunkConfig,
            ExperimentChunkConfig,
            build_chunker,
        )
        from ingestion.loader import load_documents
    except ImportError:
        from chunk_config import ChunkConfig, ExperimentChunkConfig, build_chunker
        from loader import load_documents


def _chunk_length_stats(chunk_texts: list[str]) -> dict[str, float | int]:
    if not chunk_texts:
        return {
            "count": 0,
            "avg_chars": 0,
            "min_chars": 0,
            "max_chars": 0,
        }

    lengths = [len(text) for text in chunk_texts]
    return {
        "count": len(lengths),
        "avg_chars": round(mean(lengths), 2),
        "min_chars": min(lengths),
        "max_chars": max(lengths),
    }


def run_chunking_experiment(documents: list[Any], config: ExperimentChunkConfig) -> dict[str, Any]:
    strategy_results: dict[str, dict[str, Any]] = {}

    for strategy_cfg in config.strategies:
        chunker = build_chunker(strategy_cfg)
        strategy_key = (
            f"{strategy_cfg.strategy}_{strategy_cfg.chunk_size}_{strategy_cfg.chunk_overlap}"
        )
        per_doc = []

        for doc in sorted(documents, key=lambda item: getattr(item, "doc_id", "")):
            chunks = chunker.chunk_document(doc)
            chunk_texts = [chunk.text for chunk in chunks]
            per_doc.append(
                {
                    "doc_id": getattr(doc, "doc_id", "unknown_doc"),
                    "chunk_count": len(chunks),
                    "stats": _chunk_length_stats(chunk_texts),
                    "chunks": [
                        {
                            "chunk_id": chunk.chunk_id,
                            "text": chunk.text,
                            "metadata": chunk.metadata,
                        }
                        for chunk in chunks
                    ],
                }
            )

        strategy_results[strategy_key] = {
            "strategy_config": {
                "strategy": strategy_cfg.strategy,
                "chunk_size": strategy_cfg.chunk_size,
                "chunk_overlap": strategy_cfg.chunk_overlap,
                "min_chunk_chars": strategy_cfg.min_chunk_chars,
                "separators": strategy_cfg.separators,
            },
            "documents": per_doc,
        }

    return strategy_results


def default_experiment_config() -> ExperimentChunkConfig:
    return ExperimentChunkConfig(
        strategies=[
            ChunkConfig(strategy="fixed", chunk_size=512, chunk_overlap=50, min_chunk_chars=20),
            ChunkConfig(
                strategy="recursive",
                chunk_size=512,
                chunk_overlap=50,
                min_chunk_chars=20,
                separators=["\n\n", "\n", ". ", " "],
            ),
        ]
    )


def _to_embedding_payload(experiment_results: dict[str, Any]) -> list[dict[str, Any]]:
    payload = []
    for strategy_result in experiment_results.values():
        for doc_result in strategy_result["documents"]:
            for chunk in doc_result["chunks"]:
                payload.append(
                    {
                        "id": chunk["chunk_id"],
                        "text": chunk["text"],
                        "metadata": chunk["metadata"],
                    }
                )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run chunking strategy experiments.")
    parser.add_argument(
        "--data-path",
        default="data/sample",
        help="Directory containing source documents.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional JSON file path for full experiment output.",
    )
    args = parser.parse_args()

    docs = load_documents(args.data_path)
    config = default_experiment_config()
    results = run_chunking_experiment(docs, config)

    print(f"Loaded docs: {len(docs)}")
    for strategy_name, strategy_result in results.items():
        per_doc_counts = [doc_result["chunk_count"] for doc_result in strategy_result["documents"]]
        total_chunks = sum(per_doc_counts)
        print(
            f"strategy={strategy_name} | docs={len(per_doc_counts)} | total_chunks={total_chunks}"
        )

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "experiment": results,
            "embedding_payload": _to_embedding_payload(results),
        }
        output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
        print(f"Saved experiment output to {output_path}")


if __name__ == "__main__":
    main()
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from generation.llm import generate_answer
from evaluation.metrics import faithfulness_score, hallucination_flag, mrr, precision_at_k
from retrieval.hybrid import HybridRetriever

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "sample"
DATASET_PATH = BASE_DIR / "evaluation" / "dataset.json"
REPORT_PATH = BASE_DIR / "evaluation" / "report.json"


def _average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _build_report(results: list[dict[str, object]]) -> dict[str, object]:
    return {
        "summary": {
            "count": len(results),
            "avg_precision@5": _average([float(item["precision@5"]) for item in results]),
            "avg_mrr": _average([float(item["mrr"]) for item in results]),
            "avg_faithfulness": _average([float(item["faithfulness"]) for item in results]),
            "hallucination_rate": _average([float(item["hallucination"]) for item in results]),
        },
        "results": results,
    }


def _save_report(results: list[dict[str, object]]) -> None:
    REPORT_PATH.write_text(
        json.dumps(_build_report(results), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    retriever = HybridRetriever()
    retriever.build_from_data_path(str(DATA_PATH), strategy="fixed", chunk_size=512)

    with open(DATASET_PATH, encoding="utf-8") as f:
        dataset = json.load(f)

    results: list[dict[str, object]] = []

    for item in dataset:
        question = item["question"]
        print(f"Evaluating: {question}")

        retrieved_docs = retriever.search(question, k=5)
        retrieved_texts = [doc["text"] for doc in retrieved_docs]
        answer = generate_answer(
            question,
            retrieved_texts,
            prompt_type="citation",
            timeout_seconds=10.0,
        )
        context = "\n".join(retrieved_texts)

        source_doc = item.get("source_doc", "")
        precision = precision_at_k(retrieved_docs, source_doc, k=5)
        reciprocal_rank = mrr(retrieved_docs, source_doc)

        try:
            faithfulness = faithfulness_score(question, answer, context)
        except Exception as exc:
            print(f"  Faithfulness call failed: {exc}, using default score 3")
            faithfulness = 3

        hallucination = hallucination_flag(faithfulness)

        results.append(
            {
                "query": question,
                "answer": answer,
                "precision@5": precision,
                "mrr": reciprocal_rank,
                "faithfulness": faithfulness,
                "hallucination": hallucination,
            }
        )

        _save_report(results)

    report = _build_report(results)
    REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
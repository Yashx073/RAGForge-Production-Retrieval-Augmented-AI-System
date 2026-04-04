from __future__ import annotations

from typing import Any

from nltk.tokenize import wordpunct_tokenize
from rank_bm25 import BM25Okapi


def build_documents_from_chunks(chunks: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "id": i,
            "text": chunk,
        }
        for i, chunk in enumerate(chunks)
    ]


class BM25Retriever:
    def __init__(self, documents: list[dict[str, Any]]):
        self.documents = documents
        self.tokenized_docs = [self.tokenize(doc["text"]) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return wordpunct_tokenize(text.lower())

    @classmethod
    def from_chunks(cls, chunks: list[str]) -> "BM25Retriever":
        documents = build_documents_from_chunks(chunks)
        return cls(documents)

    def search(self, query: str, k: int = 5) -> list[dict[str, Any]]:
        tokenized_query = self.tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True,
        )

        top_k = ranked_indices[:k]

        return [
            {
                "id": self.documents[i]["id"],
                "text": self.documents[i]["text"],
                "bm25_score": float(scores[i]),
            }
            for i in top_k
        ]

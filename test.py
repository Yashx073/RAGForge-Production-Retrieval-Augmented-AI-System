import json

from retrieval.hybrid import HybridRetriever


queries = [
	"what is cnn architecture",
	"define gradient descent",
	"transformer attention formula",
]


def main() -> None:
	retriever = HybridRetriever()
	retriever.build_from_data_path(data_path="data/sample")

	output = {}
	for query in queries:
		results = retriever.search(query, k=3, candidate_k=10)
		output[query] = [
			{
				"id": item["id"],
				"score": round(float(item["score"]), 4),
				"text": item["text"],
			}
			for item in results
		]

	print(json.dumps(output, indent=2))


if __name__ == "__main__":
	main()

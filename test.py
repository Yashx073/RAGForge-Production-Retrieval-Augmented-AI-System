from __future__ import annotations

import json

import faiss
import numpy as np


with open("data/embeddings.json", "r", encoding="utf-8") as f:
	records = json.load(f)

documents = [{"id": item["id"], "text": item["text"]} for item in records]
embeddings = np.asarray([item["embedding"] for item in records], dtype="float32")

faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

print("vectors in index:", index.ntotal)

# Offline retrieval demo: use the first vector as a query vector.
query_embedding = embeddings[0:1]
scores, indices = index.search(query_embedding, k=min(3, index.ntotal))

print("top indices:", indices[0].tolist())
print("top scores:", scores[0].tolist())
print("top texts:")
for i in indices[0]:
	print("-", documents[i]["text"])
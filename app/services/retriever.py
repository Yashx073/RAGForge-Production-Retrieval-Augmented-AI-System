import faiss
import numpy as np

class VectorStore:
    def __init__(self,dim):
        self.index = faiss.IndexFlatL2(dim)
        self.texts = []

    def add(self, embeddings, texts):
        self.index.add(np.array(embeddings, dtype=np.float32))
        self.texts.extend(texts)

    def search(self, query_embedding, k=5):
        if self.index.ntotal == 0 or not self.texts:
            return []

        k = min(k, self.index.ntotal, len(self.texts))
        if k <= 0:
            return []

        _, indices = self.index.search(np.array([query_embedding], dtype=np.float32), k)
        return [self.texts[i] for i in indices[0] if i >= 0 and i < len(self.texts)]
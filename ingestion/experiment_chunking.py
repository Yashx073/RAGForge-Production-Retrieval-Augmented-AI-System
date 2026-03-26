# experiment_chunking.py

try:
    from .chunking import fixed_chunk, recursive_chunk, sentence_chunk
    from .loader import load_documents
except ImportError:
    from ingestion.chunking import fixed_chunk, recursive_chunk, sentence_chunk
    from ingestion.loader import load_documents


def run_chunking_experiment(doc):
    return {
        "fixed_512": fixed_chunk(doc, 512, 50),
        "recursive": recursive_chunk(doc, 512, 50),
        "sentence_5": sentence_chunk(doc, 5),
    }


if __name__ == "__main__":
    data_path = "/home/yashx073/Desktop/RAGForge—Production-Retrieval-Augmented-AI-System/data/sample"
    docs = load_documents(data_path)

    print(f"Loaded docs: {len(docs)}")
    for i, doc in enumerate(docs, 1):
        exp = run_chunking_experiment(doc)
        print(
            f"{i}. id={doc.doc_id} | fixed={len(exp['fixed_512'])} | "
            f"recursive={len(exp['recursive'])} | sentence={len(exp['sentence_5'])}"
        )
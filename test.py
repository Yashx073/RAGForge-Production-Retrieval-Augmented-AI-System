from ingestion.loader import load_documents

data_path = "/home/yashx073/Desktop/RAGForge—Production-Retrieval-Augmented-AI-System/data/sample"

docs = load_documents(data_path)
print(len(docs))

if docs:
    for i, d in enumerate(docs, 1):
        print(f"{i}. id={d.doc_id}, source={d.metadata.get('source')}, file={d.metadata.get('file_name')}")
else:
    print("No documents loaded. Check data_path and loader outputs.")
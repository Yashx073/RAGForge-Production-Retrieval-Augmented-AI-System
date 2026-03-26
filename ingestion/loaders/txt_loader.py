# ingestion/loaders/txt_loader.py

from pathlib import Path
from ingestion.schema import Document

def load_txt(folder):

    documents = []

    for file in Path(folder).glob("*.txt"):

        text = open(file, encoding="utf-8").read()

        documents.append(
            Document(
                doc_id=file.stem,
                text=text,
                metadata={
                    "source": "txt",
                    "file_name": file.name
                }
            )
        )

    return documents
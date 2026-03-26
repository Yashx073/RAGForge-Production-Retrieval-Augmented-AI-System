# ingestion/loaders/md_loader.py

from pathlib import Path
from ingestion.schema import Document

def load_md(folder):

    docs = []

    for file in Path(folder).glob("*.md"):

        text = open(file, encoding="utf-8").read()

        docs.append(
            Document(
                doc_id=file.stem,
                text=text,
                metadata={
                    "source": "markdown",
                    "file_name": file.name
                }
            )
        )

    return docs
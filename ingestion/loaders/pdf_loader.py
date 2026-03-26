# ingestion/loaders/pdf_loader.py

from pypdf import PdfReader
from ingestion.schema import Document

def load_pdf(file_path):

    reader = PdfReader(file_path)

    docs = []

    for page_num, page in enumerate(reader.pages):

        text = page.extract_text()

        docs.append(
            Document(
                doc_id=f"{file_path}_{page_num}",
                text=text,
                metadata={
                    "source": "pdf",
                    "page": page_num,
                    "file_name": file_path
                }
            )
        )

    return docs
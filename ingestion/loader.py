from pathlib import Path
from pypdf import PdfReader
from typing import Any


def load_pdf(file_path: str):
    reader = PdfReader(file_path)
    pages = []
    for page_number, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append({
                "page": page_number + 1,
                "text": text
            })
    return pages


def load_txt(file_path: str):
    text = Path(file_path).read_text(encoding="utf-8")
    return [{
        "page": None,
        "text": text
    }]


def load_document(file_path: str):
    extension = Path(file_path).suffix.lower()
    if extension == ".pdf":
        return load_pdf(file_path)
    elif extension == ".txt":
        return load_txt(file_path)
    else:
        raise ValueError(f"Unsupported file type: {extension}")


def load_documents(data_path: str) -> list[dict[str, Any]]:
    """Load all supported documents from a directory."""
    path = Path(data_path)
    documents = []
    
    for file_path in path.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() in [".pdf", ".txt", ".md", ".html"]:
            try:
                pages = load_document(str(file_path))
                for page in pages:
                    documents.append({
                        "text": page["text"],
                        "metadata": {
                            "source": str(file_path),
                            "page": page["page"]
                        }
                    })
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    
    return documents
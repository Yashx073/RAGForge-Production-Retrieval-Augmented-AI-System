from pathlib import Path
from pypdf import PdfReader


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
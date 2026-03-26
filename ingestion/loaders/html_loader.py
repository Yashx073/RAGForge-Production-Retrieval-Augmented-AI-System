# ingestion/loaders/html_loader.py

from bs4 import BeautifulSoup
from ingestion.schema import Document
from pathlib import Path

def load_html(file_path):
    file_path = Path(file_path)

    html = open(file_path).read()

    soup = BeautifulSoup(html, "html.parser")

    text = soup.get_text()

    return Document(
        doc_id=file_path.stem,  # e.g. "page"
        text=text,
        metadata={
            "source": "html",
            "file_name": file_path.name,  # e.g. "page.html"
        },
    )
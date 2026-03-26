# ingestion/loader.py

import hashlib
import re
from pathlib import Path

from ingestion.loaders.txt_loader import load_txt
from ingestion.loaders.pdf_loader import load_pdf
from ingestion.loaders.md_loader import load_md
from ingestion.loaders.html_loader import load_html


def _as_list(result):
    if result is None:
        return []
    if isinstance(result, list):
        return result
    return [result]


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _apply_quality_checks(documents):
    filtered = []
    seen_hashes = set()

    for doc in documents:
        text = _clean_text(getattr(doc, "text", ""))

        # skip too-short text
        if len(text.strip()) < 20:
            continue

        # dedupe by content hash
        text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
        if text_hash in seen_hashes:
            continue
        seen_hashes.add(text_hash)

        # write cleaned text back
        try:
            doc.text = text
        except Exception:
            pass

        filtered.append(doc)

    return filtered


def load_documents(data_path):
    documents = []
    txt_dirs_loaded = set()
    md_dirs_loaded = set()

    for file in Path(data_path).rglob("*"):
        if not file.is_file():
            continue

        suffix = file.suffix.lower()

        if suffix == ".txt":
            if file.parent not in txt_dirs_loaded:
                documents.extend(_as_list(load_txt(file.parent)))
                txt_dirs_loaded.add(file.parent)

        elif suffix == ".pdf":
            documents.extend(_as_list(load_pdf(file)))

        elif suffix == ".md":
            if file.parent not in md_dirs_loaded:
                documents.extend(_as_list(load_md(file.parent)))
                md_dirs_loaded.add(file.parent)

        elif suffix == ".html":
            documents.extend(_as_list(load_html(file)))

    return _apply_quality_checks(documents)
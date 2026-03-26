import nltk
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _get_text(obj) -> str:
    if isinstance(obj, str):
        return obj
    return getattr(obj, "text", str(obj))


def fixed_chunk(text_or_doc, size: int = 512, overlap: int = 50):
    text = _get_text(text_or_doc)
    if not text:
        return []

    if overlap >= size:
        overlap = max(0, size - 1)

    chunks = []
    start = 0
    step = size - overlap

    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += step

    return chunks


def recursive_chunk(text_or_doc, chunk_size: int = 512, chunk_overlap: int = 50):
    text = _get_text(text_or_doc)
    if not text:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    return splitter.split_text(text)


def sentence_chunk(text_or_doc, max_sentences: int = 5):
    text = _get_text(text_or_doc)
    if not text:
        return []

    try:
        sentences = nltk.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        sentences = nltk.sent_tokenize(text)

    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunks.append(" ".join(sentences[i : i + max_sentences]))

    return chunks
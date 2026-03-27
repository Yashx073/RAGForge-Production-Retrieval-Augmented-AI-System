import nltk

try:
    from .fixed_chunker import FixedChunker
    from .recursive_chunker import RecursiveChunker
except ImportError:
    from fixed_chunker import FixedChunker
    from recursive_chunker import RecursiveChunker


def _get_text(obj) -> str:
    if isinstance(obj, str):
        return obj
    return getattr(obj, "text", str(obj))


def fixed_chunk(text_or_doc, size: int = 512, overlap: int = 50):
    chunker = FixedChunker(chunk_size=size, chunk_overlap=overlap)
    return [chunk.text for chunk in chunker.chunk_document(text_or_doc)]


def recursive_chunk(text_or_doc, chunk_size: int = 512, chunk_overlap: int = 50):
    chunker = RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [chunk.text for chunk in chunker.chunk_document(text_or_doc)]


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
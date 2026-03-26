from dataclasses import dataclass

@dataclass
class Document:
    doc_id: str
    text: str
    metadata: dict
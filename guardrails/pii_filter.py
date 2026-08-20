import re

PII_PATTERNS = {
    "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    "phone": r"\b\d{10}\b",
    "credit_card": r"\b\d{4}-\d{4}-\d{4}-\d{4}\b",
    "aadhaar": r"\b\d{4}\s\d{4}\s\d{4}\b",
}

def detect_pii(text: str) -> list[str]:
    found = []
    for key, pattern in PII_PATTERNS.items():
        if re.search(pattern, text):
            found.append(key)
    return found
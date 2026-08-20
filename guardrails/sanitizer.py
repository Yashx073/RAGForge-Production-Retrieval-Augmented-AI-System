BLOCKED_CONTEXT_PATTERNS = [
    "system prompt",
    "ignore instructions",
    "assistant must",
]

def sanitize_context(text: str) -> str:
    text_lower = text.lower()
    for pattern in BLOCKED_CONTEXT_PATTERNS:
        if pattern in text_lower:
            return ""
    return text
import re

INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"reveal system prompt",
    r"disregard above",
    r"bypass safety",
    r"developer mode",
    r"pretend to be system",
    r"show hidden prompt",
    r"do anything now",
    r"hidden policy",
    r"system message",
    r"act as",
]

def detect_prompt_injection(text: str) -> bool:
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower):
            return True
    return False

def validate_query(query: str) -> dict:
    if detect_prompt_injection(query):
        return {
            "blocked": True,
            "reason": "Prompt injection attempt detected"
        }
    return {"blocked": False}
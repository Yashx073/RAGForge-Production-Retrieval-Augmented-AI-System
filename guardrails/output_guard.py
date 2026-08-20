from guardrails.pii_filter import detect_pii

def validate_output(answer: str) -> dict:
    pii = detect_pii(answer)
    if pii:
        return {
            "blocked": True,
            "reason": f"PII detected: {pii}"
        }
    return {"blocked": False}
import ollama
from generation.prompt_builder import SYSTEM_PROMPT, build_prompt


def generate_answer(
    query: str,
    retrieved_chunks: list[str],
    model: str = "qwen2.5-coder:7b",
    timeout_seconds: float = 30.0,
    prompt_type: str = "citation",
) -> tuple[str, dict[str, int]]:

    prompt = build_prompt(
        query,
        retrieved_chunks,
        prompt_type=prompt_type,
    )

    try:
        client = ollama.Client(host="http://127.0.0.1:11434", timeout=timeout_seconds)
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.strip(),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
    except Exception as exc:
        return (
            "LLM call failed. Ensure Ollama is running and the model is pulled. "
            f"Details: {exc}",
            {"input": 0, "output": 0, "total": 0}
        )

    # Extract token counts from Ollama response
    input_tokens = response.get("prompt_eval_count", 0)
    output_tokens = response.get("eval_count", 0)

    return response["message"]["content"], {
        "input": input_tokens,
        "output": output_tokens,
        "total": input_tokens + output_tokens,
    }


def generate_with_prompt(
    prompt: str,
    model: str = "qwen2.5-coder:7b",
    timeout_seconds: float = 30.0,
) -> tuple[str, dict[str, int]]:
    """Generate answer using a pre-built prompt (for guardrails integration)."""
    try:
        client = ollama.Client(host="http://127.0.0.1:11434", timeout=timeout_seconds)
        response = client.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT.strip(),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
    except Exception as exc:
        return (
            "LLM call failed. Ensure Ollama is running and the model is pulled. "
            f"Details: {exc}",
            {"input": 0, "output": 0, "total": 0}
        )

    input_tokens = response.get("prompt_eval_count", 0)
    output_tokens = response.get("eval_count", 0)

    return response["message"]["content"], {
        "input": input_tokens,
        "output": output_tokens,
        "total": input_tokens + output_tokens,
    }
import ollama

from generation.prompt_builder import SYSTEM_PROMPT, build_prompt


def generate_answer(
    query: str,
    retrieved_chunks: list[str],
    model: str = "qwen2.5-coder:14b",
    timeout_seconds: float = 30.0,
    prompt_type: str = "citation",
) -> str:

    prompt = build_prompt(
        query,
        retrieved_chunks,
        prompt_type=prompt_type,
    )

    try:
        client = ollama.Client(timeout=timeout_seconds)
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
            f"Details: {exc}"
        )

    return response["message"]["content"]
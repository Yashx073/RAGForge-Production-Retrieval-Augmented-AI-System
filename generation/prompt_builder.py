PROMPT_BASELINE = """
Context:
{context}

Question:
{query}

Answer:
"""

PROMPT_GROUNDED = """
Answer ONLY using the context below.

If answer not present, say:
"I don't have enough information"

Context:
{context}

Question:
{query}

Answer:
"""

PROMPT_CITATION = """
Use ONLY the provided context.

Cite chunk numbers like [1], [2].

Context:
{context}

Question:
{query}

Answer with citations:
"""

PROMPT_JSON = """
Answer using only context.

Return JSON:

{
"answer": "...",
"citations": [numbers],
"confidence": 0-1
}

Context:
{context}

Question:
{query}
"""

PROMPTS = {
    "baseline": PROMPT_BASELINE,
    "grounded": PROMPT_GROUNDED,
    "citation": PROMPT_CITATION,
    "json": PROMPT_JSON,
}

SYSTEM_PROMPT = "You are a grounded AI assistant. Follow the user prompt strictly."

def format_context(chunks: list[str]) -> str:

    formatted = []

    for i, chunk in enumerate(chunks):

        formatted.append(
            f"[{i+1}] {chunk}"
        )

    return "\n\n".join(formatted)

def build_prompt(
    query: str,
    chunks: list[str],
    prompt_type: str = "grounded",
) -> str:

    context = format_context(chunks)

    template = PROMPTS.get(prompt_type)
    if template is None:
        valid_types = ", ".join(PROMPTS.keys())
        raise ValueError(f"Unknown prompt_type '{prompt_type}'. Use one of: {valid_types}")

    return template.format(context=context, query=query)
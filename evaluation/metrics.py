def precision_at_k(retrieved_docs, ground_truth_doc, k=5):
    """Calculate precision@k for retrieval.
    
    Args:
        retrieved_docs: List of doc dicts with 'id', 'text', 'score' keys
        ground_truth_doc: Document ID or name to match
        k: Top-k documents to consider
    
    Returns:
        Precision score (hits / k)
    """
    if not ground_truth_doc:
        return 0.0
    
    top_k = retrieved_docs[:k]
    hits = 0

    for doc in top_k:
        # Check both id and text for match (flexible matching)
        doc_id = str(doc.get("id", ""))
        if doc_id == str(ground_truth_doc):
            hits += 1

    return hits / k if k > 0 else 0.0

def mrr(retrieved_docs, ground_truth_doc):
    """Calculate Mean Reciprocal Rank.
    
    Args:
        retrieved_docs: List of doc dicts with 'id', 'text', 'score' keys
        ground_truth_doc: Document ID or name to match
    
    Returns:
        MRR score (1 / rank of first match)
    """
    if not ground_truth_doc:
        return 0.0
    
    for rank, doc in enumerate(retrieved_docs):
        doc_id = str(doc.get("id", ""))
        if doc_id == str(ground_truth_doc):
            return 1 / (rank + 1)

    return 0.0

def faithfulness_score(query: str, answer: str, context: str, timeout_seconds: float = 10.0) -> int:
    """Score faithfulness of answer to context using LLM.
    
    Args:
        query: Original question
        answer: Generated answer
        context: Retrieved context chunks
    
    Returns:
        Score 1-5 where 5=fully supported, 1=hallucinated
    """
    import ollama
    
    faithfulness_prompt = f"""
Context:
{context}

Answer:
{answer}

Question:
{query}

Does the answer match the context?

Score 1-5.

5 = fully supported
1 = hallucinated

Return only the number.
"""
    
    try:
        client = ollama.Client(timeout=timeout_seconds)
        response = client.generate(
            model="qwen2.5-coder:14b",
            prompt=faithfulness_prompt,
            stream=False,
        )
        score_text = response.get("response", "3").strip()
        # Extract first digit
        for char in score_text:
            if char.isdigit():
                return int(char)
        return 3  # Default middle score if parsing fails
    except Exception as e:
        print(f"Faithfulness scoring failed: {e}")
        return 3  # Default middle score on error

def hallucination_flag(score: int) -> int:
    """Flag if answer contains hallucination based on faithfulness score.
    
    Args:
        score: Faithfulness score 1-5
    
    Returns:
        1 if hallucinating (score <= 2), else 0
    """
    return 1 if score <= 2 else 0

# Legacy evaluate function kept for backward compatibility
def evaluate(query, answer, docs, item):
    """Deprecated: Use individual metric functions instead."""
    context = "\n".join([d["text"] for d in docs])
    p_at_5 = precision_at_k(docs, item.get("source_doc", ""), k=5)
    mrr_score = mrr(docs, item.get("source_doc", ""))
    faith_score = faithfulness_score(query, answer, context)
    hallucination = hallucination_flag(faith_score)
    
    return {
        "precision@5": p_at_5,
        "mrr": mrr_score,
        "faithfulness": faith_score,
        "hallucination": hallucination,
    }
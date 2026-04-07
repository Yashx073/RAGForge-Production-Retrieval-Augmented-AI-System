"""STEP 9: Test reranker independently"""


def test_rerank_documents():
    """Test reranking documents independently.
    
    Expected: relevant ML docs ranked higher.
    """
    from retrieval.rerank import rerank_documents

    query = "What is gradient descent?"
    docs = [
        {"text": "Gradient descent is an optimization algorithm"},
        {"text": "Dogs are mammals"},
        {"text": "Backpropagation uses gradients"},
    ]

    results = rerank_documents(query, docs, top_k=3)

    print("\n=== Test Results ===")
    for r in results:
        print(f"- {r['text']}: {r['rerank_score']:.3f}")
    
    # Assert that ML-related docs are ranked higher
    assert "Gradient" in results[0]["text"] or "gradient" in results[0]["text"]
    assert "Dogs" not in results[0]["text"]
    print("✓ Test passed: Relevant docs ranked higher")


if __name__ == "__main__":
    test_rerank_documents()

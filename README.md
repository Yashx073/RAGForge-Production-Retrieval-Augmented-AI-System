# RAGForge - Production Retrieval-Augmented AI System

A modular Retrieval-Augmented Generation (RAG) project with ingestion, hybrid retrieval, reranking, answer generation, and evaluation workflows.

## Current Repository Structure

```text
.
├── api/
├── data/
│   ├── embeddings.json
│   ├── index_documents.json
│   ├── index_ready.json
│   └── sample/
├── evaluation/
├── generation/
├── guardrails/
├── ingestion/
│   ├── chunking/
│   └── loaders/
├── observability/
├── retrieval/
├── tests/
├── config.yaml
└── requirements.txt
```

## Implemented Modules

- Ingestion
  - Multi-format loading (txt/md/html/pdf)
  - Configurable chunking (fixed + recursive)
  - Sentence-transformers embedding pipeline
- Retrieval
  - Dense retrieval with FAISS
  - Sparse retrieval with BM25
  - Hybrid score fusion
- Reranking
  - Cross-encoder reranker (`BAAI/bge-reranker-large`)
  - Candidate pruning workflow (`top_20` -> rerank -> `top_5`)
- Generation
  - Prompt builder + Ollama-backed response generation
- Evaluation
  - Precision@k, MRR, faithfulness, hallucination rate
  - Report generation to JSON

## Environment Setup

1. Create and activate virtual environment.
2. Install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start

### 1) Run chunking experiments

```bash
prime-run python ingestion/chunking/experiment_chunking.py \
  --data-path data/sample \
  --output evaluation/chunking_experiment_output.json
```

### 2) Build embeddings from documents

```bash
prime-run python ingestion/embed.py \
  --data-path data/sample \
  --strategy fixed \
  --chunk-size 512 \
  --chunk-overlap 50 \
  --output data/embeddings.json
```

### 3) Run reranker test

```bash
prime-run python tests/test_reranker.py
```

### 4) Run full evaluation

```bash
prime-run python evaluation/eval_runner.py
```

## Configuration

Current reranker config in `config.yaml`:

```yaml
reranker:
  enabled: true
  model: bge-reranker-large
  top_k: 5
  rerank_k: 20
```

## Roadmap (Added From Your Plan)

## STEP 5 - Guardrails (Week 24)

### 5.1 Prompt Injection Protection

Detect patterns like:

- `ignore previous instructions`
- `reveal system prompt`

Action: reject or sanitize.

Example:

```python
blocked_patterns = [
    "ignore previous",
    "reveal prompt",
    "system prompt",
]
```

### 5.2 PII Filtering

Regex filters for:

- email
- phone
- aadhaar
- credit card

## STEP 6 - Optimization (Week 25-26)

### 6.1 Cost Calculation

Track:

- input tokens
- output tokens
- embedding tokens

Formula:

```text
total_cost = embedding_cost + generation_cost
```

### 6.2 Semantic Caching

- Cache query-answer pairs in Redis.
- If semantic similarity > 0.95, return cached answer.

### 6.3 Latency Tracking

Measure:

- embedding latency
- retrieval latency
- rerank latency
- generation latency

Store in:

- Prometheus
- Grafana

### 6.4 Quantization Experiments

Test:

- smaller embedding model
- smaller reranker
- context reduction

Compare table:

```text
| model | latency | cost | accuracy |
```

## STEP 7 - Async Ingestion Pipeline (Week 27)

Pipeline:

```text
upload doc
  -> chunk
  -> embed
  -> store
```

FastAPI background task pattern:

```python
@app.post("/upload")
async def upload():
    background_tasks.add_task(process_doc)
```

## STEP 8 - API Layer (Week 27-28)

FastAPI endpoint pattern:

```python
@app.post("/query")
async def query(q):
    results = retrieve(q)
    reranked = rerank(results)
    answer = generate(q, reranked)
    return answer
```

## STEP 9 - Dashboard (Week 28)

Track:

- latency dashboard: `| stage | ms |`
- cost dashboard: `| query | cost |`
- evaluation dashboard: `| metric | score |`

Tools:

- Grafana
- Streamlit

## Final Project Deliverables

Target production layout:

```text
production-rag/
```

Should contain at least:

- ingestion pipeline
- retrieval system (dense + sparse + hybrid)
- reranking module
- generation module
- guardrails
- API layer
- observability (cost + latency)
- evaluation suite
- dashboard integration

## Notes

- API and observability folders are present; implementation can be expanded in upcoming roadmap steps.
- Ensure Ollama is running before generation/evaluation workflows.
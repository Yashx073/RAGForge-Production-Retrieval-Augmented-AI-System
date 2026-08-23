# RAGForge — Production Retrieval-Augmented AI System

A production-grade, fully-local Retrieval-Augmented Generation (RAG) system with a web UI, hybrid retrieval, cross-encoder reranking, guardrails, and observability dashboards — powered by Ollama so no data ever leaves your machine.

---

## 1. Problem Statement

Standard LLM chatbots have three critical production problems:

| Problem | Description |
|---------|-------------|
| **Hallucination** | LLMs generate confident but false answers when they lack knowledge |
| **Stale knowledge** | Model weights are frozen at training time — they can't know your private documents |
| **No accountability** | Answers can't be traced back to verifiable sources |

**RAGForge solves this** by grounding every answer in *your* documents:

1. Ingest documents (PDF / TXT / MD / HTML) → chunk → embed → index
2. At query time, retrieve the most relevant chunks using **hybrid search** (dense FAISS + sparse BM25)
3. **Rerank** candidates with a cross-encoder for precision
4. Generate an answer with a **local LLM, strictly from the retrieved context**, with citations
5. Apply **guardrails** (prompt-injection blocking, PII filtering) on the way in and out
6. Track **latency, token usage, and cloud-equivalent cost** for every query

---

## 2. Architecture

```text
                        ┌─────────────────────────────┐
                        │      WEB UI (React +        │
                        │      Vite + Tailwind)       │
                        │   Chat · Documents · Eval ·  │
                        │   Performance · Cost         │
                        └──────────────┬──────────────┘
                                       │ /api proxy (Vite)
                                       ▼
                        ┌─────────────────────────────┐
                        │        FastAPI :8000         │
                        │  /query /documents /metrics  │
                        │  /evaluation/summary /health │
                        └──────────────┬──────────────┘
                                       │
                 ┌─────────────────────┼─────────────────────┐
                 ▼                     ▼                     ▼
          ┌────────────┐       ┌────────────┐        ┌────────────┐
          │ GUARDRAILS │       │  METRICS   │        │ EVALUATION │
          │ injection  │       │ latency ·  │        │ P@5 · MRR  │
          │ PII filter │       │ tokens ·   │        │ hit-rate   │
          │ sanitizer  │       │ cost       │        │            │
          └────────────┘       └────────────┘        └────────────┘
                 │
                 ▼
     ┌──────────────────────────── RAG PIPELINE ───────────────────────────┐
     │                                                                      │
     │   Query Embedding (nomic-embed-text via Ollama)                      │
     │            │                                                         │
     │      ┌─────┴─────┐                                                   │
     │      ▼           ▼                                                   │
     │   FAISS        BM25        ← hybrid fusion (0.6 dense / 0.4 bm25)    │
     │   (dense)      (sparse)                                              │
     │      └─────┬─────┘                                                   │
     │            ▼                                                         │
     │   top-20 candidates → Cross-Encoder Reranker (bge-reranker-large)    │
     │            │                                                         │
     │            ▼                                                         │
     │   top-5 chunks → Prompt builder (citation template)                  │
     │            │                                                         │
     │            ▼                                                         │
     │   Ollama LLM: qwen2.5-coder:7b → grounded answer with [n] citations  │
     │            │                                                         │
     │            ▼                                                         │
     │   Output guard (PII check) → response                                │
     └──────────────────────────────────────────────────────────────────────┘
```

### Repository structure

```text
.
├── api/                    # FastAPI app, RAG service, schemas, metrics store
├── retrieval/              # Hybrid retriever, FAISS dense, BM25, cross-encoder reranker
├── generation/             # Ollama LLM client, prompt builder (citation/grounded templates)
├── ingestion/              # Loaders (pdf/txt/md/html), chunkers, embedding pipeline
│   └── chunking/           # Fixed vs recursive chunking experiments
├── guardrails/             # Input guard, PII filter, output guard, context sanitizer
├── evaluation/             # Precision@k, MRR, faithfulness, hallucination metrics
├── observability/          # Observability helpers
├── ui/                     # React + Vite + Tailwind web dashboard
├── tests/                  # Pipeline, guardrail, and reranker tests
├── data/
│   ├── sample/             # Sample documents
│   ├── uploads/            # User-uploaded documents
│   └── processed/          # Index artifacts + metrics.json
├── config.yaml             # Reranker configuration
└── requirements.txt
```

---

## 3. How It Works

### Ingestion (upload or startup)

```text
document → loader (PDF/TXT/MD/HTML) → chunker (512 chars, 50 overlap)
        → embeddings (nomic-embed-text, 768-dim via Ollama)
        → FAISS index + BM25 index → ready
```

### Query pipeline

1. **Input guard** — blocks prompt-injection patterns (`ignore previous instructions`, `reveal system prompt`, …)
2. **Hybrid retrieval** — FAISS dense search + BM25 sparse search, scores normalized and fused (`0.6 × dense + 0.4 × bm25`), top-20 candidates
3. **Reranking** — `BAAI/bge-reranker-large` cross-encoder scores each candidate against the query; top-5 survive
4. **Generation** — chunks are formatted into a citation prompt; `qwen2.5-coder:7b` (local via Ollama) answers *only from context*, citing `[1]`, `[2]`, …
5. **Output guard** — generated answer is scanned for PII (emails, phone numbers, credit cards, Aadhaar) before returning
6. **Metrics** — per-stage latency, token counts, and cloud-equivalent cost are recorded for the dashboards

### Safety behavior

| Threat | Defense |
|--------|---------|
| Prompt injection in user query | Input guard rejects the query |
| Malicious content inside documents | Context sanitizer neutralizes it before generation |
| PII leakage in answers | Output guard blocks the response |
| Unanswerable questions | Model responds "I don't have enough information" instead of guessing |

---

## 4. Features

**RAG pipeline**
- Multi-format ingestion: PDF, TXT, MD, HTML
- Fixed + recursive chunking strategies
- Hybrid retrieval: FAISS (dense) + BM25 (sparse) with weighted fusion
- Cross-encoder reranking (top-20 → top-5)
- Citation-grounded generation with a local LLM

**Web UI** (React + Vite + Tailwind)
- 💬 **Chat** — full-height conversation, expandable source cards with rerank scores, latency display, suggestion chips
- 📚 **Documents** — upload / list / search / delete knowledge-base files with automatic re-indexing
- 📊 **Evaluation** — retrieval-only eval (Precision@5, Hit Rate@5, MRR) against a built-in QA set
- ⚡ **Performance** — request count, avg / P50 / P95 / P99 latency, per-stage breakdown
- 💰 **Cost** — token usage and cloud-equivalent cost estimates with per-component breakdown

**API** (FastAPI, interactive docs at `/docs`)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/query` | Run the full RAG pipeline (`{"query": "...", "top_k": 5}`) |
| GET | `/documents` | List knowledge-base documents |
| POST | `/documents` | Upload a file (multipart) and re-index |
| DELETE | `/documents/{id}` | Delete a document and re-index |
| GET | `/evaluation/summary` | Retrieval evaluation metrics |
| GET | `/metrics/performance` | Latency stats and percentiles |
| GET | `/metrics/cost` | Token usage and cost estimates |

**Guardrails** — prompt-injection detection, PII filtering, context sanitization, output validation

---

## 5. Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | React 19, Vite, Tailwind CSS v4 |
| API | FastAPI, Pydantic, Uvicorn |
| LLM inference | Ollama (`qwen2.5-coder:7b`) |
| Embeddings | Ollama (`nomic-embed-text`, 768-dim) |
| Dense retrieval | FAISS (`IndexFlatIP`) |
| Sparse retrieval | BM25 (`rank-bm25`) |
| Reranking | `BAAI/bge-reranker-large` (sentence-transformers) |
| PDF parsing | pypdf |

Everything runs **100% locally** — no API keys, no data leaves the machine.

---

## 6. Getting Started

### Prerequisites

- **Python 3.11+**
- **Node.js 20+** (for the UI)
- **Ollama** — [install](https://ollama.com/download), then pull the models:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5-coder:7b
```

### 1) Start Ollama

```bash
ollama serve          # if not already running as a service
```

### 2) Backend setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3) Start the API

The index is built automatically from `data/` on startup (first start takes a minute — it embeds every document).

```bash
PYTHONPATH=. .venv/bin/uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Verify:

```bash
curl http://127.0.0.1:8000/health
# {"status":"healthy"}

curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is RAGForge?"}'
```

Interactive API docs: **http://127.0.0.1:8000/docs**

### 4) Start the UI

```bash
cd ui
npm install
npm run dev
```

Open **http://localhost:5173** — the Vite dev server proxies `/api/*` to the backend, so no CORS setup is needed.

### 5) (Optional) Run the tests

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ -v
```

---

## 7. Configuration

### `config.yaml` (reranker)

```yaml
reranker:
  enabled: true
  model: bge-reranker-large
  top_k: 5
  rerank_k: 20
```

### Retrieval weights

Hybrid fusion weights are set in `api/rag_service.py`:

```python
RAGService(dense_weight=0.6, bm25_weight=0.4, candidate_k=20)
```

### Cost model

Cloud-equivalent pricing used for local-inference cost estimates lives in `api/metrics_store.py`:

```python
INPUT_COST_PER_MTOK  = 0.15   # USD per 1M input tokens
OUTPUT_COST_PER_MTOK = 0.60   # USD per 1M output tokens
EMBEDDING_COST_PER_MTOK = 0.02
RERANK_COST_PER_QUERY = 0.0008
```

Metrics are persisted to `data/processed/metrics.json` (last 10,000 queries).

---

## 8. Roadmap

- [x] Ingestion pipeline (multi-format, chunking, embeddings)
- [x] Hybrid retrieval (FAISS + BM25) with reranking
- [x] Guardrails (injection, PII, sanitization)
- [x] FastAPI layer with documents management
- [x] Web dashboard (chat, documents, evaluation, performance, cost)
- [x] Latency / token / cost tracking
- [ ] LLM-judged faithfulness & hallucination scoring in the Evaluation page
- [ ] Semantic caching (Redis) for repeated queries
- [ ] Async background ingestion with job progress bars
- [ ] Streaming responses (SSE) in the chat UI
- [ ] Prometheus + Grafana export
- [ ] pgvector as a production vector store alternative to FAISS
- [ ] Quantization experiments (smaller embedder / reranker) with quality comparison

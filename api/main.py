import shutil
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from api.schemas import (
    QueryRequest,
    QueryResponse,
    Source,
    DocumentInfo,
    DocumentsResponse,
    UploadResponse,
)
from api.rag_service import rag_service
from api import metrics_store

DATA_DIR = Path("data")
UPLOAD_DIR = DATA_DIR / "uploads"
SAMPLE_DIR = DATA_DIR / "sample"
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".html"}


def _iter_document_files():
    for directory in (SAMPLE_DIR, UPLOAD_DIR):
        if directory.exists():
            for f in sorted(directory.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                    yield f


app = FastAPI(
    title="Production RAG API",
    version="1.0.0",
    description="Production-grade Retrieval Augmented Generation API with Ollama"
)


@app.on_event("startup")
async def startup_event():
    rag_service.initialize()


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/documents", response_model=DocumentsResponse)
async def list_documents():
    documents = []
    for f in _iter_document_files():
        documents.append(
            DocumentInfo(
                id=str(f),
                name=f.name,
                chunks=0,
                size=f.stat().st_size,
                status="ready",
            )
        )
    return DocumentsResponse(documents=documents)


@app.post("/documents", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / file.filename
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    # Rebuild the index so the new document is searchable
    try:
        rag_service.rebuild()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return UploadResponse(
        success=True,
        document_id=str(dest),
        message=f"Uploaded and indexed {file.filename}",
    )


@app.delete("/documents/{document_id:path}")
async def delete_document(document_id: str):
    path = Path(document_id)
    # Only allow deleting files inside managed directories
    resolved = path.resolve()
    allowed = {SAMPLE_DIR.resolve(), UPLOAD_DIR.resolve()}
    if not any(str(resolved).startswith(str(d)) for d in allowed):
        raise HTTPException(status_code=400, detail="Invalid document id")
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="Document not found")

    resolved.unlink()
    rag_service.rebuild()

    return {"success": True, "message": f"Deleted {resolved.name}"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    start = time.perf_counter()
    try:
        answer, sources, stage_latencies, token_counts = rag_service.query(
            query=request.query,
            top_k=request.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    total_latency = (time.perf_counter() - start) * 1000
    stage_latencies.setdefault("total_ms", total_latency)

    metrics_store.record_query(
        query=request.query,
        latencies=stage_latencies,
        tokens=token_counts,
        num_sources=len(sources),
    )

    # Convert sources to schema
    source_objects = [
        Source(
            document_id=s.get("document_id", "unknown"),
            chunk_id=s.get("chunk_id", "0"),
            text=s.get("text", ""),
            score=s.get("score"),
        )
        for s in sources
    ]

    return QueryResponse(
        answer=answer,
        sources=source_objects,
        latency_ms=total_latency,
    )


EVAL_SET = [
    {"query": "What is RAGForge?", "ground_truth": "intro.txt"},
    {"query": "How do I install RAGForge?", "ground_truth": "faq.md"},
    {"query": "What is the architecture of RAGForge?", "ground_truth": "intro.txt"},
]


@app.get("/evaluation/summary")
async def evaluation_summary():
    """Run retrieval-only evaluation (precision@5, MRR) against a built-in QA set."""
    if not rag_service._initialized:
        rag_service.initialize()
    if rag_service.retriever is None:
        raise HTTPException(status_code=503, detail="Index not built yet")

    from evaluation.metrics import precision_at_k, mrr

    p_scores, mrr_scores = [], []
    for item in EVAL_SET:
        results = rag_service.retriever.search(item["query"], k=5)
        docs = [
            {
                "id": r.get("metadata", {}).get("source", ""),
                "text": r.get("text", ""),
            }
            for r in results
        ]
        gt = next(
            (d["id"] for d in docs if d["id"].endswith(item["ground_truth"])),
            None,
        )
        if gt is None:
            p_scores.append(0.0)
            mrr_scores.append(0.0)
            continue
        p_scores.append(precision_at_k(docs, gt, k=5))
        mrr_scores.append(mrr(docs, gt))

    n = len(EVAL_SET)
    return {
        "precision_at_5": sum(p_scores) / n,
        "recall_at_5": sum(1 for p in p_scores if p > 0) / n,
        "mrr": sum(mrr_scores) / n,
        "total_evaluations": n,
    }


@app.get("/metrics/performance")
async def metrics_performance():
    return metrics_store.performance_summary()


@app.get("/metrics/cost")
async def metrics_cost():
    return metrics_store.cost_summary()
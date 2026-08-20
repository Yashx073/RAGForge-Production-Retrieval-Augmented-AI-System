from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from pydantic import BaseModel
from pathlib import Path
import uuid

from ingestion.pipeline import process_document, get_job_status, update_job_status, load_index
from retrieval.bm25 import BM25Retriever
from retrieval.hybrid import HybridRetriever
from retrieval.rerank import rerank_documents
from generation.llm import generate_with_prompt
from observability.latency import LatencyTracker
from guardrails.pipeline import rag_pipeline

app = FastAPI(title="RAGForge Async Ingestion API")

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class QueryRequest(BaseModel):
    query: str
    job_id: str
    k: int = 5
    use_reranker: bool = True


@app.post("/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    job_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    
    content = await file.read()
    with open(file_path, "wb") as buffer:
        buffer.write(content)
    
    update_job_status(job_id, "queued", filename=file.filename)
    
    background_tasks.add_task(process_document, str(file_path), job_id)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "filename": file.filename
    }


@app.get("/jobs/{job_id}")
async def get_job_status_endpoint(job_id: str):
    job = get_job_status(job_id)
    if "error" in job:
        raise HTTPException(status_code=404, detail=job["error"])
    return job


@app.get("/jobs")
async def list_jobs():
    from ingestion.pipeline import _load_jobs
    jobs = _load_jobs()
    return {
        "jobs": [
            {"job_id": k, **v} for k, v in jobs.items()
        ]
    }


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    from ingestion.pipeline import _load_jobs, _save_jobs
    jobs = _load_jobs()
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    del jobs[job_id]
    _save_jobs(jobs)
    return {"status": "deleted", "job_id": job_id}


@app.post("/query")
async def query_rag(request: QueryRequest):
    job = get_job_status(request.job_id)
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job not completed")
    
    tracker = LatencyTracker()
    
    # Load indexes
    tracker.start("load_index")
    dense_retriever = load_index(request.job_id)
    bm25_path = Path("data/processed/indexes") / f"{request.job_id}_bm25.json"
    bm25_retriever = BM25Retriever.load(str(bm25_path))
    tracker.stop("load_index")
    
    # Create hybrid retriever
    hybrid = HybridRetriever()
    hybrid.dense = dense_retriever
    hybrid.bm25 = bm25_retriever
    hybrid.documents = dense_retriever.metadata
    
    # Retrieve
    tracker.start("retrieval")
    retrieved_docs = hybrid.search(request.query, k=request.k, candidate_k=20)
    tracker.stop("retrieval")
    
    retrieved_texts = [doc["text"] for doc in retrieved_docs]
    
    # Rerank
    if request.use_reranker:
        tracker.start("reranking")
        reranked_docs = rerank_documents(request.query, retrieved_docs, top_k=request.k)
        tracker.stop("reranking")
        reranked_texts = [doc["text"] for doc in reranked_docs]
    else:
        reranked_texts = retrieved_texts
    
    # Generate
    tracker.start("generation")
    
    # Use guardrails pipeline
    def retrieve_fn(q):
        return reranked_docs
    
    answer = rag_pipeline(request.query, retrieve_fn, generate_with_prompt)
    tracker.stop("generation")
    
    latency_results = tracker.get_results()
    latency_results["total_ms"] = tracker.get_total_ms()
    
    return {
        "answer": answer,
        "latency": latency_results,
        "sources": [{"text": d["text"][:200], "score": d.get("score", 0)} for d in reranked_docs[:3]]
    }


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
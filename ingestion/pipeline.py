import json
import os
import uuid
from pathlib import Path
from typing import Any

from ingestion.loader import load_document
from ingestion.chunker import chunk_pages
from ingestion.embed import embed_chunks
from retrieval.bm25 import BM25Retriever, build_documents_from_chunks
from retrieval.dense_faiss import DenseRetriever


JOBS_FILE = Path("data/processed/jobs.json")
INDEX_DIR = Path("data/processed/indexes")
INDEX_DIR.mkdir(parents=True, exist_ok=True)


def _load_jobs() -> dict:
    if JOBS_FILE.exists():
        return json.loads(JOBS_FILE.read_text())
    return {}


def _save_jobs(jobs: dict) -> None:
    JOBS_FILE.write_text(json.dumps(jobs, indent=2))


def update_job_status(job_id: str, status: str, **kwargs) -> None:
    jobs = _load_jobs()
    if job_id not in jobs:
        jobs[job_id] = {}
    jobs[job_id]["status"] = status
    jobs[job_id].update(kwargs)
    _save_jobs(jobs)


def get_job_status(job_id: str) -> dict:
    jobs = _load_jobs()
    return jobs.get(job_id, {"error": "Job not found"})


def save_index(retriever: DenseRetriever, job_id: str) -> None:
    import faiss
    index_path = INDEX_DIR / f"{job_id}.faiss"
    meta_path = INDEX_DIR / f"{job_id}_meta.json"
    faiss.write_index(retriever.index, str(index_path))
    meta_path.write_text(json.dumps(retriever.metadata, ensure_ascii=False, indent=2))


def load_index(job_id: str) -> DenseRetriever:
    import faiss
    index_path = INDEX_DIR / f"{job_id}.faiss"
    meta_path = INDEX_DIR / f"{job_id}_meta.json"
    retriever = DenseRetriever()
    retriever.index = faiss.read_index(str(index_path))
    retriever.metadata = json.loads(meta_path.read_text())
    return retriever


def ingest_document(file_path: str, job_id: str) -> dict[str, Any]:
    update_job_status(job_id, "processing", filename=Path(file_path).name)
    
    try:
        # 1. Load document
        pages = load_document(file_path)
        update_job_status(job_id, "processing", pages=len(pages))
        
        # 2. Chunk
        chunks = chunk_pages(pages)
        update_job_status(job_id, "processing", chunks=len(chunks))
        
        # 3. Generate embeddings
        chunks = embed_chunks(chunks)
        
        # 4. Build retrieval indexes
        retriever = DenseRetriever()
        retriever.build_index(chunks)
        
        # 5. Build BM25
        chunk_texts = [chunk["text"] for chunk in chunks]
        bm25 = BM25Retriever(build_documents_from_chunks(chunk_texts))
        
        # 6. Save indexes
        save_index(retriever, job_id)
        
        # 7. Save BM25
        bm25_path = INDEX_DIR / f"{job_id}_bm25.json"
        bm25.save(str(bm25_path))
        
        update_job_status(job_id, "completed", chunks=len(chunks))
        return {"status": "completed", "chunks": len(chunks), "job_id": job_id}
    
    except Exception as e:
        update_job_status(job_id, "failed", error=str(e))
        raise


def process_document(file_path: str, job_id: str) -> None:
    try:
        ingest_document(file_path, job_id)
    except Exception as e:
        update_job_status(job_id, "failed", error=str(e))
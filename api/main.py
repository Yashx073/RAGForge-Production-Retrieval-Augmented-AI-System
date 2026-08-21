import time
from fastapi import FastAPI, HTTPException
from api.schemas import QueryRequest, QueryResponse, Source
from api.rag_service import rag_service

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


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    start = time.perf_counter()
    try:
        answer, sources, stage_latencies = rag_service.query(
            query=request.query,
            top_k=request.top_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    total_latency = (time.perf_counter() - start) * 1000

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
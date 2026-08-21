from pydantic import BaseModel, Field
from typing import Optional

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)

class Source(BaseModel):
    document_id: str
    chunk_id: str
    text: str
    score: Optional[float] = None

class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
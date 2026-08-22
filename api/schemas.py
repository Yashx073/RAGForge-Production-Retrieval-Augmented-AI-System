from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

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

# Documents
class DocumentInfo(BaseModel):
    id: str
    name: str
    chunks: int
    size: int
    status: str

class DocumentsResponse(BaseModel):
    documents: List[DocumentInfo]

class UploadResponse(BaseModel):
    success: bool
    document_id: str
    message: str

# Evaluation
class EvaluationSummary(BaseModel):
    precision_at_5: Optional[float] = None
    recall_at_5: Optional[float] = None
    mrr: Optional[float] = None
    ndcg_at_5: Optional[float] = None
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    context_relevance: Optional[float] = None
    hallucination_rate: Optional[float] = None
    total_evaluations: int = 0

class EvaluationHistoryItem(BaseModel):
    timestamp: str
    query: str
    precision_at_5: Optional[float] = None
    mrr: Optional[float] = None
    faithfulness: Optional[float] = None
    hallucination_rate: Optional[float] = None

class EvaluationHistoryResponse(BaseModel):
    evaluations: List[EvaluationHistoryItem]

# Performance
class LatencyBreakdown(BaseModel):
    query_embedding_ms: float
    faiss_retrieval_ms: float
    bm25_retrieval_ms: float
    reranking_ms: float
    prompt_construction_ms: float
    llm_generation_ms: float
    total_ms: float

class PercentileLatencies(BaseModel):
    p50: float
    p95: float
    p99: float

class PerformanceResponse(BaseModel):
    total_requests: int
    avg_latency_ms: float
    latency_breakdown: LatencyBreakdown
    percentiles: PercentileLatencies

# Cost
class CostBreakdown(BaseModel):
    embeddings: float
    reranking: float
    llm: float

class CostResponse(BaseModel):
    total_queries: int
    avg_cost_per_query: float
    monthly_equivalent: float
    total_tokens: int
    cost_breakdown: CostBreakdown

# Settings
class RagSettings(BaseModel):
    llm_model: str
    embedding_model: str
    dense_top_k: int
    bm25_top_k: int
    final_top_k: int
    dense_weight: float
    bm25_weight: float
    enable_reranking: bool
    enable_semantic_cache: bool

class SettingsResponse(BaseModel):
    settings: RagSettings
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime


class UploadResponse(BaseModel):
    job_id: str
    status: str
    filename: str


class JobStatus(BaseModel):
    job_id: str
    status: str
    filename: str
    pages: Optional[int] = None
    chunks: Optional[int] = None
    error: Optional[str] = None
    created_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class JobListResponse(BaseModel):
    jobs: List[JobStatus]


class HealthResponse(BaseModel):
    status: str
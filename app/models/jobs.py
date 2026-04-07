from enum import Enum
from typing import Any
from pydantic import BaseModel


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class JobResponse(BaseModel):
    job_id: str
    agent: str
    status: JobStatus
    result: Any = None
    error: str | None = None
    created_at: str
    updated_at: str


class JobCreated(BaseModel):
    job_id: str
    status: JobStatus = JobStatus.pending

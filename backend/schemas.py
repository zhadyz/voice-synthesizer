"""
Pydantic Schemas for API Request/Response Validation
"""

from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, List
import re
import uuid
from backend.models import JobStatus


# Job Schemas
class JobResponse(BaseModel):
    """Job status response"""
    id: str
    user_id: str
    type: str
    status: JobStatus
    progress: float
    created_at: datetime
    completed_at: Optional[datetime] = None
    quality_snr: Optional[float] = None
    quality_pesq: Optional[float] = None
    error_message: Optional[str] = None
    output_audio_path: Optional[str] = None

    class Config:
        from_attributes = True


class JobListResponse(BaseModel):
    """List of jobs"""
    jobs: List[JobResponse]
    total: int


class TrainingStartRequest(BaseModel):
    """Request to start training"""
    job_id: str
    model_name: str = Field(..., min_length=1, max_length=50)

    @field_validator('job_id')
    @classmethod
    def validate_job_id(cls, v: str) -> str:
        """Validate job_id is a valid UUID"""
        try:
            uuid.UUID(v)
        except ValueError:
            raise ValueError("job_id must be a valid UUID")
        return v

    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v: str) -> str:
        """Validate model_name contains only safe characters"""
        if not re.match(r'^[a-zA-Z0-9_-]{1,50}$', v):
            raise ValueError(
                "model_name must contain only alphanumeric characters, "
                "underscores, and hyphens (1-50 characters)"
            )
        return v


class ConversionStartRequest(BaseModel):
    """Request to start voice conversion"""
    job_id: str
    model_id: str
    output_name: str = Field(..., min_length=1, max_length=100)

    @field_validator('job_id', 'model_id')
    @classmethod
    def validate_ids(cls, v: str) -> str:
        """Validate IDs are valid UUIDs or safe strings"""
        # Try UUID first
        try:
            uuid.UUID(v)
            return v
        except ValueError:
            pass

        # Allow alphanumeric with underscores/hyphens
        if re.match(r'^[a-zA-Z0-9_-]{1,100}$', v):
            return v

        raise ValueError("ID must be a valid UUID or safe alphanumeric string")

    @field_validator('output_name')
    @classmethod
    def validate_output_name(cls, v: str) -> str:
        """Validate output_name contains only safe characters"""
        if not re.match(r'^[a-zA-Z0-9_\s-]{1,100}$', v):
            raise ValueError(
                "output_name must contain only alphanumeric characters, "
                "spaces, underscores, and hyphens"
            )
        return v


# Voice Model Schemas
class VoiceModelResponse(BaseModel):
    """Voice model response"""
    id: str
    user_id: str
    model_name: str
    created_at: datetime
    quality_snr: Optional[float] = None
    quality_score: Optional[float] = None
    training_duration_seconds: Optional[float] = None

    class Config:
        from_attributes = True


class VoiceModelListResponse(BaseModel):
    """List of voice models"""
    models: List[VoiceModelResponse]
    total: int


# Upload Schemas
class UploadResponse(BaseModel):
    """File upload response"""
    job_id: str
    filename: str
    size_mb: float
    status: str
    message: str


# Health Check Schema
class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    redis_connected: bool
    database_connected: bool


# Progress Stream Schema
class ProgressUpdate(BaseModel):
    """Real-time progress update"""
    job_id: str
    status: JobStatus
    progress: float
    message: str
    timestamp: datetime

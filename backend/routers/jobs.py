"""
Job Management Endpoints
Start training/conversion jobs and query status
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime

from backend.database import get_db
from backend.models import Job, JobStatus, VoiceModel
from backend.schemas import (
    JobResponse,
    JobListResponse,
    TrainingStartRequest,
    ConversionStartRequest
)

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.post("/train", response_model=JobResponse)
async def start_training(
    request: TrainingStartRequest,
    db: Session = Depends(get_db)
):
    """
    Start RVC training job

    - Validates job exists and is in PENDING state
    - Enqueues preprocessing and training tasks
    - Returns job details
    """

    job = db.query(Job).filter(Job.id == request.job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.type != "training":
        raise HTTPException(status_code=400, detail="Job is not a training job")

    if job.status != JobStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail=f"Job already started or completed. Current status: {job.status}"
        )

    # Update job status
    job.status = JobStatus.PREPROCESSING
    job.updated_at = datetime.utcnow()
    db.commit()

    # TODO: Enqueue ARQ job when worker is ready
    # redis = await create_pool(RedisSettings())
    # await redis.enqueue_job('preprocess_audio', job.id, job.input_audio_path, job.user_id)

    return JobResponse.model_validate(job)


@router.post("/convert", response_model=JobResponse)
async def start_conversion(
    request: ConversionStartRequest,
    db: Session = Depends(get_db)
):
    """
    Start voice conversion job

    - Validates job and model exist
    - Enqueues conversion task
    - Returns job details
    """

    job = db.query(Job).filter(Job.id == request.job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.type != "conversion":
        raise HTTPException(status_code=400, detail="Job is not a conversion job")

    if job.status != JobStatus.PENDING:
        raise HTTPException(
            status_code=400,
            detail=f"Job already started or completed. Current status: {job.status}"
        )

    # Validate model exists
    model = db.query(VoiceModel).filter(VoiceModel.id == request.model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Voice model not found")

    # Update job with model reference
    job.model_id = request.model_id
    job.status = JobStatus.CONVERTING
    job.updated_at = datetime.utcnow()
    db.commit()

    # TODO: Enqueue ARQ job when worker is ready
    # redis = await create_pool(RedisSettings())
    # await redis.enqueue_job('convert_voice', job.id, model.model_path, job.input_audio_path, request.output_name)

    return JobResponse.model_validate(job)


@router.get("/status/{job_id}", response_model=JobResponse)
async def get_job_status(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Get job status and details

    Returns current status, progress, quality metrics, and error information
    """

    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return JobResponse.model_validate(job)


@router.get("/list", response_model=JobListResponse)
async def list_jobs(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    job_type: Optional[str] = Query(None, description="Filter by job type (training/conversion)"),
    status: Optional[JobStatus] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of jobs to return"),
    offset: int = Query(0, ge=0, description="Number of jobs to skip"),
    db: Session = Depends(get_db)
):
    """
    List jobs with optional filters

    - Filter by user_id, job_type, or status
    - Supports pagination with limit and offset
    - Returns jobs sorted by creation date (newest first)
    """

    query = db.query(Job)

    # Apply filters
    if user_id:
        query = query.filter(Job.user_id == user_id)
    if job_type:
        query = query.filter(Job.type == job_type)
    if status:
        query = query.filter(Job.status == status)

    # Get total count
    total = query.count()

    # Apply pagination and sorting
    jobs = query.order_by(Job.created_at.desc()).offset(offset).limit(limit).all()

    return JobListResponse(
        jobs=[JobResponse.model_validate(job) for job in jobs],
        total=total
    )


@router.delete("/{job_id}")
async def cancel_job(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Cancel a running job

    - Only cancels jobs in PENDING, PREPROCESSING, or TRAINING state
    - Cannot cancel COMPLETED or FAILED jobs
    """

    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot cancel job in {job.status} state"
        )

    # Update status to failed with cancellation message
    job.status = JobStatus.FAILED
    job.error_message = "Job cancelled by user"
    job.completed_at = datetime.utcnow()
    job.updated_at = datetime.utcnow()
    db.commit()

    # TODO: Cancel ARQ job if running

    return {"message": "Job cancelled successfully", "job_id": job_id}


@router.post("/{job_id}/retry", response_model=JobResponse)
async def retry_job(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Retry a failed job

    - Only retries jobs in FAILED state
    - Resets status to PENDING
    """

    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.FAILED:
        raise HTTPException(
            status_code=400,
            detail=f"Can only retry failed jobs. Current status: {job.status}"
        )

    # Reset job status
    job.status = JobStatus.PENDING
    job.progress = 0.0
    job.error_message = None
    job.completed_at = None
    job.updated_at = datetime.utcnow()
    db.commit()

    return JobResponse.model_validate(job)

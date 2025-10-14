"""
Server-Sent Events (SSE) for Real-Time Progress
Provides streaming progress updates for long-running jobs
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import asyncio
import json
from datetime import datetime

from backend.database import get_db, SessionLocal
from backend.models import Job, JobStatus

router = APIRouter(prefix="/api/stream", tags=["stream"])


@router.get("/progress/{job_id}")
async def stream_progress(job_id: str):
    """
    SSE endpoint for real-time job progress updates

    - Streams progress updates every 2 seconds
    - Automatically terminates when job completes or fails
    - Use EventSource API on frontend to consume
    """

    async def event_generator():
        """Generate SSE events with job progress"""

        # Verify job exists
        db = SessionLocal()
        job = db.query(Job).filter(Job.id == job_id).first()
        db.close()

        if not job:
            yield f"event: error\ndata: {json.dumps({'error': 'Job not found'})}\n\n"
            return

        # Stream updates
        while True:
            try:
                # Query current job status
                db = SessionLocal()
                job = db.query(Job).filter(Job.id == job_id).first()
                db.close()

                if not job:
                    yield f"event: error\ndata: {json.dumps({'error': 'Job not found'})}\n\n"
                    break

                # Prepare progress data
                data = {
                    "job_id": job.id,
                    "status": job.status.value,
                    "progress": job.progress,
                    "message": _get_status_message(job),
                    "timestamp": datetime.utcnow().isoformat(),
                    "quality_snr": job.quality_snr,
                    "error": job.error_message
                }

                # Send SSE event
                yield f"data: {json.dumps(data)}\n\n"

                # Stop streaming if job is done
                if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                    # Send completion event
                    completion_data = {
                        "job_id": job.id,
                        "status": job.status.value,
                        "final": True,
                        "output_path": job.output_audio_path,
                        "quality_snr": job.quality_snr,
                        "error": job.error_message
                    }
                    yield f"event: complete\ndata: {json.dumps(completion_data)}\n\n"
                    break

                # Wait before next update
                await asyncio.sleep(2)

            except Exception as e:
                error_data = {"error": str(e)}
                yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable buffering in nginx
        }
    )


@router.get("/multi-progress")
async def stream_multi_progress(user_id: str):
    """
    SSE endpoint for monitoring multiple jobs

    - Streams updates for all active jobs for a user
    - Useful for dashboard views
    """

    async def event_generator():
        """Generate SSE events for multiple jobs"""

        while True:
            try:
                db = SessionLocal()

                # Get all active jobs for user
                jobs = db.query(Job).filter(
                    Job.user_id == user_id,
                    Job.status.in_([
                        JobStatus.PENDING,
                        JobStatus.PREPROCESSING,
                        JobStatus.TRAINING,
                        JobStatus.CONVERTING
                    ])
                ).all()

                db.close()

                # Prepare data for all jobs
                jobs_data = []
                for job in jobs:
                    jobs_data.append({
                        "job_id": job.id,
                        "type": job.type,
                        "status": job.status.value,
                        "progress": job.progress,
                        "message": _get_status_message(job)
                    })

                data = {
                    "user_id": user_id,
                    "jobs": jobs_data,
                    "timestamp": datetime.utcnow().isoformat()
                }

                yield f"data: {json.dumps(data)}\n\n"

                # Stop if no active jobs
                if not jobs:
                    break

                await asyncio.sleep(3)

            except Exception as e:
                error_data = {"error": str(e)}
                yield f"event: error\ndata: {json.dumps(error_data)}\n\n"
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


def _get_status_message(job: Job) -> str:
    """Generate human-readable status message"""

    status_messages = {
        JobStatus.PENDING: "Job queued, waiting to start...",
        JobStatus.PREPROCESSING: f"Preprocessing audio... ({int(job.progress * 100)}%)",
        JobStatus.TRAINING: f"Training voice model... ({int(job.progress * 100)}%)",
        JobStatus.CONVERTING: f"Converting audio... ({int(job.progress * 100)}%)",
        JobStatus.COMPLETED: "Job completed successfully!",
        JobStatus.FAILED: f"Job failed: {job.error_message or 'Unknown error'}"
    }

    return status_messages.get(job.status, "Processing...")

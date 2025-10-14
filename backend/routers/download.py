"""
Download Endpoints
Serve converted audio files and model artifacts
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
import os
from pathlib import Path

from backend.database import get_db
from backend.models import Job, JobStatus

router = APIRouter(prefix="/api/download", tags=["download"])


@router.get("/audio/{job_id}")
async def download_audio(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Download converted or processed audio file

    - Returns audio file for completed jobs
    - Supports WAV, MP3, and other formats
    """

    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job.status}"
        )

    if not job.output_audio_path:
        raise HTTPException(status_code=404, detail="Output file not available")

    output_path = Path(job.output_audio_path)
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found on disk")

    # Determine media type based on extension
    media_types = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.m4a': 'audio/mp4',
        '.flac': 'audio/flac',
        '.ogg': 'audio/ogg'
    }
    media_type = media_types.get(output_path.suffix.lower(), 'audio/wav')

    # Generate download filename
    download_filename = f"{job.type}_{job_id}{output_path.suffix}"

    return FileResponse(
        path=str(output_path),
        media_type=media_type,
        filename=download_filename,
        headers={
            "Content-Disposition": f"attachment; filename={download_filename}"
        }
    )


@router.get("/audio/{job_id}/stream")
async def stream_audio(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Stream audio file for in-browser playback

    - Similar to download but with inline content disposition
    - Allows browser to play audio directly
    """

    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed. Current status: {job.status}"
        )

    if not job.output_audio_path:
        raise HTTPException(status_code=404, detail="Output file not available")

    output_path = Path(job.output_audio_path)
    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found on disk")

    # Determine media type
    media_types = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.m4a': 'audio/mp4',
        '.flac': 'audio/flac',
        '.ogg': 'audio/ogg'
    }
    media_type = media_types.get(output_path.suffix.lower(), 'audio/wav')

    return FileResponse(
        path=str(output_path),
        media_type=media_type,
        headers={
            "Content-Disposition": "inline",
            "Accept-Ranges": "bytes"
        }
    )


@router.get("/input/{job_id}")
async def download_input_audio(
    job_id: str,
    db: Session = Depends(get_db)
):
    """
    Download original input audio file

    - Useful for comparing input vs output
    - Available for any job with uploaded file
    """

    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if not job.input_audio_path:
        raise HTTPException(status_code=404, detail="Input file not available")

    input_path = Path(job.input_audio_path)
    if not input_path.exists():
        raise HTTPException(status_code=404, detail="Input file not found on disk")

    # Determine media type
    media_types = {
        '.wav': 'audio/wav',
        '.mp3': 'audio/mpeg',
        '.m4a': 'audio/mp4',
        '.flac': 'audio/flac',
        '.ogg': 'audio/ogg'
    }
    media_type = media_types.get(input_path.suffix.lower(), 'audio/wav')

    download_filename = f"input_{job_id}{input_path.suffix}"

    return FileResponse(
        path=str(input_path),
        media_type=media_type,
        filename=download_filename,
        headers={
            "Content-Disposition": f"attachment; filename={download_filename}"
        }
    )

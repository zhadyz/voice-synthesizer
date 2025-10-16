"""
File Upload Endpoints
Handles training audio and target audio uploads
"""

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form, Request
from sqlalchemy.orm import Session
import uuid
import os
import re
from pathlib import Path
from datetime import datetime
import librosa
import logging

from backend.database import get_db
from backend.models import Job, JobStatus
from backend.schemas import UploadResponse
from backend.auth import verify_api_key

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/upload", tags=["upload"])

# Upload directory
UPLOAD_DIR = Path(__file__).parent.parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# File validation constants
ALLOWED_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.aac'}
MAX_FILE_SIZE_MB = 100
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


def sanitize_filename(filename: str) -> tuple[str, str]:
    """
    Sanitize filename to prevent path traversal attacks

    Returns:
        tuple: (sanitized_filename, file_extension)
    """
    # Get basename to strip any directory components
    filename = os.path.basename(filename)

    # Remove dangerous characters and path traversal patterns
    filename = re.sub(r'[^\w\s.-]', '_', filename)
    filename = filename.replace('..', '_')

    # Extract extension
    file_ext = Path(filename).suffix.lower()

    return filename, file_ext


def validate_audio_file(filename: str, file_size: int) -> str:
    """
    Validate audio file extension and size

    Returns:
        str: Sanitized file extension
    """
    _, ext = sanitize_filename(filename)

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file format. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    if file_size > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
        )

    return ext


@router.post("/training-audio", response_model=UploadResponse)
async def upload_training_audio(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Form(default="default_user"),
    db: Session = Depends(get_db),
    _auth: str = Depends(verify_api_key)
):
    """
    Upload user's voice recording for model training

    - Accepts audio files up to 100MB
    - Supported formats: MP3, WAV, M4A, FLAC, OGG, AAC
    - Creates a training job in the database
    """

    # SECURITY: Check Content-Length header BEFORE reading file
    content_length = request.headers.get('content-length')
    if content_length and int(content_length) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
        )

    # Validate file extension
    file_ext = validate_audio_file(file.filename, 0)  # Size checked via header

    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # SECURITY: Use UUID-based filename to prevent path traversal
    safe_filename = f"{job_id}{file_ext}"
    file_path = UPLOAD_DIR / safe_filename

    # SECURITY: Stream file to disk with size validation
    bytes_written = 0
    try:
        with open(file_path, "wb") as f:
            while chunk := await file.read(8192):  # 8KB chunks
                bytes_written += len(chunk)
                if bytes_written > MAX_FILE_SIZE_BYTES:
                    f.close()
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File write failed: {e}")
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to save file. Please try again."
        )

    # Create job in database
    job = Job(
        id=job_id,
        user_id=user_id,
        type="training",
        status=JobStatus.PENDING,
        input_audio_path=str(file_path),
        created_at=datetime.utcnow()
    )

    try:
        db.add(job)
        db.commit()
    except Exception as e:
        # Cleanup file if database fails
        logger.error(f"Database error: {e}")
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Database error. Please try again."
        )

    size_mb = bytes_written / (1024 * 1024)

    return UploadResponse(
        job_id=job_id,
        filename=safe_filename,
        size_mb=round(size_mb, 2),
        status="uploaded",
        message=f"Training audio uploaded successfully. Job ID: {job_id}"
    )


@router.post("/target-audio", response_model=UploadResponse)
async def upload_target_audio(
    request: Request,
    file: UploadFile = File(...),
    user_id: str = Form(default="default_user"),
    db: Session = Depends(get_db),
    _auth: str = Depends(verify_api_key)
):
    """
    Upload audio to convert to user's voice

    - Accepts audio files up to 100MB
    - Supported formats: MP3, WAV, M4A, FLAC, OGG, AAC
    - Creates a conversion job in the database
    """

    # SECURITY: Check Content-Length header BEFORE reading file
    content_length = request.headers.get('content-length')
    if content_length and int(content_length) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
        )

    # Validate file extension
    file_ext = validate_audio_file(file.filename, 0)  # Size checked via header

    # Generate unique job ID
    job_id = str(uuid.uuid4())

    # SECURITY: Use UUID-based filename to prevent path traversal
    safe_filename = f"{job_id}{file_ext}"
    file_path = UPLOAD_DIR / safe_filename

    # SECURITY: Stream file to disk with size validation
    bytes_written = 0
    try:
        with open(file_path, "wb") as f:
            while chunk := await file.read(8192):  # 8KB chunks
                bytes_written += len(chunk)
                if bytes_written > MAX_FILE_SIZE_BYTES:
                    f.close()
                    file_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File too large. Maximum size: {MAX_FILE_SIZE_MB}MB"
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File write failed: {e}")
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to save file. Please try again."
        )

    # Create job in database
    job = Job(
        id=job_id,
        user_id=user_id,
        type="conversion",
        status=JobStatus.PENDING,
        input_audio_path=str(file_path),
        created_at=datetime.utcnow()
    )

    try:
        db.add(job)
        db.commit()
    except Exception as e:
        # Cleanup file if database fails
        logger.error(f"Database error: {e}")
        file_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail="Database error. Please try again."
        )

    size_mb = bytes_written / (1024 * 1024)

    return UploadResponse(
        job_id=job_id,
        filename=safe_filename,
        size_mb=round(size_mb, 2),
        status="uploaded",
        message=f"Target audio uploaded successfully. Job ID: {job_id}"
    )


@router.get("/validate/{job_id}")
async def validate_uploaded_audio(
    job_id: str,
    db: Session = Depends(get_db),
    _auth: str = Depends(verify_api_key)
):
    """
    Validate uploaded audio file and get metadata

    Returns audio duration, sample rate, and other metadata
    """

    job = db.query(Job).filter(Job.id == job_id).first()
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    audio_path = job.input_audio_path
    if not os.path.exists(audio_path):
        raise HTTPException(status_code=404, detail="Audio file not found")

    try:
        # Load audio metadata
        y, sr = librosa.load(audio_path, sr=None, duration=1.0)  # Load first 1 second
        duration = librosa.get_duration(path=audio_path)

        return {
            "job_id": job_id,
            "valid": True,
            "sample_rate": sr,
            "duration_seconds": round(duration, 2),
            "channels": 1 if y.ndim == 1 else y.shape[0],
            "file_size_mb": round(os.path.getsize(audio_path) / (1024 * 1024), 2)
        }
    except Exception as e:
        return {
            "job_id": job_id,
            "valid": False,
            "error": str(e)
        }

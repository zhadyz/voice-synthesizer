"""
Voice Model Management Endpoints
List, retrieve, and delete trained voice models
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional
from pathlib import Path

from backend.database import get_db
from backend.models import VoiceModel, Job
from backend.schemas import VoiceModelResponse, VoiceModelListResponse

router = APIRouter(prefix="/api/models", tags=["models"])


@router.get("/list", response_model=VoiceModelListResponse)
async def list_models(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    limit: int = Query(50, ge=1, le=100, description="Maximum number of models to return"),
    offset: int = Query(0, ge=0, description="Number of models to skip"),
    db: Session = Depends(get_db)
):
    """
    List trained voice models

    - Filter by user_id
    - Supports pagination
    - Returns models sorted by creation date (newest first)
    """

    query = db.query(VoiceModel)

    # Apply filter
    if user_id:
        query = query.filter(VoiceModel.user_id == user_id)

    # Get total count
    total = query.count()

    # Apply pagination and sorting
    models = query.order_by(VoiceModel.created_at.desc()).offset(offset).limit(limit).all()

    return VoiceModelListResponse(
        models=[VoiceModelResponse.model_validate(model) for model in models],
        total=total
    )


@router.get("/{model_id}", response_model=VoiceModelResponse)
async def get_model(
    model_id: str,
    db: Session = Depends(get_db)
):
    """
    Get voice model details

    Returns model metadata and quality metrics
    """

    model = db.query(VoiceModel).filter(VoiceModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    return VoiceModelResponse.model_validate(model)


@router.delete("/{model_id}")
async def delete_model(
    model_id: str,
    delete_files: bool = Query(True, description="Also delete model files from disk"),
    db: Session = Depends(get_db)
):
    """
    Delete voice model

    - Removes model from database
    - Optionally deletes model files from disk
    """

    model = db.query(VoiceModel).filter(VoiceModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Delete files if requested
    files_deleted = []
    if delete_files:
        model_path = Path(model.model_path)
        if model_path.exists():
            if model_path.is_dir():
                # Delete model directory
                import shutil
                shutil.rmtree(model_path)
                files_deleted.append(str(model_path))
            else:
                model_path.unlink()
                files_deleted.append(str(model_path))

        # Delete training audio if exists
        if model.training_audio_path:
            training_path = Path(model.training_audio_path)
            if training_path.exists():
                training_path.unlink()
                files_deleted.append(str(training_path))

    # Delete from database
    db.delete(model)
    db.commit()

    return {
        "message": "Model deleted successfully",
        "model_id": model_id,
        "files_deleted": files_deleted
    }


@router.get("/{model_id}/stats")
async def get_model_stats(
    model_id: str,
    db: Session = Depends(get_db)
):
    """
    Get detailed statistics for a voice model

    - Training duration
    - Number of conversions performed
    - Average quality metrics
    """

    model = db.query(VoiceModel).filter(VoiceModel.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")

    # Count conversions using this model
    conversion_count = db.query(Job).filter(
        Job.model_id == model_id,
        Job.type == "conversion"
    ).count()

    # Get completed conversions
    completed_conversions = db.query(Job).filter(
        Job.model_id == model_id,
        Job.type == "conversion",
        Job.status == "completed"
    ).count()

    # Calculate average quality from conversions
    conversions = db.query(Job).filter(
        Job.model_id == model_id,
        Job.type == "conversion",
        Job.quality_snr.isnot(None)
    ).all()

    avg_snr = None
    if conversions:
        snr_values = [c.quality_snr for c in conversions if c.quality_snr is not None]
        if snr_values:
            avg_snr = sum(snr_values) / len(snr_values)

    # Get model size
    model_size_mb = None
    model_path = Path(model.model_path)
    if model_path.exists():
        if model_path.is_dir():
            model_size_mb = sum(
                f.stat().st_size for f in model_path.rglob('*') if f.is_file()
            ) / (1024 * 1024)
        else:
            model_size_mb = model_path.stat().st_size / (1024 * 1024)

    return {
        "model_id": model_id,
        "model_name": model.model_name,
        "created_at": model.created_at,
        "training_duration_seconds": model.training_duration_seconds,
        "training_quality_snr": model.quality_snr,
        "conversions": {
            "total": conversion_count,
            "completed": completed_conversions,
            "failed": conversion_count - completed_conversions
        },
        "avg_conversion_quality_snr": round(avg_snr, 2) if avg_snr else None,
        "model_size_mb": round(model_size_mb, 2) if model_size_mb else None
    }

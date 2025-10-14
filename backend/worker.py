"""
ARQ Background Worker
Async job queue for long-running training and conversion tasks
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from arq import create_pool, cron
from arq.connections import RedisSettings
import logging

from backend.database import SessionLocal
from backend.models import Job, JobStatus, VoiceModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def update_job_status(
    job_id: str,
    status: JobStatus,
    progress: float = None,
    error_message: str = None,
    output_path: str = None,
    quality_snr: float = None
):
    """Update job status in database"""
    db = SessionLocal()
    try:
        job = db.query(Job).filter(Job.id == job_id).first()
        if job:
            job.status = status
            if progress is not None:
                job.progress = progress
            if error_message:
                job.error_message = error_message
            if output_path:
                job.output_audio_path = output_path
            if quality_snr is not None:
                job.quality_snr = quality_snr
            if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
                job.completed_at = datetime.utcnow()
            job.updated_at = datetime.utcnow()
            db.commit()
            logger.info(f"Updated job {job_id}: {status.value} ({progress}%)")
    except Exception as e:
        logger.error(f"Failed to update job {job_id}: {e}")
        db.rollback()
    finally:
        db.close()


async def preprocess_audio(ctx: Dict[str, Any], job_id: str, audio_path: str, user_id: str) -> Dict:
    """
    Background task: Preprocess training audio
    - Noise reduction
    - Volume normalization
    - Quality validation
    """
    logger.info(f"Starting preprocessing for job {job_id}")
    update_job_status(job_id, JobStatus.PREPROCESSING, progress=0.0)

    try:
        # Import pipeline (lazy import to avoid startup issues)
        from src.pipeline.voice_cloning_pipeline import VoiceCloningPipeline

        # Initialize pipeline
        pipeline = VoiceCloningPipeline()

        # Update progress
        update_job_status(job_id, JobStatus.PREPROCESSING, progress=0.3)

        # Run preprocessing
        result = pipeline.preprocess_training_audio(audio_path, user_id)

        # Update with results
        update_job_status(
            job_id,
            JobStatus.PENDING,  # Ready for training
            progress=1.0,
            quality_snr=result.get("quality_snr")
        )

        logger.info(f"Preprocessing completed for job {job_id}")
        return {"status": "success", "result": result}

    except Exception as e:
        logger.error(f"Preprocessing failed for job {job_id}: {e}")
        update_job_status(
            job_id,
            JobStatus.FAILED,
            error_message=f"Preprocessing failed: {str(e)}"
        )
        return {"status": "failed", "error": str(e)}


async def train_voice_model(
    ctx: Dict[str, Any],
    job_id: str,
    clean_audio_path: str,
    model_name: str,
    user_id: str
) -> Dict:
    """
    Background task: Train RVC model
    - Takes 30-40 minutes
    - GPU intensive
    """
    logger.info(f"Starting training for job {job_id}")
    update_job_status(job_id, JobStatus.TRAINING, progress=0.0)

    try:
        # Import pipeline
        from src.pipeline.voice_cloning_pipeline import VoiceCloningPipeline

        pipeline = VoiceCloningPipeline()

        # Simulate progress updates (RVC training doesn't have built-in progress)
        async def progress_updater():
            """Update progress every minute"""
            for i in range(1, 40):
                await asyncio.sleep(60)  # Update every minute
                progress = min(i / 40, 0.95)  # Cap at 95% until complete
                update_job_status(job_id, JobStatus.TRAINING, progress=progress)

        # Start progress updater
        progress_task = asyncio.create_task(progress_updater())

        try:
            # Run training (blocking, runs in executor)
            loop = asyncio.get_event_loop()
            model_path = await loop.run_in_executor(
                None,
                pipeline.train_voice_model,
                clean_audio_path,
                model_name
            )
        finally:
            # Cancel progress updater
            progress_task.cancel()
            try:
                await progress_task
            except asyncio.CancelledError:
                pass

        # Create VoiceModel record
        db = SessionLocal()
        try:
            voice_model = VoiceModel(
                id=f"model_{job_id}",
                user_id=user_id,
                model_name=model_name,
                model_path=model_path,
                training_audio_path=clean_audio_path,
                training_job_id=job_id,
                created_at=datetime.utcnow()
            )
            db.add(voice_model)
            db.commit()
        finally:
            db.close()

        # Mark job as completed
        update_job_status(
            job_id,
            JobStatus.COMPLETED,
            progress=1.0,
            output_path=model_path
        )

        logger.info(f"Training completed for job {job_id}")
        return {"status": "success", "model_path": model_path}

    except Exception as e:
        logger.error(f"Training failed for job {job_id}: {e}")
        update_job_status(
            job_id,
            JobStatus.FAILED,
            error_message=f"Training failed: {str(e)}"
        )
        return {"status": "failed", "error": str(e)}


async def convert_voice(
    ctx: Dict[str, Any],
    job_id: str,
    model_path: str,
    target_audio: str,
    output_name: str
) -> Dict:
    """
    Background task: Voice conversion
    - Uses trained RVC model
    - Takes 1-5 minutes depending on audio length
    """
    logger.info(f"Starting conversion for job {job_id}")
    update_job_status(job_id, JobStatus.CONVERTING, progress=0.0)

    try:
        # Import pipeline
        from src.pipeline.voice_cloning_pipeline import VoiceCloningPipeline

        pipeline = VoiceCloningPipeline()

        # Update progress
        update_job_status(job_id, JobStatus.CONVERTING, progress=0.2)

        # Run conversion (blocking)
        loop = asyncio.get_event_loop()
        output_path = await loop.run_in_executor(
            None,
            pipeline.convert_audio,
            model_path,
            target_audio,
            output_name
        )

        # Update progress
        update_job_status(job_id, JobStatus.CONVERTING, progress=0.8)

        # Mark as completed
        update_job_status(
            job_id,
            JobStatus.COMPLETED,
            progress=1.0,
            output_path=output_path
        )

        logger.info(f"Conversion completed for job {job_id}")
        return {"status": "success", "output_path": output_path}

    except Exception as e:
        logger.error(f"Conversion failed for job {job_id}: {e}")
        update_job_status(
            job_id,
            JobStatus.FAILED,
            error_message=f"Conversion failed: {str(e)}"
        )
        return {"status": "failed", "error": str(e)}


async def cleanup_old_jobs(ctx: Dict[str, Any]) -> Dict:
    """
    Periodic task: Clean up old completed jobs
    - Runs daily
    - Removes jobs older than 30 days
    """
    logger.info("Running cleanup task")

    try:
        from datetime import timedelta

        db = SessionLocal()
        try:
            # Find old completed/failed jobs
            cutoff_date = datetime.utcnow() - timedelta(days=30)

            old_jobs = db.query(Job).filter(
                Job.status.in_([JobStatus.COMPLETED, JobStatus.FAILED]),
                Job.completed_at < cutoff_date
            ).all()

            deleted_count = 0
            for job in old_jobs:
                # Delete associated files
                if job.input_audio_path:
                    try:
                        Path(job.input_audio_path).unlink(missing_ok=True)
                    except:
                        pass

                if job.output_audio_path:
                    try:
                        Path(job.output_audio_path).unlink(missing_ok=True)
                    except:
                        pass

                # Delete job record
                db.delete(job)
                deleted_count += 1

            db.commit()
            logger.info(f"Cleaned up {deleted_count} old jobs")
            return {"status": "success", "deleted": deleted_count}

        finally:
            db.close()

    except Exception as e:
        logger.error(f"Cleanup failed: {e}")
        return {"status": "failed", "error": str(e)}


# ARQ Worker Settings
class WorkerSettings:
    """ARQ worker configuration"""

    # Redis connection
    redis_settings = RedisSettings(
        host='localhost',
        port=6379,
        database=0
    )

    # Background task functions
    functions = [
        preprocess_audio,
        train_voice_model,
        convert_voice,
        cleanup_old_jobs
    ]

    # Worker configuration
    max_jobs = 1  # Only 1 GPU training job at a time
    job_timeout = 3600  # 1 hour timeout for training jobs
    keep_result = 3600  # Keep results for 1 hour

    # Periodic tasks (cron jobs)
    cron_jobs = [
        cron(cleanup_old_jobs, hour=2, minute=0)  # Run daily at 2 AM
    ]

    # Worker name
    worker_name = "voice_cloning_worker"


# Entry point for running worker directly
if __name__ == "__main__":
    logger.info("Starting ARQ worker...")
    from arq import run_worker

    run_worker(WorkerSettings)

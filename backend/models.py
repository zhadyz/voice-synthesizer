"""
Database Models for Voice Cloning Backend
SQLAlchemy ORM models for jobs and voice models
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Enum, Text
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import enum

Base = declarative_base()


class JobStatus(str, enum.Enum):
    """Job status enumeration"""
    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    TRAINING = "training"
    CONVERTING = "converting"
    COMPLETED = "completed"
    FAILED = "failed"


class Job(Base):
    """Job tracking for training and conversion tasks"""
    __tablename__ = "jobs"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    type = Column(String, nullable=False)  # 'training' or 'conversion'
    status = Column(Enum(JobStatus), default=JobStatus.PENDING, nullable=False)
    progress = Column(Float, default=0.0)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    # File paths
    input_audio_path = Column(String, nullable=False)
    output_audio_path = Column(String, nullable=True)

    # Quality metrics
    quality_snr = Column(Float, nullable=True)
    quality_pesq = Column(Float, nullable=True)

    # Model reference (for conversion jobs)
    model_id = Column(String, nullable=True)

    # Error tracking
    error_message = Column(Text, nullable=True)

    def __repr__(self):
        return f"<Job(id={self.id}, type={self.type}, status={self.status})>"


class VoiceModel(Base):
    """Trained voice models"""
    __tablename__ = "voice_models"

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    model_name = Column(String, nullable=False)
    model_path = Column(String, nullable=False)

    # Training data
    training_audio_path = Column(String, nullable=False)
    training_duration_seconds = Column(Float, nullable=True)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Quality metrics
    quality_snr = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)

    # Training job reference
    training_job_id = Column(String, nullable=True)

    def __repr__(self):
        return f"<VoiceModel(id={self.id}, name={self.model_name})>"

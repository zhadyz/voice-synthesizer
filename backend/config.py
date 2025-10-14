"""
Backend Configuration
Centralized configuration management
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List


class Settings(BaseSettings):
    """Application settings"""

    # Application
    APP_NAME: str = "Voice Cloning API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # API Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True

    # Database
    DATABASE_URL: str = "sqlite:///./data/voice_cloning.db"
    DATABASE_ECHO: bool = False

    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: str = ""

    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # File Upload
    MAX_FILE_SIZE_MB: int = 100
    ALLOWED_EXTENSIONS: List[str] = [".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac"]

    # Directories
    UPLOAD_DIR: Path = Path("uploads")
    OUTPUT_DIR: Path = Path("outputs/converted")
    MODEL_DIR: Path = Path("models/trained")
    TEMP_DIR: Path = Path("temp")

    # Worker
    MAX_CONCURRENT_JOBS: int = 1  # Only 1 GPU job at a time
    JOB_TIMEOUT: int = 3600  # 1 hour
    KEEP_RESULT: int = 3600  # Keep results for 1 hour

    # Cleanup
    CLEANUP_ENABLED: bool = True
    CLEANUP_DAYS: int = 30  # Delete jobs older than 30 days

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/backend.log"

    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()


def init_directories():
    """Create required directories if they don't exist"""
    settings.UPLOAD_DIR.mkdir(exist_ok=True, parents=True)
    settings.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    settings.MODEL_DIR.mkdir(exist_ok=True, parents=True)
    settings.TEMP_DIR.mkdir(exist_ok=True, parents=True)

    logs_dir = Path(settings.LOG_FILE).parent
    logs_dir.mkdir(exist_ok=True, parents=True)


# Initialize directories on import
init_directories()

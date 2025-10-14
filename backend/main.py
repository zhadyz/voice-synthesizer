"""
Voice Cloning API - FastAPI Backend
Main application with routing and middleware
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from backend.database import init_db, engine
from backend.routers import upload, jobs, stream, download, models
from backend.schemas import HealthResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events
    """
    # Startup
    print("Starting Voice Cloning API...")
    init_db()
    print("Database initialized successfully")

    yield

    # Shutdown
    print("Shutting down Voice Cloning API...")
    engine.dispose()


# Create FastAPI application
app = FastAPI(
    title="Voice Cloning API",
    description="Production-ready FastAPI backend for voice cloning with RVC",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # React dev server
        "http://localhost:5173",  # Vite dev server
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)


# Exception handler for general errors
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url)
        }
    )


# Include routers
app.include_router(upload.router)
app.include_router(jobs.router)
app.include_router(stream.router)
app.include_router(download.router)
app.include_router(models.router)


# Root endpoint
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Voice Cloning API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint

    Returns:
    - API status
    - GPU availability
    - Database connection status
    - Redis connection status (TODO)
    """

    # Check GPU availability
    gpu_available = torch.cuda.is_available()
    gpu_name = None
    if gpu_available:
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except:
            pass

    # Check database connection
    database_connected = True
    try:
        from backend.database import SessionLocal
        db = SessionLocal()
        db.execute("SELECT 1")
        db.close()
    except Exception as e:
        database_connected = False
        print(f"Database health check failed: {e}")

    # Check Redis connection (TODO when ARQ is implemented)
    redis_connected = False  # Will implement with ARQ

    return HealthResponse(
        status="ok" if database_connected else "degraded",
        version="1.0.0",
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        redis_connected=redis_connected,
        database_connected=database_connected
    )


# API info endpoint
@app.get("/api/info")
async def api_info():
    """Get API information and available endpoints"""
    return {
        "name": "Voice Cloning API",
        "version": "1.0.0",
        "endpoints": {
            "upload": {
                "training_audio": "POST /api/upload/training-audio",
                "target_audio": "POST /api/upload/target-audio",
                "validate": "GET /api/upload/validate/{job_id}"
            },
            "jobs": {
                "start_training": "POST /api/jobs/train",
                "start_conversion": "POST /api/jobs/convert",
                "get_status": "GET /api/jobs/status/{job_id}",
                "list_jobs": "GET /api/jobs/list",
                "cancel": "DELETE /api/jobs/{job_id}",
                "retry": "POST /api/jobs/{job_id}/retry"
            },
            "stream": {
                "progress": "GET /api/stream/progress/{job_id}",
                "multi_progress": "GET /api/stream/multi-progress?user_id={user_id}"
            },
            "download": {
                "audio": "GET /api/download/audio/{job_id}",
                "stream_audio": "GET /api/download/audio/{job_id}/stream",
                "input": "GET /api/download/input/{job_id}"
            },
            "models": {
                "list": "GET /api/models/list",
                "get": "GET /api/models/{model_id}",
                "delete": "DELETE /api/models/{model_id}",
                "stats": "GET /api/models/{model_id}/stats"
            }
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

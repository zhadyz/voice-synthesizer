#!/bin/bash

# Voice Cloning Backend Startup Script
# Starts Redis, ARQ worker, and FastAPI server

set -e

echo "============================================"
echo "Voice Cloning Backend - Startup"
echo "============================================"

# Check if Redis is installed
if ! command -v redis-server &> /dev/null; then
    echo "ERROR: Redis is not installed"
    echo "Install Redis:"
    echo "  Ubuntu/Debian: sudo apt-get install redis-server"
    echo "  macOS: brew install redis"
    echo "  Windows: Download from https://github.com/microsoftarchive/redis/releases"
    exit 1
fi

# Check if Python environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "WARNING: No virtual environment detected"
    echo "Activating venv..."
    if [ -d "venv" ]; then
        source venv/bin/activate
    elif [ -d ".venv" ]; then
        source .venv/bin/activate
    else
        echo "ERROR: No virtual environment found"
        echo "Create one with: python -m venv venv"
        exit 1
    fi
fi

# Start Redis server
echo ""
echo "[1/3] Starting Redis server..."
if pgrep redis-server > /dev/null; then
    echo "Redis already running"
else
    redis-server --daemonize yes --port 6379
    sleep 2
    echo "Redis started on port 6379"
fi

# Start ARQ worker in background
echo ""
echo "[2/3] Starting ARQ worker..."
pkill -f "arq backend.worker" || true  # Kill existing worker
python -m arq backend.worker.WorkerSettings > logs/worker.log 2>&1 &
WORKER_PID=$!
echo "ARQ worker started (PID: $WORKER_PID)"
echo "Worker logs: logs/worker.log"

# Give worker time to start
sleep 2

# Start FastAPI server
echo ""
echo "[3/3] Starting FastAPI server..."
echo "Server will be available at: http://localhost:8000"
echo "API docs: http://localhost:8000/docs"
echo ""
echo "Press CTRL+C to stop"
echo "============================================"
echo ""

# Run FastAPI with uvicorn
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Cleanup on exit
trap "echo 'Stopping services...'; kill $WORKER_PID 2>/dev/null; redis-cli shutdown 2>/dev/null; exit" INT TERM

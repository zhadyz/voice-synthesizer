#!/bin/bash
# ARQ Worker Startup Script for Linux/Mac
# Starts the voice cloning background worker

set -e

echo "========================================"
echo "ARQ Worker Startup Script (Linux/Mac)"
echo "========================================"
echo ""

# Change to script directory
cd "$(dirname "$0")"

# Check if virtual environment exists
if [ ! -f "venv/bin/activate" ]; then
    echo "ERROR: Virtual environment not found at: venv/bin/activate"
    echo "Please run: python3 -m venv venv"
    echo "Then: pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if Redis is accessible
echo ""
echo "Checking Redis connection..."
if ! python -c "import redis; r = redis.Redis(host='localhost', port=6379); r.ping(); print('Redis connection OK')" 2>/dev/null; then
    echo "ERROR: Cannot connect to Redis on localhost:6379"
    echo "Please start Redis first using: ./start_redis.sh"
    exit 1
fi

# Start ARQ worker
echo ""
echo "========================================"
echo "Starting ARQ Worker..."
echo "========================================"
echo ""
echo "Worker will process:"
echo "  - Audio preprocessing jobs"
echo "  - Voice model training jobs"
echo "  - Voice conversion jobs"
echo "  - Automatic cleanup (daily at 2 AM)"
echo ""
echo "Press Ctrl+C to stop the worker"
echo "========================================"
echo ""

python backend/worker.py

# If worker exits
echo ""
echo "Worker stopped."

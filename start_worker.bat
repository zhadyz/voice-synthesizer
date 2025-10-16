@echo off
REM ARQ Worker Startup Script for Windows
REM Starts the voice cloning background worker

echo ========================================
echo ARQ Worker Startup Script (Windows)
echo ========================================
echo.

REM Change to project directory
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found at: venv\Scripts\activate.bat
    echo Please run: python -m venv venv
    echo Then: pip install -r requirements.txt
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Check if Redis is accessible
echo.
echo Checking Redis connection...
python -c "import redis; r = redis.Redis(host='localhost', port=6379); r.ping(); print('Redis connection OK')" 2>nul
if %errorlevel% neq 0 (
    echo ERROR: Cannot connect to Redis on localhost:6379
    echo Please start Redis first using: start_redis.bat
    pause
    exit /b 1
)

REM Start ARQ worker
echo.
echo ========================================
echo Starting ARQ Worker...
echo ========================================
echo.
echo Worker will process:
echo  - Audio preprocessing jobs
echo  - Voice model training jobs
echo  - Voice conversion jobs
echo  - Automatic cleanup (daily at 2 AM)
echo.
echo Press Ctrl+C to stop the worker
echo ========================================
echo.

python backend\worker.py

REM If worker exits
echo.
echo Worker stopped.
pause

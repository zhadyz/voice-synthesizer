@echo off
REM Voice Cloning Backend Startup Script (Windows)
REM Starts Redis, ARQ worker, and FastAPI server

echo ============================================
echo Voice Cloning Backend - Startup
echo ============================================

REM Check if virtual environment is activated
if "%VIRTUAL_ENV%"=="" (
    echo WARNING: No virtual environment detected
    echo Activating venv...
    if exist venv\Scripts\activate.bat (
        call venv\Scripts\activate.bat
    ) else if exist .venv\Scripts\activate.bat (
        call .venv\Scripts\activate.bat
    ) else (
        echo ERROR: No virtual environment found
        echo Create one with: python -m venv venv
        exit /b 1
    )
)

REM Create logs directory
if not exist logs mkdir logs

REM Check for Redis
echo.
echo [1/3] Checking Redis...
where redis-server >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Redis is not installed or not in PATH
    echo Download from: https://github.com/microsoftarchive/redis/releases
    echo Or use Docker: docker run -d -p 6379:6379 redis
    echo.
    echo Continue without Redis? ARQ worker will fail. [Y/N]
    set /p continue=
    if /i not "%continue%"=="Y" exit /b 1
) else (
    REM Start Redis if not running
    tasklist /FI "IMAGENAME eq redis-server.exe" 2>NUL | find /I /N "redis-server.exe">NUL
    if "%ERRORLEVEL%"=="0" (
        echo Redis already running
    ) else (
        start /B redis-server.exe
        timeout /t 2 /nobreak >nul
        echo Redis started on port 6379
    )
)

REM Start ARQ worker
echo.
echo [2/3] Starting ARQ worker...
taskkill /F /FI "WINDOWTITLE eq ARQ Worker*" >nul 2>&1
start "ARQ Worker" /MIN python -m arq backend.worker.WorkerSettings
timeout /t 2 /nobreak >nul
echo ARQ worker started
echo Worker logs: logs\worker.log

REM Start FastAPI server
echo.
echo [3/3] Starting FastAPI server...
echo Server will be available at: http://localhost:8000
echo API docs: http://localhost:8000/docs
echo.
echo Press CTRL+C to stop
echo ============================================
echo.

REM Run FastAPI with uvicorn
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

REM Note: Cleanup on Windows is handled manually
REM Use run_backend_stop.bat to stop all services

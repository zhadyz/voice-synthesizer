@echo off
REM Redis Startup Script for Windows
REM Starts Redis server via Docker (recommended method)

echo ========================================
echo Redis Server Startup Script (Windows)
echo ========================================
echo.

REM Check if Docker is available
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not installed or not in PATH
    echo Please install Docker Desktop from: https://www.docker.com/products/docker-desktop
    pause
    exit /b 1
)

REM Check if Redis container exists
docker ps -a --filter "name=voice-cloning-redis" --format "{{.Names}}" | findstr "voice-cloning-redis" >nul
if %errorlevel% equ 0 (
    echo Redis container already exists. Starting...
    docker start voice-cloning-redis
    if %errorlevel% neq 0 (
        echo ERROR: Failed to start Redis container
        pause
        exit /b 1
    )
) else (
    echo Creating new Redis container...
    docker run -d ^
        --name voice-cloning-redis ^
        -p 6379:6379 ^
        --restart unless-stopped ^
        redis:7-alpine redis-server --appendonly yes
    if %errorlevel% neq 0 (
        echo ERROR: Failed to create Redis container
        pause
        exit /b 1
    )
)

echo.
echo Verifying Redis connection...
timeout /t 2 /nobreak >nul
docker exec voice-cloning-redis redis-cli ping
if %errorlevel% neq 0 (
    echo ERROR: Redis is not responding
    pause
    exit /b 1
)

echo.
echo ========================================
echo SUCCESS: Redis is running on localhost:6379
echo ========================================
echo.
echo To stop Redis:  docker stop voice-cloning-redis
echo To view logs:   docker logs -f voice-cloning-redis
echo To remove:      docker rm -f voice-cloning-redis
echo.
pause

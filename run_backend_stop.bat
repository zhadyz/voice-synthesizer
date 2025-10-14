@echo off
REM Stop all backend services

echo Stopping Voice Cloning Backend services...

REM Stop FastAPI server (uvicorn)
echo Stopping FastAPI server...
taskkill /F /FI "WINDOWTITLE eq *uvicorn*" >nul 2>&1
taskkill /F /IM python.exe /FI "COMMANDLINE eq *uvicorn*" >nul 2>&1

REM Stop ARQ worker
echo Stopping ARQ worker...
taskkill /F /FI "WINDOWTITLE eq ARQ Worker*" >nul 2>&1

REM Stop Redis
echo Stopping Redis...
taskkill /F /IM redis-server.exe >nul 2>&1

echo.
echo All services stopped.
pause

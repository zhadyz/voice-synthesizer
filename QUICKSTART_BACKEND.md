# Backend Quick Start Guide

Get the Voice Cloning API running in 5 minutes!

## Prerequisites Checklist

- [ ] Python 3.10+ installed
- [ ] Redis installed (or Docker)
- [ ] Virtual environment activated
- [ ] CUDA GPU (recommended, not required)

## Step 1: Install Redis

### Windows
```bash
# Option 1: Download installer
# https://github.com/microsoftarchive/redis/releases
# Download Redis-x64-3.2.100.msi

# Option 2: Use Docker
docker run -d -p 6379:6379 --name redis redis:latest
```

### macOS
```bash
brew install redis
brew services start redis
```

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install redis-server
sudo systemctl start redis
```

## Step 2: Install Python Dependencies

```bash
# Activate your virtual environment first!
# venv\Scripts\activate  (Windows)
# source venv/bin/activate  (Linux/Mac)

# Install backend dependencies
pip install -r requirements_backend.txt
```

## Step 3: Initialize Database

```bash
python -c "from backend.database import init_db; init_db()"
```

You should see: `Database initialized at: data/voice_cloning.db`

## Step 4: Start the Backend

### Easy Way (Recommended)

**Windows**:
```bash
run_backend.bat
```

**Linux/Mac**:
```bash
chmod +x run_backend.sh
./run_backend.sh
```

### Manual Way

Open 3 terminals:

**Terminal 1 - Redis**:
```bash
redis-server
```

**Terminal 2 - Worker**:
```bash
python -m arq backend.worker.WorkerSettings
```

**Terminal 3 - API**:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

## Step 5: Verify It Works

Open your browser and visit:

1. **API Root**: http://localhost:8000
2. **API Docs**: http://localhost:8000/docs
3. **Health Check**: http://localhost:8000/health

You should see JSON responses!

## Step 6: Test with cURL

### Upload Test Audio
```bash
curl -X POST "http://localhost:8000/api/upload/training-audio" \
  -F "file=@test_audio.mp3" \
  -F "user_id=test_user"
```

Response:
```json
{
  "job_id": "abc123...",
  "filename": "test_audio.mp3",
  "size_mb": 2.5,
  "status": "uploaded",
  "message": "Training audio uploaded successfully"
}
```

### Check Job Status
```bash
curl "http://localhost:8000/api/jobs/status/abc123..."
```

### List All Jobs
```bash
curl "http://localhost:8000/api/jobs/list?user_id=test_user"
```

## Step 7: Run Tests

```bash
# Make sure backend is running first!
pytest tests/test_backend.py -v
```

## Common Issues & Solutions

### ❌ Redis Connection Failed
```
Error: Connection refused (port 6379)
```
**Solution**: Start Redis server
```bash
redis-server
# or
docker start redis
```

### ❌ Port 8000 Already in Use
```
ERROR: [Errno 48] Address already in use
```
**Solution**: Kill existing process or use different port
```bash
# Find process using port 8000
lsof -i :8000  # Mac/Linux
netstat -ano | findstr :8000  # Windows

# Change port
uvicorn backend.main:app --port 8001
```

### ❌ Module Not Found
```
ModuleNotFoundError: No module named 'fastapi'
```
**Solution**: Install dependencies
```bash
pip install -r requirements_backend.txt
```

### ❌ Database Locked
```
sqlite3.OperationalError: database is locked
```
**Solution**: Close other connections or restart backend

## Next Steps

### Connect Frontend
Update frontend API base URL:
```javascript
const API_BASE_URL = "http://localhost:8000";
```

### Test Full Workflow

1. **Upload training audio**
2. **Start training job**
3. **Monitor progress via SSE**
4. **Upload target audio**
5. **Start conversion job**
6. **Download converted audio**

See `backend/README.md` for detailed API documentation.

### Deploy to Production

- Use Gunicorn instead of Uvicorn
- Switch to PostgreSQL database
- Set up Nginx reverse proxy
- Enable HTTPS with Let's Encrypt
- Configure Redis persistence
- Set up monitoring (Prometheus/Grafana)

## Stopping the Backend

**Windows**:
```bash
run_backend_stop.bat
```

**Linux/Mac**:
Press `CTRL+C` in the terminal running the backend

**Manual**:
```bash
# Kill all services
pkill -f uvicorn
pkill -f arq
redis-cli shutdown
```

## Need Help?

Check the logs:
```bash
# Worker logs
tail -f logs/worker.log

# API logs (in terminal output)
# or configure file logging in config.py
```

## Success! 🎉

Your backend is now running at http://localhost:8000

Try the interactive API docs: http://localhost:8000/docs

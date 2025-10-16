# Redis + ARQ Infrastructure Setup

## Overview

This document provides complete setup instructions for the Redis job queue and ARQ background worker infrastructure for the Voice Synthesizer project.

## Architecture

```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│   FastAPI   │──────▶│    Redis    │◀──────│ ARQ Worker  │
│   Backend   │       │  (Docker)   │       │  Process    │
└─────────────┘       └─────────────┘       └─────────────┘
     │                      │                      │
     │                      │                      │
     ├── Enqueue Jobs       ├── Job Queue         ├── Process Jobs
     ├── Check Status       ├── Job Results       ├── Update Status
     └── Get Results        └── Pub/Sub           └── Store Results
```

## Components

### 1. Redis Server (Docker)
- **Image**: `redis:7-alpine`
- **Port**: `6379` (localhost)
- **Persistence**: AOF (Append-Only File) enabled
- **Auto-restart**: Enabled (unless manually stopped)

### 2. ARQ Worker
- **Framework**: ARQ (Async Redis Queue)
- **Python Version**: 3.13.7
- **Max Concurrent Jobs**: 1 (GPU-intensive training jobs)
- **Job Timeout**: 3600 seconds (1 hour)
- **Cron Jobs**: Daily cleanup at 2 AM

### 3. Job Types
- **Preprocessing**: Audio cleanup and validation (~30s)
- **Training**: RVC model training (30-40 minutes)
- **Conversion**: Voice synthesis (1-5 minutes)
- **Cleanup**: Automated job/file cleanup (daily)

## Installation

### Prerequisites

1. **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
   - Download: https://www.docker.com/get-docker
   - Verify: `docker --version`

2. **Python 3.13+** with virtual environment
   - Verify: `python --version`

3. **Git** (for repository management)

### Step 1: Install Python Dependencies

```bash
# Activate virtual environment
# Windows (Git Bash):
source venv/Scripts/activate

# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install arq redis sqlalchemy greenlet hiredis PyJWT

# Or install all from requirements.txt:
pip install -r requirements.txt
```

### Step 2: Start Redis Server

#### Windows

```cmd
# Double-click or run from command prompt
start_redis.bat
```

Or manually:
```cmd
docker run -d --name voice-cloning-redis -p 6379:6379 --restart unless-stopped redis:7-alpine redis-server --appendonly yes
```

#### Linux/Mac

```bash
# Make script executable (first time only)
chmod +x start_redis.sh

# Run script
./start_redis.sh
```

Or manually:
```bash
docker run -d --name voice-cloning-redis -p 6379:6379 --restart unless-stopped redis:7-alpine redis-server --appendonly yes
```

### Step 3: Verify Redis

```bash
# Using Docker
docker exec voice-cloning-redis redis-cli ping
# Expected output: PONG

# Using Python
python -c "import redis; r = redis.Redis(host='localhost', port=6379); print(r.ping())"
# Expected output: True
```

### Step 4: Start ARQ Worker

#### Windows

```cmd
# Double-click or run from command prompt
start_worker.bat
```

Or manually:
```cmd
cd "C:\path\to\Speech Synthesis"
venv\Scripts\activate.bat
python backend\worker.py
```

#### Linux/Mac

```bash
# Make script executable (first time only)
chmod +x start_worker.sh

# Run script
./start_worker.sh
```

Or manually:
```bash
cd /path/to/Speech\ Synthesis
source venv/bin/activate
python backend/worker.py
```

## Verification

### Check Worker Status

When the worker starts successfully, you should see:

```
INFO:__main__:Starting ARQ worker...
INFO:arq.worker:Starting worker for 5 functions: preprocess_audio, train_voice_model, convert_voice, cleanup_old_jobs, cron:cleanup_old_jobs
INFO:arq.worker:redis_version=7.4.6 mem_usage=1.23M clients_connected=3 db_keys=0
```

### Test Job Submission (via Python)

```python
import asyncio
from arq import create_pool
from arq.connections import RedisSettings

async def test_job():
    redis = await create_pool(RedisSettings(host='localhost', port=6379))

    # Enqueue a test job
    job = await redis.enqueue_job('preprocess_audio',
                                   job_id='test_123',
                                   audio_path='/path/to/audio.wav',
                                   user_id='test_user')

    print(f"Job enqueued: {job.job_id}")

    # Check job status
    result = await job.result(timeout=60)
    print(f"Job result: {result}")

asyncio.run(test_job())
```

## Management Commands

### Redis Container Management

```bash
# Start Redis
docker start voice-cloning-redis

# Stop Redis
docker stop voice-cloning-redis

# Restart Redis
docker restart voice-cloning-redis

# View Redis logs
docker logs -f voice-cloning-redis

# Remove Redis container (WARNING: deletes data)
docker rm -f voice-cloning-redis

# Access Redis CLI
docker exec -it voice-cloning-redis redis-cli
```

### Redis CLI Commands

```bash
# Ping Redis
redis-cli ping

# Check queue length
redis-cli llen arq:queue

# View all keys
redis-cli keys '*'

# Monitor real-time commands
redis-cli monitor

# Get server info
redis-cli info

# Flush all data (WARNING: destructive)
redis-cli flushall
```

### Worker Management

```bash
# Run worker in foreground (for debugging)
python backend/worker.py

# Run worker in background (Linux/Mac)
nohup python backend/worker.py > logs/worker.log 2>&1 &

# Run worker in background (Windows - use startup script or Task Scheduler)
start /B python backend\worker.py > logs\worker.log 2>&1

# View worker logs (if running in background)
tail -f logs/worker.log  # Linux/Mac
type logs\worker.log     # Windows
```

## Production Deployment

### Docker Compose (Recommended)

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    container_name: voice-cloning-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    command: redis-server --appendonly yes
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 3

  worker:
    build: .
    container_name: voice-cloning-worker
    depends_on:
      - redis
    environment:
      - REDIS_HOST=redis
      - REDIS_PORT=6379
    volumes:
      - ./data:/app/data
      - ./uploads:/app/uploads
      - ./outputs:/app/outputs
      - ./models:/app/models
    restart: unless-stopped

volumes:
  redis-data:
```

### Systemd Service (Linux)

Create `/etc/systemd/system/voice-cloning-worker.service`:

```ini
[Unit]
Description=Voice Cloning ARQ Worker
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/Speech Synthesis
ExecStart=/path/to/Speech Synthesis/venv/bin/python /path/to/Speech Synthesis/backend/worker.py
Restart=always
RestartSec=10
StandardOutput=append:/var/log/voice-cloning/worker.log
StandardError=append:/var/log/voice-cloning/worker.log

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable voice-cloning-worker
sudo systemctl start voice-cloning-worker
sudo systemctl status voice-cloning-worker
```

### Windows Service (Windows)

Use **NSSM** (Non-Sucking Service Manager):

1. Download NSSM: https://nssm.cc/download
2. Install service:
   ```cmd
   nssm install VoiceCloningWorker "C:\path\to\venv\Scripts\python.exe" "C:\path\to\backend\worker.py"
   nssm set VoiceCloningWorker AppDirectory "C:\path\to\Speech Synthesis"
   nssm set VoiceCloningWorker AppStdout "C:\path\to\logs\worker.log"
   nssm set VoiceCloningWorker AppStderr "C:\path\to\logs\worker.log"
   nssm start VoiceCloningWorker
   ```

## Configuration

### Environment Variables

Create `.env` file in project root:

```bash
# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

# Worker Configuration
MAX_CONCURRENT_JOBS=1
JOB_TIMEOUT=3600
KEEP_RESULT=3600

# Database
DATABASE_URL=sqlite:///./data/voice_cloning.db

# Cleanup
CLEANUP_ENABLED=true
CLEANUP_DAYS=30
```

### Redis Configuration

Edit `backend/worker.py` if you need custom Redis settings:

```python
class WorkerSettings:
    redis_settings = RedisSettings(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        database=int(os.getenv('REDIS_DB', 0)),
        password=os.getenv('REDIS_PASSWORD', None)
    )

    max_jobs = int(os.getenv('MAX_CONCURRENT_JOBS', 1))
    job_timeout = int(os.getenv('JOB_TIMEOUT', 3600))
    keep_result = int(os.getenv('KEEP_RESULT', 3600))
```

## Monitoring

### Health Checks

```bash
# Redis health
docker exec voice-cloning-redis redis-cli ping

# Worker health (check if process is running)
# Linux/Mac:
ps aux | grep "backend/worker.py"

# Windows:
tasklist | findstr python
```

### Metrics

Access Redis stats:
```bash
docker exec voice-cloning-redis redis-cli info stats
```

Key metrics:
- `total_commands_processed`: Total commands executed
- `instantaneous_ops_per_sec`: Current operations/second
- `rejected_connections`: Connection rejections
- `expired_keys`: Expired job results

### Logging

Configure logging in `backend/worker.py`:

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/worker.log'),
        logging.StreamHandler()
    ]
)
```

## Troubleshooting

### Issue: Redis connection refused

**Symptoms**: `ConnectionRefusedError: [Errno 111] Connection refused`

**Solutions**:
1. Check if Redis is running: `docker ps | grep redis`
2. Verify port mapping: `docker port voice-cloning-redis`
3. Check firewall settings
4. Restart Redis: `docker restart voice-cloning-redis`

### Issue: Worker not processing jobs

**Symptoms**: Jobs stay in "queued" status

**Solutions**:
1. Check worker is running: `ps aux | grep worker.py`
2. Check worker logs for errors
3. Verify Redis connection from worker
4. Restart worker process

### Issue: Jobs timing out

**Symptoms**: Jobs fail with timeout errors

**Solutions**:
1. Increase `job_timeout` in `WorkerSettings`
2. Check GPU availability for training jobs
3. Verify sufficient disk space
4. Monitor system resources (CPU, RAM, GPU)

### Issue: Redis out of memory

**Symptoms**: `OOM command not allowed when used memory > 'maxmemory'`

**Solutions**:
1. Increase Docker memory limit
2. Enable Redis eviction policy
3. Reduce `keep_result` time
4. Clear old job results: `redis-cli flushdb`

## Performance Tuning

### Redis Optimization

Edit Redis configuration (create `redis.conf`):

```conf
# Memory management
maxmemory 2gb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000

# Performance
tcp-backlog 511
timeout 0
tcp-keepalive 300
```

Use custom config:
```bash
docker run -d --name voice-cloning-redis \
  -p 6379:6379 \
  -v $(pwd)/redis.conf:/usr/local/etc/redis/redis.conf \
  --restart unless-stopped \
  redis:7-alpine redis-server /usr/local/etc/redis/redis.conf
```

### Worker Optimization

- **GPU Jobs**: Keep `max_jobs=1` to avoid GPU memory issues
- **CPU Jobs**: Increase `max_jobs` for parallel processing
- **Network**: Use Redis connection pooling
- **Monitoring**: Enable detailed logging only in debug mode

## Security

### Production Checklist

- [ ] Enable Redis password authentication
- [ ] Use TLS/SSL for Redis connections
- [ ] Restrict Redis port to localhost or VPN
- [ ] Implement job result encryption
- [ ] Set up monitoring and alerting
- [ ] Regular backups of Redis data
- [ ] Audit job access logs
- [ ] Implement rate limiting

### Enable Redis Authentication

```bash
# Start Redis with password
docker run -d --name voice-cloning-redis \
  -p 6379:6379 \
  --restart unless-stopped \
  redis:7-alpine redis-server --requirepass YOUR_STRONG_PASSWORD --appendonly yes
```

Update worker settings:
```python
redis_settings = RedisSettings(
    host='localhost',
    port=6379,
    password='YOUR_STRONG_PASSWORD'
)
```

## Backup and Recovery

### Backup Redis Data

```bash
# Create backup
docker exec voice-cloning-redis redis-cli BGSAVE

# Copy backup file
docker cp voice-cloning-redis:/data/dump.rdb ./backups/redis-backup-$(date +%Y%m%d).rdb

# Automated daily backup
docker exec voice-cloning-redis redis-cli BGSAVE && \
docker cp voice-cloning-redis:/data/dump.rdb ./backups/redis-backup-$(date +%Y%m%d).rdb
```

### Restore Redis Data

```bash
# Stop Redis
docker stop voice-cloning-redis

# Copy backup to container
docker cp ./backups/redis-backup-20251015.rdb voice-cloning-redis:/data/dump.rdb

# Start Redis
docker start voice-cloning-redis
```

## Support

For issues or questions:
- Check logs: `docker logs voice-cloning-redis` and `logs/worker.log`
- Review this documentation
- Check Redis docs: https://redis.io/documentation
- Check ARQ docs: https://arq-docs.helpmanual.io/

---

**Version**: 1.0.0
**Last Updated**: 2025-10-15
**Status**: Production Ready

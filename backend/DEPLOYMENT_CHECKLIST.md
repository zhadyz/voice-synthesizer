# Backend Deployment Checklist

Quick reference for deploying the Voice Cloning API backend.

## Pre-Deployment Checklist

### ✅ Development Environment
- [ ] Python 3.10+ installed
- [ ] Virtual environment created and activated
- [ ] All dependencies installed (`pip install -r requirements_backend.txt`)
- [ ] Redis server installed and running
- [ ] CUDA drivers installed (for GPU support)
- [ ] Database initialized (`python -c "from backend.database import init_db; init_db()"`)

### ✅ Configuration
- [ ] `.env` file created from `.env.example`
- [ ] Database URL configured
- [ ] Redis connection configured
- [ ] CORS origins set for frontend
- [ ] File size limits configured
- [ ] Directory permissions set

### ✅ Testing
- [ ] Health check passes (`curl http://localhost:8000/health`)
- [ ] File upload works
- [ ] Job creation works
- [ ] SSE streaming works
- [ ] Test suite passes (`pytest tests/test_backend.py`)

---

## Quick Start (Development)

### Windows
```bash
# 1. Activate environment
venv\Scripts\activate

# 2. Start all services
run_backend.bat

# 3. Verify
curl http://localhost:8000/health

# 4. Stop services
run_backend_stop.bat
```

### Linux/Mac
```bash
# 1. Activate environment
source venv/bin/activate

# 2. Start all services
chmod +x run_backend.sh
./run_backend.sh

# 3. Verify
curl http://localhost:8000/health

# 4. Stop (CTRL+C)
```

---

## Manual Deployment

### Step 1: Start Redis
```bash
# Linux/Mac
redis-server

# Windows
redis-server.exe

# Docker (recommended for production)
docker run -d -p 6379:6379 --name redis redis:latest
```

### Step 2: Start ARQ Worker
```bash
python -m arq backend.worker.WorkerSettings

# With logging
python -m arq backend.worker.WorkerSettings > logs/worker.log 2>&1 &
```

### Step 3: Start FastAPI Server
```bash
# Development
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# Production
gunicorn backend.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log
```

---

## Production Deployment

### Option 1: Systemd Services (Linux)

Create `/etc/systemd/system/voice-cloning-api.service`:
```ini
[Unit]
Description=Voice Cloning FastAPI Server
After=network.target redis.service

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/voice-cloning
Environment="PATH=/opt/voice-cloning/venv/bin"
ExecStart=/opt/voice-cloning/venv/bin/gunicorn \
  backend.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
Restart=always

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/voice-cloning-worker.service`:
```ini
[Unit]
Description=Voice Cloning ARQ Worker
After=network.target redis.service

[Service]
Type=simple
User=www-data
WorkingDirectory=/opt/voice-cloning
Environment="PATH=/opt/voice-cloning/venv/bin"
ExecStart=/opt/voice-cloning/venv/bin/python -m arq backend.worker.WorkerSettings
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable voice-cloning-api voice-cloning-worker
sudo systemctl start voice-cloning-api voice-cloning-worker
sudo systemctl status voice-cloning-api voice-cloning-worker
```

### Option 2: Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - REDIS_HOST=redis
      - DATABASE_URL=postgresql://user:pass@db/voicecloning
    depends_on:
      - redis
      - db
    volumes:
      - ./uploads:/app/uploads
      - ./outputs:/app/outputs
      - ./models:/app/models

  worker:
    build: .
    command: python -m arq backend.worker.WorkerSettings
    environment:
      - REDIS_HOST=redis
      - DATABASE_URL=postgresql://user:pass@db/voicecloning
    depends_on:
      - redis
      - db
    volumes:
      - ./uploads:/app/uploads
      - ./outputs:/app/outputs
      - ./models:/app/models

  db:
    image: postgres:15-alpine
    environment:
      - POSTGRES_USER=voicecloning
      - POSTGRES_PASSWORD=your_secure_password
      - POSTGRES_DB=voicecloning
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  redis_data:
  postgres_data:
```

Start:
```bash
docker-compose up -d
docker-compose logs -f
```

### Option 3: Nginx Reverse Proxy

Create `/etc/nginx/sites-available/voice-cloning`:
```nginx
upstream voice_cloning_api {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name api.yourdomain.com;

    # Redirect to HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/api.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.yourdomain.com/privkey.pem;

    client_max_body_size 100M;

    location / {
        proxy_pass http://voice_cloning_api;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # SSE streaming - disable buffering
    location /api/stream/ {
        proxy_pass http://voice_cloning_api/api/stream/;
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection '';
        proxy_http_version 1.1;
        chunked_transfer_encoding off;
    }
}
```

Enable:
```bash
sudo ln -s /etc/nginx/sites-available/voice-cloning /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

---

## Environment Variables

### Required
```env
DATABASE_URL=sqlite:///./data/voice_cloning.db
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Optional
```env
API_HOST=0.0.0.0
API_PORT=8000
MAX_FILE_SIZE_MB=100
MAX_CONCURRENT_JOBS=1
JOB_TIMEOUT=3600
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
```

### Production
```env
DEBUG=False
DATABASE_URL=postgresql://user:pass@localhost/voicecloning
REDIS_PASSWORD=your_secure_password
LOG_LEVEL=WARNING
```

---

## Monitoring & Logs

### Check Service Status
```bash
# Systemd
sudo systemctl status voice-cloning-api
sudo systemctl status voice-cloning-worker

# Docker
docker-compose ps
docker-compose logs api
docker-compose logs worker

# Manual
ps aux | grep uvicorn
ps aux | grep arq
```

### View Logs
```bash
# Worker logs
tail -f logs/worker.log

# API logs (if using Gunicorn)
tail -f logs/access.log
tail -f logs/error.log

# Nginx logs
sudo tail -f /var/log/nginx/access.log
sudo tail -f /var/log/nginx/error.log
```

### Health Check
```bash
# Basic check
curl http://localhost:8000/health

# Detailed check
curl http://localhost:8000/health | jq .

# Expected response
{
  "status": "ok",
  "version": "1.0.0",
  "gpu_available": true,
  "gpu_name": "NVIDIA RTX 3090",
  "redis_connected": true,
  "database_connected": true
}
```

---

## Troubleshooting

### Redis Connection Failed
```bash
# Check Redis is running
redis-cli ping

# Should return: PONG

# If not running:
redis-server  # Start Redis

# Check Redis logs
redis-cli INFO | grep redis_version
```

### Port Already in Use
```bash
# Find process using port 8000
# Linux/Mac
lsof -i :8000

# Windows
netstat -ano | findstr :8000

# Kill process
kill -9 <PID>  # Linux/Mac
taskkill /PID <PID> /F  # Windows
```

### Database Errors
```bash
# Reinitialize database
rm data/voice_cloning.db
python -c "from backend.database import init_db; init_db()"

# For PostgreSQL
psql -U postgres
DROP DATABASE voicecloning;
CREATE DATABASE voicecloning;
```

### Worker Not Processing Jobs
```bash
# Check ARQ worker is running
ps aux | grep arq

# Check Redis queue
redis-cli
> KEYS mendicant_bias:*
> LLEN mendicant_bias:queue

# Restart worker
pkill -f arq
python -m arq backend.worker.WorkerSettings &
```

### GPU Not Detected
```bash
# Check NVIDIA drivers
nvidia-smi

# Check PyTorch GPU
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Performance Tuning

### API Server
```bash
# Increase workers (CPU-bound)
gunicorn backend.main:app --workers 4

# Increase timeout for long requests
gunicorn backend.main:app --timeout 120

# Increase worker connections
gunicorn backend.main:app --worker-connections 1000
```

### Database
```python
# Use PostgreSQL for production
DATABASE_URL=postgresql://user:pass@localhost/voicecloning

# Enable connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=20
)
```

### Redis
```bash
# Increase memory
redis-server --maxmemory 2gb

# Enable persistence
redis-server --appendonly yes

# Use Redis Cluster for scaling
```

---

## Security Checklist

- [ ] Enable HTTPS (Let's Encrypt)
- [ ] Add authentication (JWT tokens)
- [ ] Implement rate limiting
- [ ] Validate file contents (not just extensions)
- [ ] Sanitize user inputs
- [ ] Set secure CORS origins
- [ ] Use environment variables for secrets
- [ ] Enable Redis password
- [ ] Use database connection encryption
- [ ] Set up firewall rules
- [ ] Regular security updates

---

## Backup & Recovery

### Database Backup
```bash
# SQLite
cp data/voice_cloning.db backups/voice_cloning_$(date +%Y%m%d).db

# PostgreSQL
pg_dump voicecloning > backups/voicecloning_$(date +%Y%m%d).sql
```

### File Backup
```bash
# Backup uploads and models
tar -czf backups/files_$(date +%Y%m%d).tar.gz uploads/ outputs/ models/

# Restore
tar -xzf backups/files_20250101.tar.gz
```

---

## Scaling Strategies

### Horizontal Scaling
1. Load balancer (nginx/HAProxy)
2. Multiple API instances
3. Shared Redis cluster
4. Shared PostgreSQL/cloud database
5. S3/Azure Blob for file storage

### Vertical Scaling
1. Increase worker count
2. Upgrade GPU
3. Add more RAM
4. Faster SSD storage

### Optimization
1. Enable Redis caching
2. Use CDN for static files
3. Implement request batching
4. Database query optimization
5. Async I/O everywhere

---

## Success Metrics

Monitor these metrics:

- API response time (< 200ms for most endpoints)
- Upload success rate (> 99%)
- Job completion rate (> 95%)
- Worker queue length (< 10)
- Database connection pool usage (< 80%)
- Redis memory usage (< 80%)
- Disk space available (> 20%)
- GPU utilization during training (> 90%)

---

## Support

For issues or questions:
- Check logs: `logs/worker.log`, `logs/error.log`
- View API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- Redis status: `redis-cli INFO`
- Database status: Check `data/voice_cloning.db`

---

**Last Updated**: 2025-10-13
**Version**: 1.0.0

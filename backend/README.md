# Voice Cloning Backend API

Production-ready FastAPI backend for voice cloning with RVC (Retrieval-based Voice Conversion).

## Features

- **File Upload**: Support for MP3, WAV, M4A, FLAC, OGG, AAC formats (up to 100MB)
- **Job Queue**: ARQ-based async job processing with Redis
- **Real-time Progress**: Server-Sent Events (SSE) for live progress tracking
- **Database**: SQLite with SQLAlchemy ORM (upgradeable to PostgreSQL)
- **Model Management**: Store and manage trained voice models
- **RESTful API**: Full CRUD operations with automatic OpenAPI docs

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Frontend  │────▶│   FastAPI   │────▶│   SQLite    │
│   (React)   │     │   Backend   │     │  Database   │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │    Redis    │
                    │   + ARQ     │
                    └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │ RVC Pipeline│
                    │   (GPU)     │
                    └─────────────┘
```

## Quick Start

### Prerequisites

- Python 3.10+
- Redis server
- CUDA-capable GPU (recommended)

### Installation

1. **Install Redis**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install redis-server

   # macOS
   brew install redis

   # Windows
   # Download from https://github.com/microsoftarchive/redis/releases
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize database**:
   ```bash
   python -c "from backend.database import init_db; init_db()"
   ```

### Running the Backend

**Windows**:
```bash
run_backend.bat
```

**Linux/Mac**:
```bash
chmod +x run_backend.sh
./run_backend.sh
```

**Manual start**:
```bash
# Terminal 1: Redis
redis-server

# Terminal 2: ARQ Worker
python -m arq backend.worker.WorkerSettings

# Terminal 3: FastAPI
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

### Stopping the Backend

**Windows**:
```bash
run_backend_stop.bat
```

**Linux/Mac**:
```bash
# Press CTRL+C in the terminal running run_backend.sh
# Or kill processes:
pkill -f uvicorn
pkill -f arq
redis-cli shutdown
```

## API Endpoints

### Base URL
```
http://localhost:8000
```

### Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

#### Upload
- `POST /api/upload/training-audio` - Upload voice recording for training
- `POST /api/upload/target-audio` - Upload audio to convert
- `GET /api/upload/validate/{job_id}` - Validate uploaded audio

#### Jobs
- `POST /api/jobs/train` - Start training job
- `POST /api/jobs/convert` - Start conversion job
- `GET /api/jobs/status/{job_id}` - Get job status
- `GET /api/jobs/list` - List jobs with filters
- `DELETE /api/jobs/{job_id}` - Cancel job
- `POST /api/jobs/{job_id}/retry` - Retry failed job

#### Streaming (SSE)
- `GET /api/stream/progress/{job_id}` - Real-time job progress
- `GET /api/stream/multi-progress?user_id={id}` - Monitor multiple jobs

#### Download
- `GET /api/download/audio/{job_id}` - Download output audio
- `GET /api/download/audio/{job_id}/stream` - Stream audio in browser
- `GET /api/download/input/{job_id}` - Download input audio

#### Models
- `GET /api/models/list` - List trained models
- `GET /api/models/{model_id}` - Get model details
- `DELETE /api/models/{model_id}` - Delete model
- `GET /api/models/{model_id}/stats` - Get model statistics

#### Health
- `GET /health` - Health check (GPU, DB, Redis status)
- `GET /api/info` - API information

## Usage Examples

### Upload Training Audio

```python
import requests

url = "http://localhost:8000/api/upload/training-audio"
files = {"file": open("my_voice.mp3", "rb")}
data = {"user_id": "user123"}

response = requests.post(url, files=files, data=data)
job_id = response.json()["job_id"]
print(f"Job ID: {job_id}")
```

### Start Training

```python
url = "http://localhost:8000/api/jobs/train"
data = {
    "job_id": job_id,
    "model_name": "my_voice_model"
}

response = requests.post(url, json=data)
print(response.json())
```

### Monitor Progress (SSE)

```python
import requests
import json

url = f"http://localhost:8000/api/stream/progress/{job_id}"

with requests.get(url, stream=True) as response:
    for line in response.iter_lines():
        if line and line.startswith(b"data:"):
            data = json.loads(line.decode()[5:])
            print(f"Progress: {data['progress']*100:.0f}% - {data['message']}")
```

### JavaScript/Frontend Example

```javascript
// SSE in browser
const eventSource = new EventSource(
  `http://localhost:8000/api/stream/progress/${jobId}`
);

eventSource.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log(`Progress: ${data.progress * 100}%`);

  if (data.status === "completed") {
    eventSource.close();
    downloadAudio(data.job_id);
  }
};

eventSource.onerror = (error) => {
  console.error("SSE error:", error);
  eventSource.close();
};
```

## Database Schema

### Jobs Table
```sql
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    type TEXT NOT NULL,  -- 'training' or 'conversion'
    status TEXT NOT NULL,  -- 'pending', 'preprocessing', etc.
    progress REAL DEFAULT 0.0,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    completed_at TIMESTAMP,
    input_audio_path TEXT,
    output_audio_path TEXT,
    model_id TEXT,
    quality_snr REAL,
    quality_pesq REAL,
    error_message TEXT
);
```

### Voice Models Table
```sql
CREATE TABLE voice_models (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_path TEXT NOT NULL,
    training_audio_path TEXT,
    training_duration_seconds REAL,
    created_at TIMESTAMP,
    quality_snr REAL,
    quality_score REAL,
    training_job_id TEXT
);
```

## Configuration

### Environment Variables

Create `.env` file:
```env
# Database
DATABASE_URL=sqlite:///./data/voice_cloning.db

# Redis
REDIS_HOST=localhost
REDIS_PORT=6379

# API
API_HOST=0.0.0.0
API_PORT=8000

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173

# File Upload
MAX_FILE_SIZE_MB=100
UPLOAD_DIR=./uploads

# Worker
MAX_CONCURRENT_JOBS=1
JOB_TIMEOUT=3600
```

## Testing

Run test suite:
```bash
# Make sure backend is running first
pytest tests/test_backend.py -v

# Run specific test
pytest tests/test_backend.py::TestFileUpload::test_upload_training_audio -v

# Run with coverage
pytest tests/ --cov=backend --cov-report=html
```

## Performance

### Expected Timings
- **File Upload**: < 1 second (for 50MB file)
- **Preprocessing**: 30-60 seconds
- **Training**: 30-40 minutes (40GB dataset, RTX 3090)
- **Conversion**: 1-5 minutes (depending on audio length)

### Optimization Tips
1. **Use SSD for uploads/outputs**: Faster I/O
2. **Increase worker count**: For parallel conversions (not training)
3. **Use PostgreSQL**: Better performance for high traffic
4. **Add caching**: Redis cache for model metadata
5. **Enable GPU**: 10-20x faster than CPU

## Troubleshooting

### Redis Connection Error
```
Error: Could not connect to Redis
```
**Solution**: Start Redis server
```bash
redis-server
```

### Database Locked Error
```
sqlite3.OperationalError: database is locked
```
**Solution**: Increase timeout or switch to PostgreSQL
```python
engine = create_engine(
    DATABASE_URL,
    connect_args={"timeout": 30}
)
```

### Port Already in Use
```
ERROR: Address already in use
```
**Solution**: Change port or kill existing process
```bash
# Find process
lsof -i :8000  # Linux/Mac
netstat -ano | findstr :8000  # Windows

# Kill process
kill <PID>  # Linux/Mac
taskkill /PID <PID> /F  # Windows
```

## Production Deployment

### Docker
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Nginx Reverse Proxy
```nginx
server {
    listen 80;
    server_name api.voicecloning.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api/stream/ {
        proxy_pass http://localhost:8000/api/stream/;
        proxy_buffering off;
        proxy_cache off;
        proxy_set_header Connection '';
        chunked_transfer_encoding off;
    }
}
```

## License

MIT License - see LICENSE file

## Contributors

- HOLLOWED_EYES (Backend Architecture & Implementation)

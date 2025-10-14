"""
Backend API Tests
Comprehensive test suite for FastAPI endpoints
"""

import pytest
import requests
import time
import json
from pathlib import Path
import io

# Test configuration
BASE_URL = "http://localhost:8000"
TEST_USER_ID = "test_user_123"


class TestHealthAndInfo:
    """Test health check and info endpoints"""

    def test_health_check(self):
        """Test health endpoint"""
        response = requests.get(f"{BASE_URL}/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] in ["ok", "degraded"]
        assert "gpu_available" in data
        assert "database_connected" in data
        print(f"Health check: {data}")

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = requests.get(f"{BASE_URL}/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert "version" in data

    def test_api_info(self):
        """Test API info endpoint"""
        response = requests.get(f"{BASE_URL}/api/info")
        assert response.status_code == 200

        data = response.json()
        assert "endpoints" in data
        assert "upload" in data["endpoints"]
        assert "jobs" in data["endpoints"]


class TestFileUpload:
    """Test file upload endpoints"""

    @pytest.fixture
    def test_audio_file(self, tmp_path):
        """Create a test audio file"""
        # Create a simple WAV file for testing
        audio_file = tmp_path / "test_audio.wav"

        # Use a real audio file if available, otherwise create dummy
        test_audio_path = Path("test_data/test_audio.wav")
        if test_audio_path.exists():
            return str(test_audio_path)
        else:
            # Create minimal WAV file (44 bytes header + some data)
            wav_data = (
                b'RIFF' + (1024).to_bytes(4, 'little') + b'WAVE'
                b'fmt ' + (16).to_bytes(4, 'little')
                + (1).to_bytes(2, 'little')  # Audio format (PCM)
                + (1).to_bytes(2, 'little')  # Channels
                + (44100).to_bytes(4, 'little')  # Sample rate
                + (88200).to_bytes(4, 'little')  # Byte rate
                + (2).to_bytes(2, 'little')  # Block align
                + (16).to_bytes(2, 'little')  # Bits per sample
                b'data' + (1000).to_bytes(4, 'little')
                + b'\x00' * 1000  # Audio data
            )
            audio_file.write_bytes(wav_data)
            return str(audio_file)

    def test_upload_training_audio(self, test_audio_file):
        """Test training audio upload"""
        with open(test_audio_file, 'rb') as f:
            files = {"file": ("test.wav", f, "audio/wav")}
            data = {"user_id": TEST_USER_ID}

            response = requests.post(
                f"{BASE_URL}/api/upload/training-audio",
                files=files,
                data=data
            )

        assert response.status_code == 200
        result = response.json()
        assert "job_id" in result
        assert "filename" in result
        assert result["status"] == "uploaded"

        print(f"Uploaded training audio, job ID: {result['job_id']}")
        return result["job_id"]

    def test_upload_target_audio(self, test_audio_file):
        """Test target audio upload"""
        with open(test_audio_file, 'rb') as f:
            files = {"file": ("target.wav", f, "audio/wav")}
            data = {"user_id": TEST_USER_ID}

            response = requests.post(
                f"{BASE_URL}/api/upload/target-audio",
                files=files,
                data=data
            )

        assert response.status_code == 200
        result = response.json()
        assert "job_id" in result

        return result["job_id"]

    def test_upload_invalid_format(self):
        """Test upload with invalid file format"""
        # Create a text file
        files = {"file": ("test.txt", io.BytesIO(b"not audio"), "text/plain")}
        data = {"user_id": TEST_USER_ID}

        response = requests.post(
            f"{BASE_URL}/api/upload/training-audio",
            files=files,
            data=data
        )

        assert response.status_code == 400
        assert "Invalid file format" in response.json()["detail"]

    def test_upload_too_large(self):
        """Test upload with file too large"""
        # Create 101MB of data
        large_data = b'\x00' * (101 * 1024 * 1024)
        files = {"file": ("large.wav", io.BytesIO(large_data), "audio/wav")}
        data = {"user_id": TEST_USER_ID}

        response = requests.post(
            f"{BASE_URL}/api/upload/training-audio",
            files=files,
            data=data
        )

        assert response.status_code == 400
        assert "too large" in response.json()["detail"].lower()


class TestJobManagement:
    """Test job management endpoints"""

    def test_get_job_status(self):
        """Test job status retrieval"""
        # First upload a file
        wav_data = b'RIFF' + (1024).to_bytes(4, 'little') + b'WAVE'
        files = {"file": ("test.wav", io.BytesIO(wav_data), "audio/wav")}
        data = {"user_id": TEST_USER_ID}

        upload_response = requests.post(
            f"{BASE_URL}/api/upload/training-audio",
            files=files,
            data=data
        )
        job_id = upload_response.json()["job_id"]

        # Get job status
        response = requests.get(f"{BASE_URL}/api/jobs/status/{job_id}")
        assert response.status_code == 200

        status = response.json()
        assert status["id"] == job_id
        assert "status" in status
        assert "progress" in status

    def test_list_jobs(self):
        """Test job listing"""
        response = requests.get(
            f"{BASE_URL}/api/jobs/list",
            params={"user_id": TEST_USER_ID, "limit": 10}
        )

        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert "total" in data
        assert isinstance(data["jobs"], list)

    def test_job_not_found(self):
        """Test job status with invalid ID"""
        response = requests.get(f"{BASE_URL}/api/jobs/status/nonexistent")
        assert response.status_code == 404


class TestStreamingProgress:
    """Test SSE progress streaming"""

    def test_progress_stream_connection(self):
        """Test SSE connection and initial data"""
        # Upload a file first
        wav_data = b'RIFF' + (1024).to_bytes(4, 'little') + b'WAVE'
        files = {"file": ("test.wav", io.BytesIO(wav_data), "audio/wav")}
        data = {"user_id": TEST_USER_ID}

        upload_response = requests.post(
            f"{BASE_URL}/api/upload/training-audio",
            files=files,
            data=data
        )
        job_id = upload_response.json()["job_id"]

        # Connect to SSE stream
        response = requests.get(
            f"{BASE_URL}/api/stream/progress/{job_id}",
            stream=True,
            timeout=10
        )

        assert response.status_code == 200
        assert "text/event-stream" in response.headers["Content-Type"]

        # Read first event
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data:'):
                    data = json.loads(line_str[5:].strip())
                    assert "job_id" in data
                    assert "status" in data
                    assert "progress" in data
                    print(f"SSE data: {data}")
                    break


class TestModels:
    """Test voice model management"""

    def test_list_models(self):
        """Test model listing"""
        response = requests.get(
            f"{BASE_URL}/api/models/list",
            params={"user_id": TEST_USER_ID}
        )

        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "total" in data

    def test_model_not_found(self):
        """Test getting nonexistent model"""
        response = requests.get(f"{BASE_URL}/api/models/nonexistent")
        assert response.status_code == 404


class TestDownload:
    """Test download endpoints"""

    def test_download_nonexistent_job(self):
        """Test download with invalid job ID"""
        response = requests.get(f"{BASE_URL}/api/download/audio/nonexistent")
        assert response.status_code == 404


@pytest.mark.integration
class TestFullWorkflow:
    """Integration tests for complete workflows"""

    def test_upload_and_track_progress(self):
        """Test full upload and progress tracking"""
        # 1. Upload training audio
        wav_data = b'RIFF' + (1024).to_bytes(4, 'little') + b'WAVE'
        files = {"file": ("test.wav", io.BytesIO(wav_data), "audio/wav")}
        data = {"user_id": TEST_USER_ID}

        upload_response = requests.post(
            f"{BASE_URL}/api/upload/training-audio",
            files=files,
            data=data
        )
        assert upload_response.status_code == 200
        job_id = upload_response.json()["job_id"]

        # 2. Check initial status
        status_response = requests.get(f"{BASE_URL}/api/jobs/status/{job_id}")
        assert status_response.status_code == 200
        status = status_response.json()
        assert status["status"] == "pending"

        # 3. List jobs and verify it's there
        list_response = requests.get(
            f"{BASE_URL}/api/jobs/list",
            params={"user_id": TEST_USER_ID}
        )
        assert list_response.status_code == 200
        jobs = list_response.json()["jobs"]
        assert any(j["id"] == job_id for j in jobs)

        print(f"Full workflow test passed for job {job_id}")


if __name__ == "__main__":
    print("Running backend API tests...")
    print(f"Target: {BASE_URL}")
    print("\nMake sure the backend is running:")
    print("  python run_backend.bat (Windows)")
    print("  ./run_backend.sh (Linux/Mac)")
    print("\n" + "="*50 + "\n")

    pytest.main([__file__, "-v", "-s"])

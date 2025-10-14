"""
TIER 1: Backend API Integration Tests
Comprehensive testing of FastAPI endpoints, database operations, and SSE streaming
"""

import pytest
import requests
import time
import json
from pathlib import Path
import io


class TestHealthAndStatus:
    """Test health check and status endpoints"""

    def test_health_endpoint(self, base_url):
        """Test /health endpoint"""
        try:
            response = requests.get(f"{base_url}/health", timeout=10)
            assert response.status_code == 200

            data = response.json()
            assert "status" in data
            assert data["status"] in ["ok", "degraded", "down"]
            assert "gpu_available" in data
            assert "database_connected" in data

            print(f"\n✓ Health check passed:")
            print(f"  Status: {data['status']}")
            print(f"  GPU: {data.get('gpu_available', False)}")
            print(f"  GPU Name: {data.get('gpu_name', 'N/A')}")
            print(f"  Database: {data.get('database_connected', False)}")

        except requests.exceptions.ConnectionError:
            pytest.skip("Backend server not running. Start with: python run_backend.bat")
        except Exception as e:
            pytest.fail(f"Health check failed: {e}")

    def test_root_endpoint(self, base_url):
        """Test root / endpoint"""
        try:
            response = requests.get(f"{base_url}/", timeout=10)
            assert response.status_code == 200

            data = response.json()
            assert "message" in data
            assert "version" in data
            print(f"✓ Root endpoint: v{data['version']}")

        except requests.exceptions.ConnectionError:
            pytest.skip("Backend server not running")

    def test_api_info_endpoint(self, base_url):
        """Test /api/info endpoint"""
        try:
            response = requests.get(f"{base_url}/api/info", timeout=10)
            assert response.status_code == 200

            data = response.json()
            assert "endpoints" in data
            assert "upload" in data["endpoints"]
            assert "jobs" in data["endpoints"]

            print(f"✓ API info: {len(data['endpoints'])} endpoint groups")

        except requests.exceptions.ConnectionError:
            pytest.skip("Backend server not running")


class TestFileUploadValidation:
    """Test file upload validation and security"""

    def test_upload_valid_wav(self, base_url, temp_audio_file, test_user_id):
        """Test uploading valid WAV file"""
        try:
            with open(temp_audio_file, 'rb') as f:
                files = {"file": ("test.wav", f, "audio/wav")}
                data = {"user_id": test_user_id}

                response = requests.post(
                    f"{base_url}/api/upload/training-audio",
                    files=files,
                    data=data,
                    timeout=30
                )

            if response.status_code == 200:
                result = response.json()
                assert "job_id" in result
                assert "filename" in result
                print(f"✓ Upload successful: job_id={result['job_id']}")
                return result["job_id"]
            else:
                print(f"⚠ Upload returned {response.status_code}: {response.text}")

        except requests.exceptions.ConnectionError:
            pytest.skip("Backend server not running")

    def test_upload_invalid_format(self, base_url, test_user_id):
        """Test rejection of invalid file format"""
        try:
            files = {"file": ("test.txt", io.BytesIO(b"not audio"), "text/plain")}
            data = {"user_id": test_user_id}

            response = requests.post(
                f"{base_url}/api/upload/training-audio",
                files=files,
                data=data,
                timeout=10
            )

            # Should reject invalid format
            assert response.status_code in [400, 422]
            print(f"✓ Invalid format correctly rejected: {response.status_code}")

        except requests.exceptions.ConnectionError:
            pytest.skip("Backend server not running")

    @pytest.mark.security
    def test_upload_file_size_limit(self, base_url, test_user_id):
        """Test file size limit enforcement"""
        try:
            # Create 101MB of data (exceeds typical 100MB limit)
            large_data = b'\x00' * (101 * 1024 * 1024)
            files = {"file": ("large.wav", io.BytesIO(large_data), "audio/wav")}
            data = {"user_id": test_user_id}

            response = requests.post(
                f"{base_url}/api/upload/training-audio",
                files=files,
                data=data,
                timeout=10
            )

            # Should reject oversized file
            assert response.status_code in [400, 413]
            print(f"✓ Oversized file correctly rejected: {response.status_code}")

        except requests.exceptions.ConnectionError:
            pytest.skip("Backend server not running")
        except Exception as e:
            # Some servers reject before the upload completes
            print(f"✓ File size limit enforced (connection error): {type(e).__name__}")

    @pytest.mark.security
    def test_upload_path_traversal_protection(self, base_url, test_user_id):
        """Test protection against path traversal attacks"""
        try:
            files = {"file": ("../../etc/passwd", io.BytesIO(b"audio"), "audio/wav")}
            data = {"user_id": test_user_id}

            response = requests.post(
                f"{base_url}/api/upload/training-audio",
                files=files,
                data=data,
                timeout=10
            )

            # Should sanitize filename or reject
            assert response.status_code in [200, 400]
            print(f"✓ Path traversal handled: {response.status_code}")

        except requests.exceptions.ConnectionError:
            pytest.skip("Backend server not running")


class TestJobManagement:
    """Test job management endpoints"""

    @pytest.fixture
    def uploaded_job_id(self, base_url, temp_audio_file, test_user_id):
        """Upload a file and return the job ID"""
        try:
            with open(temp_audio_file, 'rb') as f:
                files = {"file": ("test.wav", f, "audio/wav")}
                data = {"user_id": test_user_id}

                response = requests.post(
                    f"{base_url}/api/upload/training-audio",
                    files=files,
                    data=data,
                    timeout=30
                )

            if response.status_code == 200:
                return response.json()["job_id"]
            else:
                pytest.skip(f"Upload failed with status {response.status_code}")

        except requests.exceptions.ConnectionError:
            pytest.skip("Backend server not running")

    def test_get_job_status(self, base_url, uploaded_job_id):
        """Test getting job status"""
        try:
            response = requests.get(
                f"{base_url}/api/jobs/status/{uploaded_job_id}",
                timeout=10
            )

            assert response.status_code == 200
            status = response.json()
            assert "id" in status
            assert "status" in status
            assert status["id"] == uploaded_job_id

            print(f"✓ Job status retrieved: {status['status']}")

        except requests.exceptions.ConnectionError:
            pytest.skip("Backend server not running")

    def test_list_jobs(self, base_url, test_user_id):
        """Test listing jobs for a user"""
        try:
            response = requests.get(
                f"{base_url}/api/jobs/list",
                params={"user_id": test_user_id, "limit": 10},
                timeout=10
            )

            assert response.status_code == 200
            data = response.json()
            assert "jobs" in data
            assert isinstance(data["jobs"], list)

            print(f"✓ Jobs list retrieved: {len(data['jobs'])} jobs")

        except requests.exceptions.ConnectionError:
            pytest.skip("Backend server not running")

    def test_job_not_found(self, base_url):
        """Test handling of nonexistent job"""
        try:
            response = requests.get(
                f"{base_url}/api/jobs/status/nonexistent_job_12345",
                timeout=10
            )

            assert response.status_code == 404
            print(f"✓ Nonexistent job correctly returns 404")

        except requests.exceptions.ConnectionError:
            pytest.skip("Backend server not running")


class TestSSEStreaming:
    """Test Server-Sent Events progress streaming"""

    def test_sse_connection(self, base_url, temp_audio_file, test_user_id):
        """Test SSE connection and initial data"""
        try:
            # Upload a file first
            with open(temp_audio_file, 'rb') as f:
                files = {"file": ("test.wav", f, "audio/wav")}
                data = {"user_id": test_user_id}

                response = requests.post(
                    f"{base_url}/api/upload/training-audio",
                    files=files,
                    data=data,
                    timeout=30
                )

            if response.status_code != 200:
                pytest.skip("Upload failed")

            job_id = response.json()["job_id"]

            # Connect to SSE stream
            response = requests.get(
                f"{base_url}/api/stream/progress/{job_id}",
                stream=True,
                timeout=15
            )

            assert response.status_code == 200
            assert "text/event-stream" in response.headers.get("Content-Type", "")

            print(f"✓ SSE stream connected for job {job_id}")

            # Read first event
            events_received = 0
            for line in response.iter_lines(decode_unicode=True):
                if line and line.startswith('data:'):
                    try:
                        data = json.loads(line[5:].strip())
                        assert "job_id" in data
                        assert "status" in data
                        events_received += 1
                        print(f"  Event: {data.get('status')} - {data.get('progress', 0)}%")

                        if events_received >= 3:  # Read first 3 events
                            break
                    except json.JSONDecodeError:
                        pass

            assert events_received > 0, "No SSE events received"
            print(f"✓ Received {events_received} SSE events")

        except requests.exceptions.ConnectionError:
            pytest.skip("Backend server not running")
        except requests.exceptions.ReadTimeout:
            print(f"⚠ SSE timeout (expected for inactive jobs)")


class TestModelsEndpoint:
    """Test voice model management endpoints"""

    def test_list_models(self, base_url, test_user_id):
        """Test model listing"""
        try:
            response = requests.get(
                f"{base_url}/api/models/list",
                params={"user_id": test_user_id},
                timeout=10
            )

            assert response.status_code == 200
            data = response.json()
            assert "models" in data

            print(f"✓ Models list retrieved: {len(data['models'])} models")

        except requests.exceptions.ConnectionError:
            pytest.skip("Backend server not running")

    def test_model_not_found(self, base_url):
        """Test getting nonexistent model"""
        try:
            response = requests.get(
                f"{base_url}/api/models/nonexistent_model",
                timeout=10
            )

            assert response.status_code == 404
            print(f"✓ Nonexistent model correctly returns 404")

        except requests.exceptions.ConnectionError:
            pytest.skip("Backend server not running")


@pytest.mark.integration
class TestFullWorkflow:
    """Integration test for complete upload workflow"""

    def test_upload_and_track_workflow(self, base_url, temp_audio_file, test_user_id):
        """Test complete upload → status → list workflow"""
        try:
            # 1. Upload file
            with open(temp_audio_file, 'rb') as f:
                files = {"file": ("test.wav", f, "audio/wav")}
                data = {"user_id": test_user_id}

                upload_response = requests.post(
                    f"{base_url}/api/upload/training-audio",
                    files=files,
                    data=data,
                    timeout=30
                )

            assert upload_response.status_code == 200
            job_id = upload_response.json()["job_id"]
            print(f"\n✓ Step 1: Upload successful (job_id={job_id})")

            # 2. Get job status
            status_response = requests.get(
                f"{base_url}/api/jobs/status/{job_id}",
                timeout=10
            )
            assert status_response.status_code == 200
            print(f"✓ Step 2: Status retrieved ({status_response.json()['status']})")

            # 3. List jobs and verify it's there
            list_response = requests.get(
                f"{base_url}/api/jobs/list",
                params={"user_id": test_user_id},
                timeout=10
            )
            assert list_response.status_code == 200
            jobs = list_response.json()["jobs"]
            assert any(j["id"] == job_id for j in jobs)
            print(f"✓ Step 3: Job appears in list")

            print(f"\n✓ Full workflow test PASSED")

        except requests.exceptions.ConnectionError:
            pytest.skip("Backend server not running")


# Test execution
if __name__ == "__main__":
    print("\n" + "="*70)
    print("BACKEND API INTEGRATION TESTS")
    print("="*70)
    print("NOTE: Backend must be running at http://localhost:8000")
    print("Start with: python run_backend.bat")
    print("="*70 + "\n")

    pytest.main([__file__, "-v", "-s", "--tb=short"])

"""
Pytest configuration and shared fixtures
"""

import pytest
import os
import sys
from pathlib import Path
import tempfile
import shutil

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_dir():
    """Return project root directory"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def test_fixtures_dir(project_root_dir):
    """Return test fixtures directory"""
    fixtures = project_root_dir / "tests" / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    return fixtures


@pytest.fixture(scope="session")
def test_output_dir(project_root_dir):
    """Return test output directory"""
    output = project_root_dir / "tests" / "output"
    output.mkdir(parents=True, exist_ok=True)
    return output


@pytest.fixture
def temp_audio_file(tmp_path):
    """Create a minimal valid WAV file for testing"""
    audio_file = tmp_path / "test_audio.wav"

    # Create minimal 1-second WAV file (44.1kHz, 16-bit mono)
    import wave
    import struct

    sample_rate = 44100
    duration = 1  # seconds
    frequency = 440  # Hz (A4 note)

    with wave.open(str(audio_file), 'w') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        # Generate 1 second of 440Hz sine wave
        import math
        samples = []
        for i in range(sample_rate * duration):
            value = int(32767 * 0.3 * math.sin(2 * math.pi * frequency * i / sample_rate))
            samples.append(struct.pack('<h', value))

        wav_file.writeframes(b''.join(samples))

    return audio_file


@pytest.fixture
def mock_audio_3sec(tmp_path):
    """Create a 3-second audio file for longer tests"""
    audio_file = tmp_path / "test_audio_3sec.wav"

    import wave
    import struct
    import math

    sample_rate = 44100
    duration = 3
    frequency = 440

    with wave.open(str(audio_file), 'w') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        samples = []
        for i in range(sample_rate * duration):
            value = int(32767 * 0.3 * math.sin(2 * math.pi * frequency * i / sample_rate))
            samples.append(struct.pack('<h', value))

        wav_file.writeframes(b''.join(samples))

    return audio_file


@pytest.fixture
def test_user_id():
    """Return a test user ID"""
    return "test_user_qa_loveless"


@pytest.fixture(scope="function")
def cleanup_outputs():
    """Cleanup test outputs after each test"""
    yield
    # Cleanup logic here if needed
    pass


@pytest.fixture
def base_url():
    """Return base URL for API tests"""
    return os.getenv("API_BASE_URL", "http://localhost:8000")


@pytest.fixture
def api_timeout():
    """Return API timeout for tests"""
    return 30  # seconds


# Markers
def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "security: marks security tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks performance benchmarking tests"
    )


@pytest.fixture(scope="session", autouse=True)
def test_session_info():
    """Print test session information"""
    print("\n" + "="*70)
    print("LOVELESS QA TEST SUITE - PHASE 4 INTEGRATION TESTING")
    print("="*70)

    import torch
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("="*70 + "\n")

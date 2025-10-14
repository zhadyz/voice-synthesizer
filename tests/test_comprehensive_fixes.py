"""
Comprehensive Verification Suite for Critical Fixes
Tests all three critical issues:
1. PyTorch CUDA support
2. Python 3.13 compatibility
3. Voice isolation functionality
"""

import sys
import torch
import torchaudio
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.quality_validator import QualityValidator
from src.preprocessing.voice_isolator import VoiceIsolator


def print_header(title):
    """Print a nice header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def test_pytorch_cuda():
    """Test PyTorch CUDA installation"""
    print_header("TEST 1: PyTorch CUDA Support")

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if not cuda_available:
        print("[WARN] CUDA not available - GPU will not be used")
        return False

    # Get GPU details
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    pytorch_version = torch.__version__

    print(f"GPU Count: {gpu_count}")
    print(f"GPU Name: {gpu_name}")
    print(f"CUDA Version: {cuda_version}")
    print(f"PyTorch Version: {pytorch_version}")

    # Test GPU operation
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.matmul(x, y)
        assert z.is_cuda, "Result not on GPU"
        print("[PASS] GPU operations working")
        return True
    except Exception as e:
        print(f"[FAIL] GPU operation failed: {e}")
        return False


def test_python_compatibility():
    """Test Python 3.13 compatibility with audio loading"""
    print_header("TEST 2: Python 3.13 Compatibility")

    python_version = sys.version
    print(f"Python Version: {python_version}")

    # Test torchaudio loading (replaces librosa)
    try:
        # Create a simple test audio
        test_dir = Path(__file__).parent / "fixtures"
        test_dir.mkdir(exist_ok=True)
        test_audio = test_dir / "compat_test.wav"

        # Generate test audio
        t = torch.linspace(0, 1, 22050)
        audio = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)
        torchaudio.save(str(test_audio), audio, 22050)

        # Test loading
        waveform, sr = torchaudio.load(str(test_audio))
        print(f"[PASS] Torchaudio loading works: shape={waveform.shape}, sr={sr}")

        # Test QualityValidator (uses torchaudio internally)
        validator = QualityValidator()
        results = validator.validate_audio(str(test_audio))
        print(f"[PASS] QualityValidator works with Python 3.13")
        print(f"  - Duration: {results['duration_sec']:.2f}s")
        print(f"  - Sample Rate: {results['sample_rate']} Hz")

        return True
    except Exception as e:
        print(f"[FAIL] Python 3.13 compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_voice_isolation():
    """Test voice isolation with BS-RoFormer"""
    print_header("TEST 3: Voice Isolation")

    try:
        # Create test audio with harmonics
        test_dir = Path(__file__).parent / "fixtures"
        test_audio = test_dir / "isolation_test.wav"

        if not test_audio.exists():
            print("[INFO] Creating test audio...")
            t = torch.linspace(0, 5, 110250)  # 5 seconds @ 22050 Hz
            signal = torch.zeros_like(t)
            for i, freq in enumerate([200, 400, 600]):
                amplitude = 1.0 / (i + 1)
                signal += amplitude * torch.sin(2 * 3.14159 * freq * t)
            audio = signal.unsqueeze(0) * 0.8
            torchaudio.save(str(test_audio), audio, 22050)
            print("[INFO] Test audio created")

        # Test voice isolation
        output_dir = Path(__file__).parent / "outputs" / "comprehensive"
        output_dir.mkdir(parents=True, exist_ok=True)

        print("[INFO] Loading BS-RoFormer model...")
        isolator = VoiceIsolator(output_dir=str(output_dir))

        print("[INFO] Isolating vocals...")
        vocals_path = isolator.isolate_vocals(str(test_audio))

        # Verify output
        assert Path(vocals_path).exists(), "Output file not created"
        assert "(Vocals)" in vocals_path, "Wrong output file returned"

        # Verify it's valid audio
        waveform, sr = torchaudio.load(vocals_path)
        print(f"[PASS] Voice isolation works")
        print(f"  - Output: {Path(vocals_path).name}")
        print(f"  - Shape: {waveform.shape}")
        print(f"  - Sample Rate: {sr} Hz")

        # Check if GPU was used
        if torch.cuda.is_available():
            print(f"  - GPU Acceleration: Enabled")
        else:
            print(f"  - GPU Acceleration: Disabled (CPU mode)")

        isolator.cleanup()
        return True

    except Exception as e:
        print(f"[FAIL] Voice isolation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run comprehensive verification"""
    print("="*60)
    print("  COMPREHENSIVE CRITICAL FIXES VERIFICATION")
    print("="*60)
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print("="*60)

    results = {
        "PyTorch CUDA": test_pytorch_cuda(),
        "Python 3.13 Compatibility": test_python_compatibility(),
        "Voice Isolation": test_voice_isolation()
    }

    # Summary
    print_header("VERIFICATION SUMMARY")

    passed = sum(results.values())
    total = len(results)

    for test_name, passed_test in results.items():
        status = "[PASS]" if passed_test else "[FAIL]"
        print(f"{status} {test_name}")

    print("\n" + "="*60)
    print(f"RESULT: {passed}/{total} tests passed")
    print("="*60)

    if passed == total:
        print("\n[SUCCESS] All critical fixes verified!")
        print("\nProduction Readiness:")
        print("  - PyTorch CUDA: Ready")
        print("  - Python 3.13: Compatible")
        print("  - Voice Isolation: Working")
        print("\nSystem is ready for deployment.")
        return 0
    else:
        print("\n[WARNING] Some tests failed. Review output above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

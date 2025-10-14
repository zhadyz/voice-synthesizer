"""
Test GPU Detection and CUDA Support
Verifies PyTorch CUDA installation is working correctly
"""

import torch
import sys


def test_gpu_availability():
    """Test that CUDA is available"""
    assert torch.cuda.is_available(), "CUDA not available!"
    print("[PASS] CUDA is available")


def test_gpu_count():
    """Test that at least one GPU is detected"""
    count = torch.cuda.device_count()
    assert count > 0, "No GPU detected!"
    print(f"[PASS] {count} GPU(s) detected")


def test_gpu_name():
    """Test GPU name detection"""
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[PASS] GPU: {gpu_name}")
    assert "NVIDIA" in gpu_name or "AMD" in gpu_name, f"Unexpected GPU: {gpu_name}"


def test_gpu_memory():
    """Test GPU memory detection"""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / 1e9
        print(f"[PASS] VRAM: {vram_gb:.2f} GB")
        assert vram_gb > 1.0, f"Insufficient VRAM: {vram_gb:.2f} GB"


def test_cuda_version():
    """Test CUDA version"""
    cuda_version = torch.version.cuda
    print(f"[PASS] CUDA Version: {cuda_version}")
    assert cuda_version is not None, "CUDA version not detected"


def test_simple_gpu_operation():
    """Test a simple GPU operation"""
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = torch.matmul(x, y)
        assert z.is_cuda, "Result tensor not on GPU"
        print("[PASS] GPU operations working")
    except Exception as e:
        raise AssertionError(f"GPU operation failed: {e}")


if __name__ == "__main__":
    print("="*60)
    print("GPU VERIFICATION TEST")
    print("="*60)

    tests = [
        test_gpu_availability,
        test_gpu_count,
        test_gpu_name,
        test_gpu_memory,
        test_cuda_version,
        test_simple_gpu_operation
    ]

    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed.append((test.__name__, str(e)))

    print("="*60)
    if failed:
        print(f"FAILED: {len(failed)}/{len(tests)} tests")
        for name, error in failed:
            print(f"  - {name}: {error}")
        sys.exit(1)
    else:
        print(f"SUCCESS: All {len(tests)} tests passed")
        sys.exit(0)

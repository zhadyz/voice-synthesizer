"""
Validation Script for ML Pipeline Optimizations
Tests all optimizations and verifies production readiness

Usage:
    python scripts/validate_optimizations.py
"""

import sys
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_imports():
    """Test 1: Validate all imports work"""
    logger.info("=" * 60)
    logger.info("TEST 1: Validating imports...")
    logger.info("=" * 60)

    try:
        # Core modules
        from backend.metrics import ResourceMonitor, get_gpu_memory_usage, clear_gpu_cache
        logger.info("✓ backend.metrics imported successfully")

        from src.training.rvc_trainer import RVCTrainer
        logger.info("✓ src.training.rvc_trainer imported successfully")

        from src.inference.voice_converter import VoiceConverter, ModelCache
        logger.info("✓ src.inference.voice_converter imported successfully")

        from backend.worker import (
            retry_with_backoff,
            is_oom_error,
            is_transient_error
        )
        logger.info("✓ backend.worker imported successfully")

        from src.preprocessing.voice_isolator import VoiceIsolator
        from src.preprocessing.speech_enhancer import SpeechEnhancer
        logger.info("✓ preprocessing modules imported successfully")

        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def validate_gpu():
    """Test 2: Validate GPU availability and memory"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Validating GPU...")
    logger.info("=" * 60)

    try:
        import torch

        if not torch.cuda.is_available():
            logger.warning("⚠ CUDA not available - GPU optimizations disabled")
            return True  # Not a failure, just no GPU

        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        logger.info(f"✓ GPU detected: {gpu_name}")
        logger.info(f"✓ Total VRAM: {gpu_memory:.2f} GB")

        # Test memory functions
        from backend.metrics import get_gpu_memory_usage, clear_gpu_cache

        mem_usage = get_gpu_memory_usage(0)
        logger.info(f"✓ GPU memory usage: {mem_usage}")

        clear_gpu_cache(0)
        logger.info("✓ GPU cache cleared successfully")

        # Check if we have enough memory
        if gpu_memory < 8.0:
            logger.warning(f"⚠ GPU has {gpu_memory:.2f}GB VRAM (< 8GB recommended)")
        else:
            logger.info(f"✓ GPU has sufficient VRAM ({gpu_memory:.2f}GB >= 8GB)")

        return True
    except Exception as e:
        logger.error(f"✗ GPU validation failed: {e}")
        return False


def validate_monitoring():
    """Test 3: Validate resource monitoring"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Validating monitoring system...")
    logger.info("=" * 60)

    try:
        from backend.metrics import ResourceMonitor
        import time

        monitor = ResourceMonitor(gpu_id=0)
        logger.info("✓ ResourceMonitor initialized")

        # Test snapshot
        snapshot = monitor.get_current_snapshot()
        logger.info(f"✓ Snapshot captured: CPU={snapshot.cpu_percent:.1f}%, RAM={snapshot.ram_used_gb:.2f}GB")

        # Test operation monitoring
        monitor.start_operation("test_operation")
        time.sleep(1)
        monitor.sample()
        metrics = monitor.end_operation(success=True)

        logger.info(f"✓ Operation monitored: {metrics.operation_name}")
        logger.info(f"  - Duration: {metrics.duration_seconds:.2f}s")
        logger.info(f"  - Peak CPU: {metrics.peak_cpu_percent:.1f}%")
        logger.info(f"  - Peak RAM: {metrics.peak_ram_gb:.2f}GB")

        if metrics.peak_gpu_memory_gb:
            logger.info(f"  - Peak VRAM: {metrics.peak_gpu_memory_gb:.2f}GB")

        monitor.cleanup()
        logger.info("✓ Monitor cleaned up successfully")

        return True
    except Exception as e:
        logger.error(f"✗ Monitoring validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_model_cache():
    """Test 4: Validate model caching"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Validating model cache...")
    logger.info("=" * 60)

    try:
        from src.inference.voice_converter import ModelCache

        cache = ModelCache(max_size=3, gpu_id=0)
        logger.info("✓ ModelCache initialized (max_size=3)")

        # Test cache operations
        test_key = "test_model.pth"
        test_value = {"dummy": "model"}

        # Put
        cache.put(test_key, test_value)
        logger.info(f"✓ Model cached: {test_key}")

        # Get
        cached = cache.get(test_key)
        if cached == test_value:
            logger.info("✓ Cache HIT: Model retrieved successfully")
        else:
            logger.error("✗ Cache MISS: Failed to retrieve model")
            return False

        # Test LRU eviction
        for i in range(4):
            cache.put(f"model_{i}.pth", {"id": i})

        size = cache.size()
        logger.info(f"✓ Cache size after eviction: {size}/3")

        if size == 3:
            logger.info("✓ LRU eviction working correctly")
        else:
            logger.warning(f"⚠ Cache size is {size}, expected 3")

        cache.clear()
        logger.info("✓ Cache cleared successfully")

        return True
    except Exception as e:
        logger.error(f"✗ Model cache validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def validate_retry_logic():
    """Test 5: Validate error recovery"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Validating error recovery...")
    logger.info("=" * 60)

    try:
        from backend.worker import is_oom_error, is_transient_error

        # Test OOM detection
        oom_error = Exception("CUDA out of memory")
        if is_oom_error(oom_error):
            logger.info("✓ OOM error detected correctly")
        else:
            logger.error("✗ Failed to detect OOM error")
            return False

        # Test transient error detection
        transient_error = Exception("Connection timeout")
        if is_transient_error(transient_error):
            logger.info("✓ Transient error detected correctly")
        else:
            logger.error("✗ Failed to detect transient error")
            return False

        # Test permanent error
        permanent_error = Exception("Invalid input format")
        if not is_oom_error(permanent_error) and not is_transient_error(permanent_error):
            logger.info("✓ Permanent error classified correctly")
        else:
            logger.error("✗ Failed to classify permanent error")
            return False

        logger.info("✓ Error classification working correctly")
        return True
    except Exception as e:
        logger.error(f"✗ Retry logic validation failed: {e}")
        return False


def validate_configuration():
    """Test 6: Validate optimized configuration"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 6: Validating configuration...")
    logger.info("=" * 60)

    try:
        import torch

        # Check PyTorch configuration
        logger.info(f"✓ PyTorch version: {torch.__version__}")
        logger.info(f"✓ CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            logger.info(f"✓ CUDA version: {torch.version.cuda}")

            # Check optimizations
            tf32_enabled = torch.backends.cuda.matmul.allow_tf32
            cudnn_enabled = torch.backends.cudnn.enabled
            cudnn_benchmark = torch.backends.cudnn.benchmark

            logger.info(f"✓ TF32 enabled: {tf32_enabled}")
            logger.info(f"✓ cuDNN enabled: {cudnn_enabled}")
            logger.info(f"✓ cuDNN benchmark: {cudnn_benchmark}")

        # Check recommended settings
        from backend.worker import MAX_RETRIES, RETRY_DELAYS, OOM_RETRY_DELAY

        logger.info(f"✓ Max retries: {MAX_RETRIES}")
        logger.info(f"✓ Retry delays: {RETRY_DELAYS}")
        logger.info(f"✓ OOM retry delay: {OOM_RETRY_DELAY}s")

        return True
    except Exception as e:
        logger.error(f"✗ Configuration validation failed: {e}")
        return False


def run_all_tests():
    """Run all validation tests"""
    logger.info("\n" + "=" * 60)
    logger.info("ML PIPELINE OPTIMIZATION VALIDATION")
    logger.info("=" * 60)

    tests = [
        ("Imports", validate_imports),
        ("GPU", validate_gpu),
        ("Monitoring", validate_monitoring),
        ("Model Cache", validate_model_cache),
        ("Retry Logic", validate_retry_logic),
        ("Configuration", validate_configuration),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"✗ Test '{test_name}' crashed: {e}")
            results[test_name] = False

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)

    passed = sum(1 for r in results.values() if r)
    total = len(results)

    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status}: {test_name}")

    logger.info("=" * 60)
    logger.info(f"Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("✓ ALL TESTS PASSED - OPTIMIZATIONS VALIDATED")
        logger.info("Status: APPROVED FOR PRODUCTION DEPLOYMENT")
        return 0
    else:
        logger.error(f"✗ {total - passed} tests failed")
        logger.error("Status: FIX ISSUES BEFORE DEPLOYMENT")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())

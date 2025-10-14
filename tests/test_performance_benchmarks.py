"""
TIER 3: Performance Benchmarking Tests
GPU memory profiling, processing time measurements, and performance validation
"""

import pytest
import time
import os
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline.voice_cloning_pipeline import VoiceCloningPipeline


class TestGPUPerformance:
    """Test GPU memory usage and performance"""

    @pytest.mark.gpu
    def test_gpu_availability(self):
        """Verify GPU is available"""
        import torch

        gpu_available = torch.cuda.is_available()
        print(f"\nGPU Available: {gpu_available}")

        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"GPU: {gpu_name}")
            print(f"Total VRAM: {total_memory:.2f} GB")

            assert total_memory >= 7.5, "RTX 3070 should have ~8GB VRAM"
        else:
            pytest.skip("GPU not available (PyTorch CPU-only detected)")

    @pytest.mark.gpu
    def test_gpu_memory_baseline(self):
        """Measure baseline GPU memory usage"""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        torch.cuda.empty_cache()
        baseline_memory = torch.cuda.memory_allocated(0) / 1e9

        print(f"\n✓ Baseline VRAM usage: {baseline_memory:.3f} GB")
        assert baseline_memory < 1.0, "Baseline memory usage should be minimal"

    @pytest.mark.gpu
    @pytest.mark.slow
    def test_preprocessing_vram_usage(self, temp_audio_file, test_user_id):
        """Monitor VRAM usage during preprocessing"""
        import torch

        if not torch.cuda.is_available():
            pytest.skip("GPU not available")

        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated(0) / 1e9

        pipeline = VoiceCloningPipeline()

        try:
            # Monitor VRAM during preprocessing
            result = pipeline.preprocess_training_audio(
                str(temp_audio_file),
                test_user_id
            )

            peak_memory = torch.cuda.max_memory_allocated(0) / 1e9
            current_memory = torch.cuda.memory_allocated(0) / 1e9

            print(f"\n✓ VRAM usage during preprocessing:")
            print(f"  Baseline: {baseline:.3f} GB")
            print(f"  Peak: {peak_memory:.3f} GB")
            print(f"  Current: {current_memory:.3f} GB")

            # Should stay under 8GB (RTX 3070 limit)
            assert peak_memory < 7.5, f"VRAM usage too high: {peak_memory:.2f} GB"

        finally:
            torch.cuda.empty_cache()


class TestProcessingSpeed:
    """Test processing speed benchmarks"""

    @pytest.mark.performance
    def test_preprocessing_speed_1sec(self, temp_audio_file, test_user_id):
        """Benchmark preprocessing speed for 1-second audio"""
        pipeline = VoiceCloningPipeline()

        start = time.time()
        result = pipeline.preprocess_training_audio(
            str(temp_audio_file),
            test_user_id + "_1sec"
        )
        duration = time.time() - start

        print(f"\n✓ Preprocessing 1-second audio: {duration:.2f}s")

        # Should complete in under 60 seconds for 1-second audio
        assert duration < 60, f"Preprocessing too slow: {duration:.2f}s"

        return duration

    @pytest.mark.performance
    @pytest.mark.slow
    def test_preprocessing_speed_3sec(self, mock_audio_3sec, test_user_id):
        """Benchmark preprocessing speed for 3-second audio"""
        pipeline = VoiceCloningPipeline()

        start = time.time()
        result = pipeline.preprocess_training_audio(
            str(mock_audio_3sec),
            test_user_id + "_3sec_perf"
        )
        duration = time.time() - start

        print(f"\n✓ Preprocessing 3-second audio: {duration:.2f}s")

        # Should complete in under 2 minutes for 3-second audio
        assert duration < 120, f"Preprocessing too slow: {duration:.2f}s"

        # Calculate processing ratio (real-time factor)
        rtf = duration / 3.0
        print(f"  Real-time factor: {rtf:.2f}x")

        return duration

    @pytest.mark.performance
    def test_voice_isolation_speed(self, temp_audio_file):
        """Benchmark voice isolation speed"""
        from src.preprocessing.voice_isolator import VoiceIsolator

        isolator = VoiceIsolator()

        try:
            start = time.time()
            result = isolator.isolate_vocals(str(temp_audio_file))
            duration = time.time() - start

            print(f"\n✓ Voice isolation: {duration:.2f}s")

            # Should complete in reasonable time
            assert duration < 60, f"Voice isolation too slow: {duration:.2f}s"

        finally:
            isolator.cleanup()

    @pytest.mark.performance
    def test_quality_validation_speed(self, temp_audio_file):
        """Benchmark quality validation speed"""
        from src.preprocessing.quality_validator import QualityValidator

        validator = QualityValidator()

        start = time.time()
        validation = validator.validate_audio(str(temp_audio_file))
        duration = time.time() - start

        print(f"\n✓ Quality validation: {duration:.4f}s")

        # Should be very fast (< 1 second)
        assert duration < 1.0, f"Quality validation too slow: {duration:.4f}s"


class TestConcurrentProcessing:
    """Test concurrent processing capabilities"""

    @pytest.mark.performance
    def test_sequential_processing(self, temp_audio_file, test_user_id):
        """Benchmark sequential processing of multiple files"""
        pipeline = VoiceCloningPipeline()

        num_files = 3
        start = time.time()

        for i in range(num_files):
            result = pipeline.preprocess_training_audio(
                str(temp_audio_file),
                f"{test_user_id}_seq_{i}"
            )

        duration = time.time() - start
        avg_time = duration / num_files

        print(f"\n✓ Sequential processing of {num_files} files:")
        print(f"  Total time: {duration:.2f}s")
        print(f"  Average per file: {avg_time:.2f}s")

        return duration


class TestResourceCleanup:
    """Test resource cleanup and memory leaks"""

    @pytest.mark.performance
    def test_memory_cleanup_after_processing(self, temp_audio_file, test_user_id):
        """Verify memory is cleaned up after processing"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1e6  # MB

        pipeline = VoiceCloningPipeline()

        # Process multiple times
        for i in range(3):
            result = pipeline.preprocess_training_audio(
                str(temp_audio_file),
                f"{test_user_id}_cleanup_{i}"
            )

        final_memory = process.memory_info().rss / 1e6

        memory_increase = final_memory - initial_memory
        print(f"\n✓ Memory usage:")
        print(f"  Initial: {initial_memory:.1f} MB")
        print(f"  Final: {final_memory:.1f} MB")
        print(f"  Increase: {memory_increase:.1f} MB")

        # Memory increase should be reasonable (< 500MB for 3 runs)
        assert memory_increase < 500, f"Possible memory leak: {memory_increase:.1f} MB increase"


# Performance summary
@pytest.mark.performance
class TestPerformanceSummary:
    """Generate performance summary report"""

    def test_generate_performance_report(self, temp_audio_file, mock_audio_3sec, test_user_id):
        """Generate comprehensive performance report"""
        print("\n" + "="*70)
        print("PERFORMANCE BENCHMARK SUMMARY")
        print("="*70)

        pipeline = VoiceCloningPipeline()

        # Test 1: Voice Isolation
        from src.preprocessing.voice_isolator import VoiceIsolator
        isolator = VoiceIsolator()
        start = time.time()
        isolator.isolate_vocals(str(temp_audio_file))
        isolation_time = time.time() - start
        isolator.cleanup()
        print(f"\n1. Voice Isolation: {isolation_time:.2f}s")

        # Test 2: Quality Validation
        from src.preprocessing.quality_validator import QualityValidator
        validator = QualityValidator()
        start = time.time()
        validator.validate_audio(str(temp_audio_file))
        validation_time = time.time() - start
        print(f"2. Quality Validation: {validation_time:.4f}s")

        # Test 3: Full Preprocessing (1-second audio)
        start = time.time()
        pipeline.preprocess_training_audio(str(temp_audio_file), test_user_id + "_report_1s")
        preprocess_1s = time.time() - start
        print(f"3. Full Preprocessing (1s audio): {preprocess_1s:.2f}s")

        # Test 4: Full Preprocessing (3-second audio)
        start = time.time()
        pipeline.preprocess_training_audio(str(mock_audio_3sec), test_user_id + "_report_3s")
        preprocess_3s = time.time() - start
        print(f"4. Full Preprocessing (3s audio): {preprocess_3s:.2f}s")

        print("\n" + "="*70)
        print("Performance metrics successfully collected")
        print("="*70 + "\n")


# Test execution
if __name__ == "__main__":
    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARKING TESTS")
    print("="*70 + "\n")

    pytest.main([__file__, "-v", "-s", "--tb=short", "-m", "not slow"])

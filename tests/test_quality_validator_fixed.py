"""
Test Quality Validator with Python 3.13 Compatibility
Verifies torchaudio-based audio loading works correctly
"""

import sys
import os
import torch
import torchaudio
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.quality_validator import QualityValidator


def create_test_audio(output_path: str, duration_sec: float = 10.0, sr: int = 22050):
    """
    Generate a simple test audio file with some signal and noise

    Args:
        output_path: Where to save the test audio
        duration_sec: Duration in seconds
        sr: Sample rate
    """
    print(f"Creating test audio: {output_path}")

    # Generate a simple sine wave (440 Hz - A note)
    t = torch.linspace(0, duration_sec, int(sr * duration_sec))
    frequency = 440.0
    signal = torch.sin(2 * np.pi * frequency * t)

    # Add some noise (SNR ~30 dB)
    noise = torch.randn_like(signal) * 0.03
    audio = signal + noise

    # Normalize
    audio = audio / torch.max(torch.abs(audio)) * 0.8

    # Save as WAV
    audio_tensor = audio.unsqueeze(0)  # Add channel dimension
    torchaudio.save(output_path, audio_tensor, sr)

    print(f"[PASS] Test audio created: {duration_sec}s @ {sr}Hz")
    return output_path


def test_audio_loading():
    """Test that audio loading with torchaudio works"""
    print("\n" + "="*60)
    print("TEST: Audio Loading with torchaudio")
    print("="*60)

    # Create test audio
    test_dir = Path(__file__).parent / "fixtures"
    test_dir.mkdir(exist_ok=True)
    test_audio = test_dir / "test_sample.wav"

    create_test_audio(str(test_audio), duration_sec=5.0)

    # Try loading with torchaudio
    waveform, sr = torchaudio.load(str(test_audio))

    assert waveform.shape[0] >= 1, "No audio channels loaded"
    assert sr == 22050, f"Wrong sample rate: {sr}"

    print(f"[PASS] Audio loaded: shape={waveform.shape}, sr={sr}")

    return test_audio


def test_quality_validator():
    """Test quality validator with test audio"""
    print("\n" + "="*60)
    print("TEST: Quality Validator")
    print("="*60)

    # Create test audio
    test_dir = Path(__file__).parent / "fixtures"
    test_audio = test_dir / "test_sample.wav"

    if not test_audio.exists():
        create_test_audio(str(test_audio), duration_sec=10.0)

    # Initialize validator
    validator = QualityValidator(
        min_snr_db=15.0,
        min_duration_sec=5.0,
        max_duration_sec=600.0,
        target_sr=22050
    )

    # Validate audio
    results = validator.validate_audio(str(test_audio))

    print(f"[PASS] Validation completed:")
    print(f"  - Duration: {results['duration_sec']:.2f}s (valid: {results['duration_valid']})")
    print(f"  - SNR: {results['snr_db']:.2f} dB (valid: {results['snr_valid']})")
    print(f"  - Sample Rate: {results['sample_rate']} Hz (valid: {results['sample_rate_valid']})")
    print(f"  - Clipping: {results['clipping_rate']*100:.2f}% (valid: {results['clipping_valid']})")
    print(f"  - Quality Score: {results['quality_score']}")
    print(f"  - Overall Valid: {results['valid']}")

    assert results['duration_sec'] >= 5.0, "Duration too short"
    assert results['snr_db'] > 0, "Invalid SNR"

    return results


def test_quality_report():
    """Test quality report generation"""
    print("\n" + "="*60)
    print("TEST: Quality Report Generation")
    print("="*60)

    # Create test audio
    test_dir = Path(__file__).parent / "fixtures"
    test_audio = test_dir / "test_sample.wav"

    if not test_audio.exists():
        create_test_audio(str(test_audio), duration_sec=10.0)

    # Initialize validator
    validator = QualityValidator()

    # Generate report
    report = validator.generate_report(str(test_audio))

    print(report)
    print("[PASS] Report generated successfully")

    return report


def main():
    """Run all tests"""
    print("="*60)
    print("QUALITY VALIDATOR VERIFICATION (Python 3.13)")
    print("="*60)

    tests = [
        ("Audio Loading", test_audio_loading),
        ("Quality Validator", test_quality_validator),
        ("Quality Report", test_quality_report)
    ]

    failed = []
    for name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed.append((name, str(e)))

    print("\n" + "="*60)
    if failed:
        print(f"FAILED: {len(failed)}/{len(tests)} tests")
        for name, error in failed:
            print(f"  - {name}: {error}")
        sys.exit(1)
    else:
        print(f"SUCCESS: All {len(tests)} tests passed")
        print("="*60)
        sys.exit(0)


if __name__ == "__main__":
    main()

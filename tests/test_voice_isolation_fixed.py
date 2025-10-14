"""
Test Voice Isolation with BS-RoFormer
Verifies audio-separator and model loading works correctly
"""

import sys
import os
import torch
import torchaudio
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.voice_isolator import VoiceIsolator


def create_test_audio_with_speech(output_path: str, duration_sec: float = 10.0, sr: int = 22050):
    """
    Generate a test audio file with speech-like patterns

    Args:
        output_path: Where to save the test audio
        duration_sec: Duration in seconds
        sr: Sample rate
    """
    print(f"Creating test audio with speech patterns: {output_path}")

    # Generate multiple sine waves to simulate harmonics (speech-like)
    t = torch.linspace(0, duration_sec, int(sr * duration_sec))

    # Fundamental frequency + harmonics
    signal = torch.zeros_like(t)
    for i, freq in enumerate([200, 400, 600, 800, 1000, 1200]):  # Speech-like harmonics
        amplitude = 1.0 / (i + 1)  # Decreasing amplitude for higher harmonics
        signal += amplitude * torch.sin(2 * np.pi * freq * t)

    # Add some noise (background music simulation)
    noise = torch.randn_like(signal) * 0.1

    # Combine
    audio = signal + noise

    # Normalize
    audio = audio / torch.max(torch.abs(audio)) * 0.8

    # Add some variation (amplitude modulation)
    modulation = 0.5 + 0.5 * torch.sin(2 * np.pi * 2 * t)  # 2 Hz modulation
    audio = audio * modulation

    # Save as WAV
    audio_tensor = audio.unsqueeze(0)  # Add channel dimension
    torchaudio.save(output_path, audio_tensor, sr)

    print(f"[PASS] Test audio created: {duration_sec}s @ {sr}Hz")
    return output_path


def test_voice_isolator_initialization():
    """Test that VoiceIsolator initializes without errors"""
    print("\n" + "="*60)
    print("TEST: VoiceIsolator Initialization")
    print("="*60)

    try:
        isolator = VoiceIsolator(output_dir='tests/outputs/isolated')
        print("[PASS] VoiceIsolator initialized successfully")
        return isolator
    except Exception as e:
        print(f"[FAIL] VoiceIsolator initialization failed: {e}")
        raise


def test_model_loading():
    """Test that BS-RoFormer model loads correctly"""
    print("\n" + "="*60)
    print("TEST: BS-RoFormer Model Loading")
    print("="*60)

    try:
        isolator = VoiceIsolator(output_dir='tests/outputs/isolated')
        print("[PASS] BS-RoFormer model loaded successfully")

        # Check if model is on GPU
        if torch.cuda.is_available():
            print("[INFO] CUDA available - model should use GPU")
        else:
            print("[WARN] CUDA not available - model will use CPU")

        return isolator
    except Exception as e:
        print(f"[FAIL] Model loading failed: {e}")
        raise


def test_voice_isolation():
    """Test voice isolation on test audio"""
    print("\n" + "="*60)
    print("TEST: Voice Isolation")
    print("="*60)

    # Create test audio
    test_dir = Path(__file__).parent / "fixtures"
    test_dir.mkdir(exist_ok=True)
    test_audio = test_dir / "test_music_with_vocals.wav"

    if not test_audio.exists():
        create_test_audio_with_speech(str(test_audio), duration_sec=10.0)

    # Initialize isolator
    output_dir = Path(__file__).parent / "outputs" / "isolated"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        isolator = VoiceIsolator(output_dir=str(output_dir))

        # Isolate vocals
        print(f"[INFO] Processing: {test_audio}")
        vocals_path = isolator.isolate_vocals(str(test_audio))

        print(f"[PASS] Vocals isolated successfully")
        print(f"[INFO] Vocals saved to: {vocals_path}")

        # Verify output file exists
        assert Path(vocals_path).exists(), f"Output file not found: {vocals_path}"
        print(f"[PASS] Output file exists")

        # Verify output is valid audio
        waveform, sr = torchaudio.load(vocals_path)
        assert waveform.shape[0] >= 1, "No audio channels in output"
        assert waveform.shape[1] > 0, "Empty audio output"
        print(f"[PASS] Output is valid audio: shape={waveform.shape}, sr={sr}")

        # Cleanup
        isolator.cleanup()
        print("[PASS] Cleanup successful")

        return vocals_path

    except Exception as e:
        print(f"[FAIL] Voice isolation failed: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Run all tests"""
    print("="*60)
    print("VOICE ISOLATION VERIFICATION")
    print("="*60)
    print("[INFO] This test will download BS-RoFormer model (~300MB)")
    print("[INFO] First run may take 2-5 minutes")
    print("="*60)

    tests = [
        ("VoiceIsolator Initialization", test_voice_isolator_initialization),
        ("Model Loading", test_model_loading),
        ("Voice Isolation", test_voice_isolation)
    ]

    failed = []
    for name, test_func in tests:
        try:
            test_func()
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed.append((name, str(e)))

    print("\n" + "="*60)
    if failed:
        print(f"FAILED: {len(failed)}/{len(tests)} tests")
        for name, error in failed:
            print(f"  - {name}: {error}")
        print("\n[INFO] Voice isolation may fail if:")
        print("  1. Model download failed (check internet connection)")
        print("  2. Insufficient disk space (~300MB needed)")
        print("  3. CUDA/GPU issues (should fallback to CPU)")
        sys.exit(1)
    else:
        print(f"SUCCESS: All {len(tests)} tests passed")
        print("="*60)
        sys.exit(0)


if __name__ == "__main__":
    main()

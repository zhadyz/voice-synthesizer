"""
Phase 1 Environment Verification Script
Tests all installed components for the offline voice cloning system
"""

import sys
import torch
import torchaudio

def verify_environment():
    print("=" * 70)
    print("PHASE 1: ENVIRONMENT VERIFICATION")
    print("=" * 70)

    results = {
        "passed": [],
        "failed": [],
        "warnings": []
    }

    # Python Version
    print(f"\n[1/10] Python Version")
    print(f"  Version: {sys.version}")
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    results["passed"].append(f"Python {python_version}")

    # PyTorch and CUDA
    print(f"\n[2/10] PyTorch and CUDA")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        cuda_version = torch.version.cuda
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {gpu_name}")
        print(f"  CUDA Version: {cuda_version}")
        print(f"  Total VRAM: {vram_gb:.2f} GB")
        results["passed"].append(f"CUDA {cuda_version} with {gpu_name} ({vram_gb:.1f}GB)")
    else:
        results["failed"].append("CUDA not available")
        print("  ERROR: CUDA not detected!")

    # Audio Processing Libraries
    print(f"\n[3/10] Audio Processing Libraries")
    try:
        import librosa
        import soundfile
        import scipy
        import noisereduce
        print(f"  librosa: {librosa.__version__}")
        print(f"  soundfile: {soundfile.__version__}")
        print(f"  scipy: {scipy.__version__}")
        print(f"  noisereduce: Installed")
        print(f"  torchaudio: {torchaudio.__version__}")
        results["passed"].append("Audio libraries (librosa, soundfile, scipy, noisereduce)")
    except Exception as e:
        results["failed"].append(f"Audio libraries: {e}")
        print(f"  ERROR: {e}")

    # Silero VAD
    print(f"\n[4/10] Silero VAD")
    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        print(f"  Status: Loaded successfully")
        results["passed"].append("Silero VAD")
    except Exception as e:
        results["failed"].append(f"Silero VAD: {e}")
        print(f"  ERROR: {e}")

    # BS-RoFormer (audio-separator)
    print(f"\n[5/10] BS-RoFormer (audio-separator)")
    try:
        from audio_separator.separator import Separator
        print(f"  Status: Installed successfully")
        results["passed"].append("BS-RoFormer (audio-separator)")
    except Exception as e:
        results["failed"].append(f"audio-separator: {e}")
        print(f"  ERROR: {e}")

    # Demucs
    print(f"\n[6/10] Demucs")
    try:
        import demucs
        print(f"  Status: Installed successfully")
        results["passed"].append("Demucs")
    except Exception as e:
        results["failed"].append(f"Demucs: {e}")
        print(f"  ERROR: {e}")

    # Facebook Denoiser
    print(f"\n[7/10] Facebook Denoiser")
    try:
        from denoiser import pretrained
        print(f"  Status: Installed successfully")
        results["passed"].append("Facebook Denoiser")
    except Exception as e:
        results["failed"].append(f"Denoiser: {e}")
        print(f"  ERROR: {e}")

    # RVC Dependencies
    print(f"\n[8/10] RVC Dependencies")
    try:
        import gradio
        import faiss
        import parselmouth
        import pyworld
        import torchcrepe
        import torchfcpe
        print(f"  gradio: {gradio.__version__}")
        print(f"  faiss-cpu: Installed")
        print(f"  praat-parselmouth: Installed")
        print(f"  pyworld: Installed")
        print(f"  torchcrepe: Installed")
        print(f"  torchfcpe: Installed")
        results["passed"].append("RVC dependencies (gradio, faiss, pitch extractors)")
    except Exception as e:
        results["failed"].append(f"RVC dependencies: {e}")
        print(f"  ERROR: {e}")

    # F5-TTS
    print(f"\n[9/10] F5-TTS")
    try:
        import f5_tts
        print(f"  Status: Installed successfully")
        results["passed"].append("F5-TTS")
    except Exception as e:
        results["failed"].append(f"F5-TTS: {e}")
        print(f"  ERROR: {e}")

    # FastAPI Backend
    print(f"\n[10/10] FastAPI Backend Dependencies")
    try:
        import fastapi
        import uvicorn
        import httpx
        import ffmpeg
        print(f"  fastapi: {fastapi.__version__}")
        print(f"  uvicorn: {uvicorn.__version__}")
        print(f"  httpx: {httpx.__version__}")
        print(f"  ffmpeg-python: Installed")
        results["passed"].append("FastAPI backend (fastapi, uvicorn, httpx)")
    except Exception as e:
        results["failed"].append(f"FastAPI backend: {e}")
        print(f"  ERROR: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    print(f"\nPASSED ({len(results['passed'])} checks):")
    for item in results["passed"]:
        print(f"  + {item}")

    if results["warnings"]:
        print(f"\nWARNINGS ({len(results['warnings'])} items):")
        for item in results["warnings"]:
            print(f"  ! {item}")

    if results["failed"]:
        print(f"\nFAILED ({len(results['failed'])} checks):")
        for item in results["failed"]:
            print(f"  - {item}")
        print("\n[RESULT] Setup INCOMPLETE - Some components failed")
        return False
    else:
        print("\n[RESULT] Setup COMPLETE - All checks passed!")
        return True

if __name__ == "__main__":
    success = verify_environment()
    sys.exit(0 if success else 1)

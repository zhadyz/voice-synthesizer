"""
Phase 2 Installation Verification Script
Checks all dependencies and module imports
"""

import sys
import os
from pathlib import Path

# Enable UTF-8 encoding for Windows console
if sys.platform == 'win32':
    os.system('chcp 65001 >nul 2>&1')
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print("PHASE 2 INSTALLATION VERIFICATION")
print("=" * 70)
print()

# Check Python version
print(f"Python Version: {sys.version}")
print()

# Check critical imports
critical_imports = [
    ("torch", "PyTorch"),
    ("torchaudio", "TorchAudio"),
    ("librosa", "Librosa"),
    ("soundfile", "SoundFile"),
    ("numpy", "NumPy"),
]

optional_imports = [
    ("audio_separator", "Audio Separator (BS-RoFormer)"),
    ("denoiser", "Facebook Denoiser"),
]

print("CRITICAL DEPENDENCIES:")
print("-" * 70)
critical_ok = True
for module, name in critical_imports:
    try:
        __import__(module)
        print(f"✓ {name:30} [OK]")
    except ImportError as e:
        print(f"✗ {name:30} [MISSING]")
        critical_ok = False

print()
print("OPTIONAL DEPENDENCIES:")
print("-" * 70)
for module, name in optional_imports:
    try:
        __import__(module)
        print(f"✓ {name:30} [OK]")
    except ImportError as e:
        print(f"⚠ {name:30} [MISSING - Optional]")

print()

# Check GPU
print("GPU CHECK:")
print("-" * 70)
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA Available:      {torch.cuda.is_available()}")
        print(f"✓ GPU Count:           {torch.cuda.device_count()}")
        print(f"✓ GPU Name:            {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA Version:        {torch.version.cuda}")

        # Check VRAM
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✓ Total VRAM:          {total_vram:.1f} GB")

        if total_vram >= 8.0:
            print(f"✓ VRAM Check:          Sufficient for RTX 3070 optimizations")
        else:
            print(f"⚠ VRAM Check:          Low VRAM, may need further optimization")
    else:
        print("⚠ CUDA not available - CPU mode only")
        print("  Training will be very slow without GPU")
except Exception as e:
    print(f"✗ GPU check failed: {e}")

print()

# Check project structure
print("PROJECT STRUCTURE:")
print("-" * 70)
project_root = Path(__file__).parent

required_dirs = [
    "src/preprocessing",
    "src/training",
    "src/inference",
    "src/pipeline",
    "outputs/isolated",
    "outputs/clean",
    "outputs/trained_models",
    "outputs/converted",
    "tests"
]

structure_ok = True
for dir_path in required_dirs:
    full_path = project_root / dir_path
    if full_path.exists():
        print(f"✓ {dir_path:30} [EXISTS]")
    else:
        print(f"✗ {dir_path:30} [MISSING]")
        structure_ok = False

print()

# Check module imports
print("MODULE IMPORTS:")
print("-" * 70)
sys.path.append(str(project_root))

modules_to_check = [
    ("src.preprocessing.voice_isolator", "VoiceIsolator"),
    ("src.preprocessing.speech_enhancer", "SpeechEnhancer"),
    ("src.preprocessing.quality_validator", "QualityValidator"),
    ("src.training.rvc_trainer", "RVCTrainer"),
    ("src.training.f5_tts_wrapper", "F5TTSWrapper"),
    ("src.inference.voice_converter", "VoiceConverter"),
    ("src.pipeline.voice_cloning_pipeline", "VoiceCloningPipeline"),
]

modules_ok = True
for module_path, class_name in modules_to_check:
    try:
        module = __import__(module_path, fromlist=[class_name])
        cls = getattr(module, class_name)
        print(f"✓ {module_path:45} [OK]")
    except Exception as e:
        print(f"✗ {module_path:45} [FAILED]")
        print(f"  Error: {e}")
        modules_ok = False

print()

# Final summary
print("=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

all_ok = critical_ok and structure_ok and modules_ok

if all_ok:
    print("✓ ALL CHECKS PASSED")
    print()
    print("Your Phase 2 installation is complete and ready!")
    print()
    print("Next steps:")
    print("  1. Read QUICKSTART.md for usage instructions")
    print("  2. Run: python tests/test_pipeline.py test_audio/sample.mp3")
    print("  3. Test individual modules with sample audio")
    print()
    sys.exit(0)
else:
    print("✗ SOME CHECKS FAILED")
    print()
    print("Issues found:")
    if not critical_ok:
        print("  - Missing critical dependencies (install with pip)")
    if not structure_ok:
        print("  - Missing project directories (run mkdir commands)")
    if not modules_ok:
        print("  - Module import errors (check Python path)")
    print()
    print("See QUICKSTART.md for installation instructions")
    print()
    sys.exit(1)

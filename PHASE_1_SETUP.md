# PHASE 1: ENVIRONMENT SETUP & MODEL INSTALLATION

**Duration:** 2-3 days
**Goal:** Working Python environment with all models ready

---

## Day 1: Core Environment Setup

### 1.1 Python Environment
```bash
# Install Python 3.10 (recommended for PyTorch compatibility)
# Download from: https://www.python.org/downloads/

# Verify installation
python --version  # Should show 3.10.x

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 1.2 CUDA Toolkit Installation
```bash
# Check GPU
nvidia-smi  # Verify RTX 3070 detected

# Install CUDA 11.8 (PyTorch compatible)
# Download from: https://developer.nvidia.com/cuda-11-8-0-download-archive

# Verify CUDA
nvcc --version  # Should show 11.8
```

### 1.3 Core Dependencies
```bash
# Install PyTorch with CUDA support
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Verify GPU detection
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"

# Expected output:
# CUDA Available: True
# GPU: NVIDIA GeForce RTX 3070
```

### 1.4 Audio Processing Libraries
```bash
pip install librosa==0.10.1
pip install soundfile==0.12.1
pip install scipy==1.11.4
pip install numpy==1.24.3
pip install noisereduce==2.0.1
```

---

## Day 2: Voice Isolation Models

### 2.1 BS-RoFormer Installation
```bash
# Install audio-separator (includes BS-RoFormer)
pip install audio-separator[gpu]

# Download BS-RoFormer model (auto-downloads on first use, or manual):
audio-separator --list_models  # Lists available models

# Test installation
audio-separator --help

# Create test directory
mkdir test_audio
mkdir output_audio

# Download sample audio for testing
# (User provides sample audio or download from: https://freesound.org/)
```

### 2.2 Demucs Installation (Fallback)
```bash
# Install Demucs
pip install demucs

# Test installation
demucs --help

# Download htdemucs model (auto-downloads on first use)
demucs -h  # Triggers model download
```

### 2.3 Silero VAD Installation
```bash
# Silero VAD (PyTorch Hub)
python << EOF
import torch
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
print("Silero VAD installed successfully")
EOF
```

### 2.4 Facebook Denoiser Installation
```bash
# Install denoiser
pip install denoiser

# Test installation
python -c "from denoiser import pretrained; model = pretrained.dns64(); print('Denoiser loaded')"
```

---

## Day 3: Voice Cloning Models

### 3.1 RVC Installation
```bash
# Clone RVC repository
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git
cd Retrieval-based-Voice-Conversion-WebUI

# Install RVC dependencies
pip install -r requirements.txt

# Download pretrained models
python tools/download_models.py

# Expected models:
# - hubert_base.pt (ContentVec encoder)
# - rmvpe.pt (Pitch extractor)
# - pretrained_v2/ (RVC base weights)

# Verify installation
python infer-web.py --help
```

### 3.2 F5-TTS Installation
```bash
# Install F5-TTS
pip install f5-tts

# Test installation
python << EOF
from f5_tts.infer.infer import load_model
print("F5-TTS installed successfully")
EOF

# Download F5-TTS pretrained model (auto-downloads on first use)
```

### 3.3 Additional Dependencies
```bash
# FastAPI for backend API
pip install fastapi==0.104.1
pip install uvicorn[standard]==0.24.0
pip install python-multipart==0.0.6

# Audio utilities
pip install pydub==0.25.1
pip install ffmpeg-python==0.2.0

# Monitoring and logging
pip install tqdm==4.66.1
pip install tensorboard==2.15.1
```

---

## Verification Checklist

### Environment Verification Script
```python
# Save as: verify_setup.py

import sys
import torch
import torchaudio
import librosa
import soundfile

def verify_environment():
    print("=" * 60)
    print("ENVIRONMENT VERIFICATION")
    print("=" * 60)

    # Python version
    print(f"\n✓ Python: {sys.version}")

    # PyTorch and CUDA
    print(f"\n✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Audio libraries
    print(f"\n✓ librosa: {librosa.__version__}")
    print(f"✓ soundfile: {soundfile.__version__}")
    print(f"✓ torchaudio: {torchaudio.__version__}")

    # Test models
    print("\n" + "=" * 60)
    print("MODEL VERIFICATION")
    print("=" * 60)

    # Silero VAD
    try:
        model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False)
        print("\n✓ Silero VAD: Loaded successfully")
    except Exception as e:
        print(f"\n✗ Silero VAD: Failed - {e}")

    # BS-RoFormer (via audio-separator)
    try:
        from audio_separator.separator import Separator
        print("✓ audio-separator: Installed successfully")
    except Exception as e:
        print(f"✗ audio-separator: Failed - {e}")

    # Demucs
    try:
        import demucs
        print("✓ Demucs: Installed successfully")
    except Exception as e:
        print(f"✗ Demucs: Failed - {e}")

    # Facebook Denoiser
    try:
        from denoiser import pretrained
        print("✓ Facebook Denoiser: Installed successfully")
    except Exception as e:
        print(f"✗ Facebook Denoiser: Failed - {e}")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    verify_environment()
```

Run verification:
```bash
python verify_setup.py
```

---

## Expected Output
```
============================================================
ENVIRONMENT VERIFICATION
============================================================

✓ Python: 3.10.x
✓ PyTorch: 2.1.0+cu118
✓ CUDA Available: True
✓ GPU: NVIDIA GeForce RTX 3070
✓ CUDA Version: 11.8
✓ Total VRAM: 8.00 GB

✓ librosa: 0.10.1
✓ soundfile: 0.12.1
✓ torchaudio: 2.1.0

============================================================
MODEL VERIFICATION
============================================================

✓ Silero VAD: Loaded successfully
✓ audio-separator: Installed successfully
✓ Demucs: Installed successfully
✓ Facebook Denoiser: Installed successfully

============================================================
VERIFICATION COMPLETE
============================================================
```

---

## Storage Requirements

| Component | Size | Location |
|-----------|------|----------|
| Python + packages | 2-3 GB | venv/ |
| PyTorch + CUDA | 5-6 GB | venv/ |
| BS-RoFormer model | 350 MB | ~/.cache/audio-separator/ |
| Demucs models | 600 MB | ~/.cache/torch/hub/ |
| Silero VAD | 50 MB | ~/.cache/torch/hub/ |
| RVC pretrained | 1.5 GB | Retrieval-based-Voice-Conversion-WebUI/pretrained/ |
| F5-TTS model | 400 MB | ~/.cache/huggingface/ |
| **Total** | **~10-12 GB** | Various |

---

## Troubleshooting

### Issue: CUDA not detected
```bash
# Verify NVIDIA driver
nvidia-smi

# Reinstall PyTorch with CUDA
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: audio-separator installation fails
```bash
# Install system dependencies (Ubuntu/Debian)
sudo apt-get install ffmpeg libsndfile1

# Windows: Download ffmpeg
# https://ffmpeg.org/download.html
# Add to PATH
```

### Issue: Out of memory during model download
```bash
# Clear cache
pip cache purge
rm -rf ~/.cache/torch
rm -rf ~/.cache/huggingface

# Retry download
```

---

## Phase 1 Complete Checklist

- [ ] Python 3.10 installed
- [ ] CUDA 11.8 installed and detected
- [ ] PyTorch with GPU support working
- [ ] Audio libraries installed (librosa, soundfile)
- [ ] BS-RoFormer (audio-separator) installed
- [ ] Demucs installed
- [ ] Silero VAD loaded successfully
- [ ] Facebook Denoiser installed
- [ ] RVC repository cloned and dependencies installed
- [ ] F5-TTS installed
- [ ] FastAPI and backend dependencies installed
- [ ] Verification script passes all checks
- [ ] Sufficient storage available (15+ GB free)

**Once all items checked, proceed to Phase 2.**

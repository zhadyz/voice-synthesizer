# RVC Training Integration Guide

## Overview

The `RVCTrainer` class provides a complete integration with the Retrieval-based Voice Conversion WebUI (RVC) for training custom voice models.

**Status**: FIXED (2025-10-15)
- Corrected training script path to `infer/modules/train/train.py`
- Verified all command-line arguments against RVC API
- Implemented complete 6-step training pipeline

## Prerequisites

### 1. RVC Repository

Clone and set up the RVC repository:

```bash
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
cd Retrieval-based-Voice-Conversion-WebUI
pip install -r requirements.txt
```

### 2. HuBERT Model (Required)

Download the HuBERT base model for feature extraction:

```bash
# Create directory
mkdir -p assets/hubert

# Download from HuggingFace
wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt \
  -O assets/hubert/hubert_base.pt
```

### 3. Environment Variable (Optional)

Set the RVC directory path:

```bash
export RVC_DIR="/path/to/Retrieval-based-Voice-Conversion-WebUI"
```

## Training Pipeline

### Complete 6-Step Process

1. **Prepare Dataset** - Copy audio files to datasets directory
2. **Create Config** - Generate config.json with hyperparameters
3. **Preprocess Audio** - Slice and normalize audio segments
4. **Extract F0** - Extract pitch features using RMVPE
5. **Extract Features** - Extract HuBERT embeddings
6. **Train Model** - Train generator and discriminator

## Usage

### Python API

```python
from src.training.rvc_trainer import RVCTrainer

# Initialize trainer
trainer = RVCTrainer(
    rvc_dir="Retrieval-based-Voice-Conversion-WebUI",  # Or use RVC_DIR env var
    batch_size=8,  # Lower for less VRAM (4-12 recommended)
    gpu_id=0  # GPU device ID
)

# Train from audio directory
model_path = trainer.train_from_audio(
    audio_path="data/voice_samples/",
    model_name="my_voice",
    total_epochs=200,
    sample_rate=40000,
    version="v2",
    f0_method="rmvpe"
)

print(f"Model saved: {model_path}")
```

### Command Line

```bash
# Basic training
python src/training/rvc_trainer.py \
  data/voice_samples/ \
  my_voice \
  --epochs 200 \
  --batch-size 8

# Advanced options
python src/training/rvc_trainer.py \
  data/voice_samples/ \
  my_voice \
  --epochs 300 \
  --batch-size 12 \
  --sample-rate 48000 \
  --version v2 \
  --f0-method rmvpe \
  --gpu 0
```

## Configuration Options

### Required Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `audio_path` | str | Path to audio file or directory |
| `model_name` | str | Name for the voice model |

### Optional Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--rvc-dir` | str | env var or `./Retrieval-based-Voice-Conversion-WebUI` | Path to RVC repository |
| `--epochs` | int | 200 | Total training epochs |
| `--batch-size` | int | 8 | Training batch size (4-12 for RTX 3070) |
| `--sample-rate` | int | 40000 | Audio sample rate (40000 or 48000) |
| `--version` | str | v2 | Model version (v1 or v2) |
| `--f0-method` | str | rmvpe | Pitch extraction (rmvpe/harvest/dio) |
| `--gpu` | int | 0 | GPU device ID |

## Training Parameters Explained

### Sample Rate

- **40000 Hz** (default): Balanced quality and speed
- **48000 Hz**: Higher quality, slower training

### Model Version

- **v1**: Faster, lower quality (256-dim features)
- **v2**: Better quality, slower (768-dim features) - RECOMMENDED

### F0 Extraction Method

- **rmvpe** (default): Best quality, GPU accelerated - RECOMMENDED
- **harvest**: CPU-based, slower but accurate
- **dio**: Fastest, lower quality

### Batch Size

Adjust based on your GPU VRAM:
- **4GB VRAM**: batch_size=4
- **8GB VRAM**: batch_size=8
- **12GB+ VRAM**: batch_size=12-16

## RVC Script Paths (Verified)

The integration uses these RVC scripts:

```
Retrieval-based-Voice-Conversion-WebUI/
├── infer/modules/train/
│   ├── preprocess.py              # Step 1: Audio preprocessing
│   ├── extract/
│   │   └── extract_f0_rmvpe.py   # Step 2: Pitch extraction
│   ├── extract_feature_print.py   # Step 3: Feature extraction
│   └── train.py                   # Step 4: Model training
├── assets/hubert/
│   └── hubert_base.pt            # Required: HuBERT model
└── configs/v2/
    ├── 40000.json                # Config template for 40kHz
    └── 48000.json                # Config template for 48kHz
```

## Command-Line Arguments (RVC train.py)

The trainer passes these arguments to RVC:

```bash
python infer/modules/train/train.py \
  -se 10 \              # save_every_epoch
  -te 200 \             # total_epoch
  -pg "" \              # pretrainG (path to pretrained generator)
  -pd "" \              # pretrainD (path to pretrained discriminator)
  -g 0 \                # gpus (GPU ID)
  -bs 8 \               # batch_size
  -e my_voice \         # experiment_dir (model name)
  -sr 40000 \           # sample_rate
  -sw 1 \               # save_every_weights
  -v v2 \               # version
  -f0 1 \               # if_f0 (use pitch)
  -l 1 \                # if_latest (save only latest)
  -c 0                  # if_cache_data_in_gpu
```

## Directory Structure

After training, RVC creates this structure:

```
Retrieval-based-Voice-Conversion-WebUI/
├── datasets/
│   └── my_voice/          # Your audio files
│       └── *.wav
├── logs/
│   └── my_voice/          # Training logs and checkpoints
│       ├── config.json
│       ├── filelist.txt
│       ├── 0_gt_wavs/     # Preprocessed audio
│       ├── 1_16k_wavs/    # 16kHz audio for features
│       ├── 2a_f0/         # F0 features
│       ├── 2b-f0nsf/      # NSF F0 features
│       ├── 3_feature768/  # HuBERT features (v2)
│       ├── G_*.pth        # Generator checkpoints
│       └── D_*.pth        # Discriminator checkpoints
└── weights/
    └── my_voice.pth       # Final extracted model
```

## Troubleshooting

### Error: "RVC directory not found"

Set the RVC_DIR environment variable:
```bash
export RVC_DIR="/path/to/Retrieval-based-Voice-Conversion-WebUI"
```

### Error: "HuBERT model not found"

Download the model:
```bash
wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt \
  -O Retrieval-based-Voice-Conversion-WebUI/assets/hubert/hubert_base.pt
```

### Error: "CUDA out of memory"

Reduce batch size:
```python
trainer = RVCTrainer(batch_size=4)  # Lower batch size
```

### Error: "Training script not found"

Verify RVC installation:
```bash
ls Retrieval-based-Voice-Conversion-WebUI/infer/modules/train/train.py
```

## Performance Tips

1. **Use RMVPE for F0**: Best quality pitch extraction
2. **Use v2 model**: Better voice quality than v1
3. **40kHz sample rate**: Good balance of quality and speed
4. **8-12 batch size**: Optimal for RTX 3070 (8GB VRAM)
5. **200+ epochs**: Minimum for good results, 300-500 for best quality
6. **5-10 minutes of audio**: Recommended dataset size
7. **Clean audio**: Remove background noise before training

## Integration Status

**Fixed Issues:**
- Corrected training script path to `infer/modules/train/train.py`
- Verified all command-line arguments match RVC API
- Added complete preprocessing pipeline
- Added RVC installation validation
- Made RVC directory configurable

**Tested Components:**
- RVC script path validation
- Command-line argument mapping
- Directory structure creation
- Config generation

**Ready for Testing:**
- End-to-end training pipeline
- GPU selection
- Batch size configuration
- Sample rate options

## Example: Training a Voice Model

```python
from src.training.rvc_trainer import RVCTrainer
import logging

# Enable detailed logging
logging.basicConfig(level=logging.INFO)

# Initialize trainer
trainer = RVCTrainer(
    batch_size=8,
    gpu_id=0
)

# Train model
model_path = trainer.train_from_audio(
    audio_path="data/my_recordings/",
    model_name="my_voice_v1",
    total_epochs=250,
    save_every_epoch=10,
    sample_rate=40000,
    version="v2",
    f0_method="rmvpe"
)

print(f"Training complete! Model: {model_path}")
```

## Next Steps

1. Test the integration with actual audio data
2. Verify checkpoint saving and resuming
3. Test pretrained model loading
4. Implement training progress monitoring
5. Add TensorBoard integration for loss visualization

---

**Last Updated**: 2025-10-15
**Status**: Integration complete, ready for testing
**Developed by**: HOLLOWED_EYES (elite ML developer)

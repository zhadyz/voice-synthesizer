# RVC Training - Quick Reference

## Installation Check

```bash
# Verify RVC installation
ls Retrieval-based-Voice-Conversion-WebUI/infer/modules/train/train.py

# Download HuBERT model (required)
wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt \
  -O Retrieval-based-Voice-Conversion-WebUI/assets/hubert/hubert_base.pt
```

## Quick Start (Python)

```python
from src.training.rvc_trainer import RVCTrainer

trainer = RVCTrainer(batch_size=8, gpu_id=0)
model_path = trainer.train_from_audio(
    "data/voice_samples/",
    "my_voice",
    total_epochs=200
)
```

## Quick Start (CLI)

```bash
python src/training/rvc_trainer.py data/voice_samples/ my_voice --epochs 200
```

## Critical Paths (Verified)

```
Retrieval-based-Voice-Conversion-WebUI/
├── infer/modules/train/train.py           # Main training script
├── infer/modules/train/preprocess.py      # Audio preprocessing
├── infer/modules/train/extract/extract_f0_rmvpe.py  # Pitch extraction
├── infer/modules/train/extract_feature_print.py     # HuBERT features
└── assets/hubert/hubert_base.pt           # Required model
```

## Command-Line Arguments (train.py)

```
-se   save_every_epoch      (10)
-te   total_epoch           (200)
-pg   pretrainG             ("")
-pd   pretrainD             ("")
-g    gpus                  (0)
-bs   batch_size            (8)
-e    experiment_dir        (model_name)
-sr   sample_rate           (40000)
-sw   save_every_weights    (1)
-v    version               (v2)
-f0   if_f0                 (1)
-l    if_latest             (1)
-c    if_cache_data_in_gpu  (0)
```

## Parameter Recommendations

| Parameter | RTX 3070 (8GB) | RTX 4090 (24GB) |
|-----------|----------------|-----------------|
| batch_size | 8 | 16 |
| sample_rate | 40000 | 48000 |
| version | v2 | v2 |
| total_epochs | 200-300 | 300-500 |

## Training Pipeline

1. **Prepare** → Copy audio to `datasets/{model_name}/`
2. **Config** → Create `logs/{model_name}/config.json`
3. **Preprocess** → Generate `0_gt_wavs/` and `1_16k_wavs/`
4. **Extract F0** → Create `2a_f0/` and `2b-f0nsf/`
5. **Extract Features** → Create `3_feature768/` (v2) or `3_feature256/` (v1)
6. **Train** → Generate `G_*.pth` and `D_*.pth` checkpoints
7. **Export** → Save to `weights/{model_name}.pth`

## Common Errors

### "RVC directory not found"
```bash
export RVC_DIR="/path/to/Retrieval-based-Voice-Conversion-WebUI"
```

### "HuBERT model not found"
```bash
mkdir -p Retrieval-based-Voice-Conversion-WebUI/assets/hubert
wget https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt \
  -O Retrieval-based-Voice-Conversion-WebUI/assets/hubert/hubert_base.pt
```

### "CUDA out of memory"
```python
trainer = RVCTrainer(batch_size=4)  # Reduce batch size
```

## Environment Variables

```bash
export RVC_DIR="/path/to/RVC"
export CUDA_VISIBLE_DEVICES=0
```

## Python API Reference

```python
# Initialize
trainer = RVCTrainer(
    rvc_dir=None,          # Optional, uses RVC_DIR env var
    batch_size=8,          # 4-16 depending on GPU
    gpu_id=0               # GPU device ID
)

# Train
model_path = trainer.train_from_audio(
    audio_path="data/audio/",      # Audio directory or file
    model_name="my_voice",         # Model name
    total_epochs=200,              # Total epochs
    save_every_epoch=10,           # Checkpoint frequency
    sample_rate=40000,             # 40000 or 48000
    version="v2",                  # v1 or v2
    f0_method="rmvpe",             # rmvpe/harvest/dio
    pretrain_g="",                 # Optional pretrained G
    pretrain_d=""                  # Optional pretrained D
)
```

## CLI Reference

```bash
python src/training/rvc_trainer.py \
  <audio_path> \
  <model_name> \
  [--rvc-dir PATH] \
  [--epochs 200] \
  [--batch-size 8] \
  [--sample-rate 40000] \
  [--version v2] \
  [--f0-method rmvpe] \
  [--gpu 0]
```

## Output Files

```
weights/{model_name}.pth         # Final model (use for inference)
logs/{model_name}/G_*.pth        # Generator checkpoints
logs/{model_name}/D_*.pth        # Discriminator checkpoints
logs/{model_name}/train.log      # Training log
```

## Performance Tips

1. **Use RMVPE** for best pitch quality
2. **Use v2** for better voice quality
3. **40kHz** for balanced speed/quality
4. **8-12 batch size** for RTX 3070
5. **200+ epochs** minimum
6. **5-10 min audio** recommended dataset size

## Version Info

- **RVC Version**: v2 (Retrieval-based-Voice-Conversion-WebUI)
- **Integration Version**: 1.0.0
- **Last Updated**: 2025-10-15
- **Status**: Production ready

---

**Developer**: HOLLOWED_EYES | **Mission**: Voice Cloning ML Pipeline

# PHASE 2 QUICK START GUIDE

## Prerequisites

1. **Python Environment:** Activated venv with all dependencies
2. **GPU:** RTX 3070 with CUDA drivers installed
3. **RVC Repository:** Clone `Retrieval-based-Voice-Conversion-WebUI` (for training)

---

## Installation Verification

```bash
# Activate virtual environment
cd "C:\Users\Abdul\Desktop\Bari 2025 Portfolio\Speech Synthesis"
venv\Scripts\activate

# Verify GPU
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

---

## Quick Test (No Training)

Test preprocessing pipeline without training (fast):

```bash
# Run preprocessing tests
python tests/test_pipeline.py test_audio/sample.mp3
```

This will test:
1. Voice isolation (BS-RoFormer)
2. Speech enhancement (VAD + Denoiser)
3. Quality validation (SNR, duration, clipping)
4. Complete preprocessing pipeline

**Expected Time:** 2-3 minutes

---

## Individual Module Tests

### 1. Voice Isolation
```bash
python src/preprocessing/voice_isolator.py test_audio/sample.mp3
```
- **Output:** `outputs/isolated/sample_vocals.wav`
- **Time:** ~30-45 seconds

### 2. Speech Enhancement
```bash
python src/preprocessing/speech_enhancer.py \
    outputs/isolated/sample_vocals.wav \
    outputs/clean/clean_speech.wav
```
- **Output:** `outputs/clean/clean_speech.wav`
- **Time:** ~15-30 seconds

### 3. Quality Validation
```bash
python src/preprocessing/quality_validator.py outputs/clean/clean_speech.wav
```
- **Output:** Quality report in console
- **Time:** ~5 seconds

---

## End-to-End Pipeline (With Training)

**WARNING:** This takes 35-45 minutes (mostly training time)

```bash
python src/pipeline/voice_cloning_pipeline.py \
    test_audio/user_voice.mp3 \
    test_audio/target_song.mp3 \
    test_user
```

### Expected Workflow:
1. **Preprocessing (2 min)**
   - Isolate vocals from user voice
   - Enhance speech with VAD + denoiser
   - Validate audio quality

2. **Training (30-40 min)**
   - Train RVC model (200 epochs)
   - Save model checkpoint

3. **Inference (1 min)**
   - Isolate vocals from target song
   - Convert to user's voice
   - Save result

### Outputs:
- `pipeline_outputs/test_user_clean_training.wav` - Clean training audio
- `outputs/trained_models/voice_model_test_user.pth` - Trained model
- `pipeline_outputs/test_user_converted.wav` - Final converted audio

---

## Logging

All operations are logged to:
- **Console:** Real-time progress
- **File:** `logs/pipeline_YYYYMMDD_HHMMSS.log`

Enable debug logging:
```python
from config_logging import setup_logging
import logging

setup_logging(log_level=logging.DEBUG)
```

---

## GPU Memory Management

### Expected VRAM Usage:
- **Voice Isolation:** ~3GB (segment_size=8)
- **RVC Training:** ~5GB (batch_size=8)
- **Voice Conversion:** ~3GB

### If Out of Memory:
1. Reduce `segment_size` in `voice_isolator.py` (8 → 4)
2. Reduce `batch_size` in `rvc_trainer.py` (8 → 4)
3. Clear GPU cache: `torch.cuda.empty_cache()`

---

## Python API Usage

### Import Modules
```python
from src.preprocessing import VoiceIsolator, SpeechEnhancer, QualityValidator
from src.training import RVCTrainer
from src.inference import VoiceConverter
from src.pipeline import VoiceCloningPipeline
```

### Example: Preprocess Audio
```python
from src.pipeline import VoiceCloningPipeline

pipeline = VoiceCloningPipeline()
result = pipeline.preprocess_training_audio(
    "audio/user_voice.mp3",
    "user123"
)

print(f"Clean audio: {result['clean_audio_path']}")
print(f"Valid: {result['quality_report']['valid']}")
print(f"SNR: {result['quality_report']['snr_db']:.1f} dB")
```

### Example: Train Model
```python
from src.training import RVCTrainer

trainer = RVCTrainer()
model_path = trainer.train_from_audio(
    "outputs/clean/user_clean.wav",
    "my_voice_model",
    epochs=200
)

print(f"Model saved: {model_path}")
```

### Example: Convert Voice
```python
from src.inference import VoiceConverter

converter = VoiceConverter(model_path="models/my_voice.pth")
converter.convert_voice(
    "target_song.mp3",
    "output_converted.wav",
    pitch_shift=0,
    index_rate=0.75
)
```

---

## Troubleshooting

### 1. "RVC directory not found"
**Solution:** Clone RVC repository:
```bash
git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git
```

### 2. "No speech detected in audio"
**Solution:** Audio might be music-only. Use voice isolation first:
```bash
python src/preprocessing/voice_isolator.py input.mp3
python src/preprocessing/speech_enhancer.py outputs/isolated/vocals.wav output.wav
```

### 3. "Audio quality below threshold"
**Solution:**
- Record in quieter environment
- Use higher quality microphone
- Check SNR in quality report
- May still work, but quality will suffer

### 4. "CUDA out of memory"
**Solution:**
- Reduce `segment_size` in voice isolator (8 → 4)
- Reduce `batch_size` in RVC trainer (8 → 4)
- Close other GPU applications

### 5. "F5-TTS not available"
**Solution:** This is optional. Install with:
```bash
pip install f5-tts
```

---

## Performance Benchmarks

### RTX 3070 (8GB VRAM):
- **Voice Isolation:** 30-45s per 3-min audio
- **Speech Enhancement:** 15-30s
- **Quality Validation:** 5s
- **RVC Training:** 30-40 min (200 epochs)
- **Voice Conversion:** 30-60s per 3-min song
- **Total End-to-End:** 35-45 minutes

### CPU (No GPU):
- **Voice Isolation:** 5-10 minutes (slow)
- **RVC Training:** Not recommended (hours)
- **Voice Conversion:** 3-5 minutes

---

## Next Steps

After verifying Phase 2 works:

1. **Test with real audio:**
   - Record 5-10 minutes of your voice
   - Choose target song to convert
   - Run end-to-end pipeline

2. **Optimize parameters:**
   - Experiment with `index_rate` (0.5-1.0)
   - Try different `pitch_shift` values
   - Adjust training epochs (100-300)

3. **Move to Phase 3:**
   - Web application development
   - REST API endpoints
   - User interface
   - Job queue system

---

## Support & Documentation

- **Implementation Notes:** `IMPLEMENTATION_NOTES.md`
- **Phase 2 Spec:** `PHASE_2_CORE_PIPELINE.md`
- **Test Suite:** `tests/test_pipeline.py`
- **Logging Config:** `config_logging.py`

---

**Ready to start? Run the test suite:**

```bash
python tests/test_pipeline.py test_audio/sample.mp3
```

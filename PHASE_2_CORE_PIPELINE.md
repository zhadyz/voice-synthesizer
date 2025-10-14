# PHASE 2: CORE ML PIPELINE IMPLEMENTATION

**Duration:** 7-10 days
**Goal:** Working voice cloning pipeline from audio input to converted output

---

## Overview

Phase 2 builds the core ML pipeline with three subsystems:
1. **Preprocessing Pipeline** (Days 1-3)
2. **Training Pipeline** (Days 4-6)
3. **Inference Pipeline** (Days 7-10)

---

## SUBSYSTEM 1: PREPROCESSING PIPELINE (Days 1-3)

### Day 1: Voice Isolation Module

**File:** `src/preprocessing/voice_isolator.py`

```python
"""
Voice Isolation using BS-RoFormer
Separates vocals from music/noise in audio files
"""

import torch
from audio_separator.separator import Separator
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class VoiceIsolator:
    def __init__(
        self,
        model_name='model_bs_roformer_ep_317_sdr_12.9755.ckpt',
        output_dir='outputs/isolated',
        segment_size=8  # For RTX 3070 8GB VRAM
    ):
        """
        Initialize voice isolation model

        Args:
            model_name: BS-RoFormer model checkpoint
            output_dir: Directory for isolated audio output
            segment_size: Lower = less VRAM (8 = ~3GB VRAM)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize separator
        logger.info(f"Loading voice isolation model: {model_name}")
        self.separator = Separator(
            output_dir=str(self.output_dir),
            output_format='wav'
        )

        # Load BS-RoFormer model
        self.separator.load_model(
            model_filename=model_name,
            segment_size=segment_size  # RTX 3070 optimization
        )

        logger.info("Voice isolator ready")

    def isolate_vocals(self, audio_path: str) -> str:
        """
        Extract vocals from audio file

        Args:
            audio_path: Path to input audio (MP3, WAV, etc.)

        Returns:
            Path to isolated vocals WAV file
        """
        logger.info(f"Isolating vocals from: {audio_path}")

        # Separate audio
        output_files = self.separator.separate(audio_path)

        # Find vocals file
        vocals_path = None
        for file in output_files:
            if 'vocals' in file.lower() or 'vocal' in file.lower():
                vocals_path = file
                break

        if vocals_path is None:
            raise ValueError("No vocals file generated")

        logger.info(f"Vocals isolated: {vocals_path}")
        return vocals_path

    def cleanup(self):
        """Clean up GPU memory"""
        if hasattr(self, 'separator'):
            del self.separator
        torch.cuda.empty_cache()


# Test script
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python voice_isolator.py <audio_file>")
        sys.exit(1)

    isolator = VoiceIsolator()
    vocals = isolator.isolate_vocals(sys.argv[1])
    print(f"✓ Vocals saved to: {vocals}")
```

**Test:**
```bash
python src/preprocessing/voice_isolator.py test_audio/sample.mp3
```

---

### Day 2: VAD and Speech Enhancement

**File:** `src/preprocessing/speech_enhancer.py`

```python
"""
Voice Activity Detection + Speech Enhancement
Removes silence and cleans up noise from isolated vocals
"""

import torch
import torchaudio
from denoiser import pretrained
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SpeechEnhancer:
    def __init__(self):
        """Initialize VAD and denoiser models"""

        # Load Silero VAD
        logger.info("Loading Silero VAD...")
        self.vad_model, self.vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False
        )
        self.get_speech_timestamps = self.vad_utils[0]

        # Load Facebook Denoiser
        logger.info("Loading Facebook Denoiser...")
        self.denoiser = pretrained.dns64()
        self.denoiser.eval()

        logger.info("Speech enhancer ready")

    def detect_speech_segments(
        self,
        audio_path: str,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100
    ):
        """
        Detect speech segments in audio using VAD

        Args:
            audio_path: Path to audio file
            min_speech_duration_ms: Minimum speech segment length
            min_silence_duration_ms: Minimum silence gap

        Returns:
            List of speech segments with timestamps
        """
        logger.info(f"Detecting speech in: {audio_path}")

        # Load audio
        wav, sr = torchaudio.load(audio_path)

        # Resample to 16kHz (VAD requirement)
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            wav_16k = resampler(wav)
        else:
            wav_16k = wav

        # Get speech timestamps
        speech_timestamps = self.get_speech_timestamps(
            wav_16k[0],  # VAD expects mono
            self.vad_model,
            sampling_rate=16000,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms
        )

        logger.info(f"Found {len(speech_timestamps)} speech segments")
        return speech_timestamps, wav, sr

    def denoise_audio(self, audio: torch.Tensor, sr: int) -> torch.Tensor:
        """
        Remove noise from audio using Facebook Denoiser

        Args:
            audio: Audio tensor (channels x samples)
            sr: Sample rate

        Returns:
            Denoised audio tensor
        """
        logger.info("Denoising audio...")

        # Denoiser expects 16kHz
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            audio_16k = resampler(audio)
        else:
            audio_16k = audio

        # Denoise
        with torch.no_grad():
            denoised = self.denoiser(audio_16k.unsqueeze(0))

        # Resample back to original rate
        if sr != 16000:
            resampler_back = torchaudio.transforms.Resample(16000, sr)
            denoised = resampler_back(denoised.squeeze(0))
        else:
            denoised = denoised.squeeze(0)

        logger.info("Denoising complete")
        return denoised

    def extract_clean_speech(
        self,
        audio_path: str,
        output_path: str,
        apply_denoising: bool = True
    ):
        """
        Extract clean speech segments from audio

        Args:
            audio_path: Input audio file
            output_path: Output clean audio file
            apply_denoising: Whether to apply denoising
        """
        # Detect speech segments
        timestamps, wav, sr = self.detect_speech_segments(audio_path)

        if len(timestamps) == 0:
            raise ValueError("No speech detected in audio")

        # Extract speech segments
        speech_segments = []
        for ts in timestamps:
            start = int(ts['start'] * sr / 16000)  # Convert 16kHz timestamps to original sr
            end = int(ts['end'] * sr / 16000)
            segment = wav[:, start:end]
            speech_segments.append(segment)

        # Concatenate segments
        clean_audio = torch.cat(speech_segments, dim=1)

        # Apply denoising
        if apply_denoising:
            clean_audio = self.denoise_audio(clean_audio, sr)

        # Save
        torchaudio.save(output_path, clean_audio, sr)
        logger.info(f"Clean speech saved to: {output_path}")

        return output_path


# Test script
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python speech_enhancer.py <input_audio> <output_audio>")
        sys.exit(1)

    enhancer = SpeechEnhancer()
    clean_audio = enhancer.extract_clean_speech(sys.argv[1], sys.argv[2])
    print(f"✓ Clean speech saved to: {clean_audio}")
```

**Test:**
```bash
python src/preprocessing/speech_enhancer.py outputs/isolated/vocals.wav outputs/clean/clean_vocals.wav
```

---

### Day 3: Quality Validation

**File:** `src/preprocessing/quality_validator.py`

```python
"""
Audio Quality Validation
Checks SNR, duration, sample rate, and speaker count
"""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class QualityValidator:
    def __init__(
        self,
        min_snr_db: float = 20.0,
        min_duration_sec: float = 5.0,
        max_duration_sec: float = 600.0,
        target_sr: int = 22050
    ):
        """
        Initialize quality validator

        Args:
            min_snr_db: Minimum acceptable SNR in dB
            min_duration_sec: Minimum audio duration
            max_duration_sec: Maximum audio duration
            target_sr: Target sample rate
        """
        self.min_snr_db = min_snr_db
        self.min_duration_sec = min_duration_sec
        self.max_duration_sec = max_duration_sec
        self.target_sr = target_sr

    def calculate_snr(self, audio: np.ndarray, sr: int) -> float:
        """
        Calculate Signal-to-Noise Ratio

        Args:
            audio: Audio signal
            sr: Sample rate

        Returns:
            SNR in dB
        """
        # Simple energy-based SNR estimation
        # Assumes noise is in silent sections

        # Compute energy in dB
        energy = librosa.feature.rms(y=audio)[0]
        energy_db = librosa.amplitude_to_db(energy)

        # Estimate noise floor (bottom 10% of energy)
        noise_floor = np.percentile(energy_db, 10)

        # Signal level (top 90% of energy)
        signal_level = np.percentile(energy_db, 90)

        # SNR = signal - noise
        snr = signal_level - noise_floor

        return float(snr)

    def validate_audio(self, audio_path: str) -> dict:
        """
        Comprehensive audio quality validation

        Args:
            audio_path: Path to audio file

        Returns:
            Validation results dict
        """
        logger.info(f"Validating: {audio_path}")

        # Load audio
        audio, sr = librosa.load(audio_path, sr=None)

        # Duration check
        duration = len(audio) / sr
        duration_valid = self.min_duration_sec <= duration <= self.max_duration_sec

        # SNR check
        snr = self.calculate_snr(audio, sr)
        snr_valid = snr >= self.min_snr_db

        # Sample rate check
        sr_valid = (sr == self.target_sr) or (sr >= 16000)

        # Check for clipping
        clipping_rate = np.mean(np.abs(audio) > 0.99)
        clipping_valid = clipping_rate < 0.01  # Less than 1% clipping

        # Overall validation
        is_valid = all([duration_valid, snr_valid, sr_valid, clipping_valid])

        results = {
            'valid': is_valid,
            'duration_sec': duration,
            'duration_valid': duration_valid,
            'snr_db': snr,
            'snr_valid': snr_valid,
            'sample_rate': sr,
            'sample_rate_valid': sr_valid,
            'clipping_rate': clipping_rate,
            'clipping_valid': clipping_valid,
            'quality_score': self._calculate_quality_score(snr, clipping_rate)
        }

        logger.info(f"Validation results: {results}")
        return results

    def _calculate_quality_score(self, snr: float, clipping_rate: float) -> str:
        """Calculate overall quality rating"""
        if snr >= 25 and clipping_rate < 0.005:
            return "EXCELLENT"
        elif snr >= 20 and clipping_rate < 0.01:
            return "GOOD"
        elif snr >= 15:
            return "ACCEPTABLE"
        else:
            return "POOR"

    def generate_report(self, audio_path: str) -> str:
        """Generate human-readable validation report"""
        results = self.validate_audio(audio_path)

        report = f"""
╔══════════════════════════════════════════════════════════╗
║              AUDIO QUALITY REPORT                        ║
╠══════════════════════════════════════════════════════════╣
║ File: {Path(audio_path).name:<44} ║
║                                                          ║
║ Duration:      {results['duration_sec']:>6.2f}s  [{('✓' if results['duration_valid'] else '✗')}]              ║
║ SNR:           {results['snr_db']:>6.2f} dB [{('✓' if results['snr_valid'] else '✗')}]              ║
║ Sample Rate:   {results['sample_rate']:>6} Hz [{('✓' if results['sample_rate_valid'] else '✗')}]              ║
║ Clipping:      {results['clipping_rate']*100:>6.2f}%  [{('✓' if results['clipping_valid'] else '✗')}]              ║
║                                                          ║
║ Quality Score: {results['quality_score']:<44} ║
║ Overall:       {('PASS' if results['valid'] else 'FAIL'):<44} ║
╚══════════════════════════════════════════════════════════╝
"""
        return report


# Test script
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python quality_validator.py <audio_file>")
        sys.exit(1)

    validator = QualityValidator()
    report = validator.generate_report(sys.argv[1])
    print(report)
```

**Test:**
```bash
python src/preprocessing/quality_validator.py outputs/clean/clean_vocals.wav
```

---

## SUBSYSTEM 2: TRAINING PIPELINE (Days 4-6)

### Day 4-5: RVC Training Wrapper

**File:** `src/training/rvc_trainer.py`

```python
"""
RVC Training Pipeline
Trains voice conversion model on user voice data
"""

import os
import sys
import subprocess
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class RVCTrainer:
    def __init__(
        self,
        rvc_dir: str = "Retrieval-based-Voice-Conversion-WebUI",
        models_dir: str = "trained_models",
        batch_size: int = 8  # Optimized for RTX 3070
    ):
        """
        Initialize RVC trainer

        Args:
            rvc_dir: Path to RVC repository
            models_dir: Directory to save trained models
            batch_size: Training batch size (lower = less VRAM)
        """
        self.rvc_dir = Path(rvc_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size

        if not self.rvc_dir.exists():
            raise ValueError(f"RVC directory not found: {rvc_dir}")

        logger.info(f"RVC trainer initialized (batch_size={batch_size})")

    def prepare_training_data(
        self,
        audio_path: str,
        model_name: str
    ) -> Path:
        """
        Prepare training data directory for RVC

        Args:
            audio_path: Path to cleaned training audio
            model_name: Name for the voice model

        Returns:
            Path to prepared dataset directory
        """
        logger.info(f"Preparing training data for model: {model_name}")

        # Create dataset directory
        dataset_dir = self.rvc_dir / "datasets" / model_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Copy/segment audio
        # RVC expects audio files in datasets/{model_name}/
        import shutil
        shutil.copy(audio_path, dataset_dir / Path(audio_path).name)

        logger.info(f"Dataset prepared: {dataset_dir}")
        return dataset_dir

    def train_model(
        self,
        model_name: str,
        dataset_dir: Path,
        epochs: int = 200,
        sample_rate: int = 40000,
        pitch_extraction: str = "rmvpe"
    ):
        """
        Train RVC voice conversion model

        Args:
            model_name: Name of voice model
            dataset_dir: Path to training dataset
            epochs: Number of training epochs
            sample_rate: Audio sample rate (40k recommended)
            pitch_extraction: Pitch extraction method (rmvpe/harvest)
        """
        logger.info(f"Starting RVC training for: {model_name}")
        logger.info(f"Epochs: {epochs}, Sample Rate: {sample_rate}, Batch Size: {self.batch_size}")

        # RVC training command
        train_script = self.rvc_dir / "train_nsf_sim_cache_sid_load_pretrain.py"

        cmd = [
            sys.executable,
            str(train_script),
            "-n", model_name,
            "-sr", str(sample_rate),
            "-e", str(epochs),
            "-bs", str(self.batch_size),
            "-g", "0",  # GPU 0
            "-pd", pitch_extraction
        ]

        logger.info(f"Training command: {' '.join(cmd)}")

        # Run training
        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.rvc_dir),
                check=True,
                capture_output=True,
                text=True
            )
            logger.info(f"Training complete: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e.stderr}")
            raise

        # Find trained model checkpoint
        weights_dir = self.rvc_dir / "weights"
        model_path = weights_dir / f"{model_name}.pth"

        if not model_path.exists():
            raise ValueError(f"Trained model not found: {model_path}")

        logger.info(f"Model saved: {model_path}")
        return model_path

    def train_from_audio(
        self,
        audio_path: str,
        model_name: str,
        epochs: int = 200
    ) -> Path:
        """
        End-to-end training from audio file

        Args:
            audio_path: Path to cleaned training audio
            model_name: Name for voice model
            epochs: Training epochs

        Returns:
            Path to trained model checkpoint
        """
        # Prepare data
        dataset_dir = self.prepare_training_data(audio_path, model_name)

        # Train model
        model_path = self.train_model(model_name, dataset_dir, epochs)

        return model_path


# Test script
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python rvc_trainer.py <clean_audio> <model_name>")
        sys.exit(1)

    trainer = RVCTrainer()
    model_path = trainer.train_from_audio(sys.argv[1], sys.argv[2])
    print(f"✓ Model trained: {model_path}")
```

---

### Day 6: F5-TTS Zero-Shot Module

**File:** `src/training/f5_tts_wrapper.py`

```python
"""
F5-TTS Zero-Shot Voice Cloning
Instant voice cloning without training
"""

import torch
import torchaudio
from f5_tts.infer.infer import load_model, infer
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class F5TTSWrapper:
    def __init__(self):
        """Initialize F5-TTS model"""
        logger.info("Loading F5-TTS model...")
        self.model = load_model()
        logger.info("F5-TTS ready")

    def clone_voice(
        self,
        reference_audio: str,
        target_text: str,
        output_path: str,
        reference_duration_sec: float = 15.0
    ):
        """
        Zero-shot voice cloning

        Args:
            reference_audio: 10-15 second sample of target voice
            target_text: Text to synthesize in cloned voice
            output_path: Where to save output audio
            reference_duration_sec: Duration of reference to use
        """
        logger.info(f"Cloning voice from: {reference_audio}")
        logger.info(f"Synthesizing: {target_text}")

        # Load reference audio
        ref_audio, sr = torchaudio.load(reference_audio)

        # Trim to specified duration
        max_samples = int(reference_duration_sec * sr)
        if ref_audio.shape[1] > max_samples:
            ref_audio = ref_audio[:, :max_samples]

        # Run inference
        output_audio = infer(
            model=self.model,
            reference_audio=ref_audio,
            reference_sr=sr,
            target_text=target_text
        )

        # Save output
        torchaudio.save(output_path, output_audio, sr)
        logger.info(f"Cloned voice saved: {output_path}")

        return output_path


# Test script
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python f5_tts_wrapper.py <reference_audio> <text> <output>")
        sys.exit(1)

    wrapper = F5TTSWrapper()
    output = wrapper.clone_voice(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"✓ Voice cloned: {output}")
```

---

## SUBSYSTEM 3: INFERENCE PIPELINE (Days 7-10)

### Day 7-8: RVC Inference Engine

**File:** `src/inference/voice_converter.py`

```python
"""
RVC Voice Conversion Inference
Converts target audio to user's voice
"""

import sys
import torch
import torchaudio
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class VoiceConverter:
    def __init__(
        self,
        rvc_dir: str = "Retrieval-based-Voice-Conversion-WebUI",
        model_path: str = None
    ):
        """
        Initialize voice converter

        Args:
            rvc_dir: Path to RVC repository
            model_path: Path to trained model checkpoint
        """
        self.rvc_dir = Path(rvc_dir)
        self.model_path = model_path

        if model_path:
            self.load_model(model_path)

        logger.info("Voice converter initialized")

    def load_model(self, model_path: str):
        """Load trained RVC model"""
        logger.info(f"Loading model: {model_path}")

        # Add RVC to path
        sys.path.append(str(self.rvc_dir))

        # Import RVC inference modules
        from infer.modules.vc.modules import VC

        # Initialize VC
        self.vc = VC()
        self.vc.get_vc(model_path)

        logger.info("Model loaded successfully")

    def convert_voice(
        self,
        input_audio: str,
        output_audio: str,
        pitch_shift: int = 0,
        index_rate: float = 0.75
    ):
        """
        Convert audio to trained voice

        Args:
            input_audio: Target audio (e.g., Michael Jackson song)
            output_audio: Output path for converted audio
            pitch_shift: Pitch adjustment in semitones
            index_rate: Feature retrieval strength (0.0-1.0)
        """
        logger.info(f"Converting: {input_audio}")
        logger.info(f"Pitch shift: {pitch_shift}, Index rate: {index_rate}")

        # Run RVC conversion
        result = self.vc.vc_single(
            sid=0,
            input_audio_path=input_audio,
            f0_up_key=pitch_shift,
            f0_file=None,
            f0_method="rmvpe",
            file_index="",
            index_rate=index_rate,
            filter_radius=3,
            resample_sr=0,
            rms_mix_rate=0.25,
            protect=0.33
        )

        # Save output
        output_audio_data, sr = result
        torchaudio.save(output_audio, torch.from_numpy(output_audio_data).unsqueeze(0), sr)

        logger.info(f"Conversion complete: {output_audio}")
        return output_audio


# Test script
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python voice_converter.py <model_path> <input_audio> <output_audio>")
        sys.exit(1)

    converter = VoiceConverter(model_path=sys.argv[1])
    output = converter.convert_voice(sys.argv[2], sys.argv[3])
    print(f"✓ Voice converted: {output}")
```

---

### Day 9-10: End-to-End Pipeline Integration

**File:** `src/pipeline/voice_cloning_pipeline.py`

```python
"""
End-to-End Voice Cloning Pipeline
Orchestrates preprocessing, training, and inference
"""

import logging
from pathlib import Path
from src.preprocessing.voice_isolator import VoiceIsolator
from src.preprocessing.speech_enhancer import SpeechEnhancer
from src.preprocessing.quality_validator import QualityValidator
from src.training.rvc_trainer import RVCTrainer
from src.inference.voice_converter import VoiceConverter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VoiceCloningPipeline:
    def __init__(self, output_dir: str = "pipeline_outputs"):
        """Initialize complete voice cloning pipeline"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.voice_isolator = VoiceIsolator()
        self.speech_enhancer = SpeechEnhancer()
        self.quality_validator = QualityValidator()
        self.rvc_trainer = RVCTrainer()

        logger.info("Voice cloning pipeline ready")

    def preprocess_training_audio(
        self,
        raw_audio_path: str,
        user_id: str
    ) -> dict:
        """
        Preprocess user's voice recording for training

        Returns:
            dict with clean_audio_path and quality_report
        """
        logger.info(f"=== PREPROCESSING TRAINING AUDIO ===")
        logger.info(f"User: {user_id}")
        logger.info(f"Input: {raw_audio_path}")

        # Stage 1: Voice isolation
        logger.info("[1/4] Isolating voice...")
        vocals_path = self.voice_isolator.isolate_vocals(raw_audio_path)

        # Stage 2: Speech enhancement
        logger.info("[2/4] Enhancing speech...")
        clean_path = self.output_dir / f"{user_id}_clean_training.wav"
        self.speech_enhancer.extract_clean_speech(vocals_path, str(clean_path))

        # Stage 3: Quality validation
        logger.info("[3/4] Validating quality...")
        validation = self.quality_validator.validate_audio(str(clean_path))
        report = self.quality_validator.generate_report(str(clean_path))
        print(report)

        if not validation['valid']:
            logger.warning("Audio quality below threshold!")
            logger.warning("Consider re-recording in a quieter environment.")

        # Stage 4: Cleanup
        logger.info("[4/4] Cleaning up...")
        self.voice_isolator.cleanup()

        return {
            'clean_audio_path': str(clean_path),
            'quality_report': validation
        }

    def train_voice_model(
        self,
        clean_audio_path: str,
        model_name: str,
        epochs: int = 200
    ) -> str:
        """
        Train RVC voice model

        Returns:
            Path to trained model checkpoint
        """
        logger.info(f"=== TRAINING VOICE MODEL ===")
        logger.info(f"Model name: {model_name}")
        logger.info(f"Training data: {clean_audio_path}")
        logger.info(f"Epochs: {epochs}")
        logger.info(f"Estimated time: 30-40 minutes (RTX 3070)")

        model_path = self.rvc_trainer.train_from_audio(
            clean_audio_path,
            model_name,
            epochs
        )

        logger.info(f"✓ Model trained: {model_path}")
        return str(model_path)

    def convert_audio(
        self,
        model_path: str,
        target_audio_path: str,
        output_name: str
    ) -> str:
        """
        Convert target audio to user's voice

        Returns:
            Path to converted audio
        """
        logger.info(f"=== CONVERTING AUDIO ===")
        logger.info(f"Model: {model_path}")
        logger.info(f"Target: {target_audio_path}")

        # Preprocess target audio (extract vocals)
        logger.info("[1/3] Isolating target vocals...")
        target_vocals = self.voice_isolator.isolate_vocals(target_audio_path)

        # Convert voice
        logger.info("[2/3] Converting voice...")
        converter = VoiceConverter(model_path=model_path)
        output_path = self.output_dir / f"{output_name}_converted.wav"
        converter.convert_voice(target_vocals, str(output_path))

        # Cleanup
        logger.info("[3/3] Cleaning up...")
        self.voice_isolator.cleanup()

        logger.info(f"✓ Conversion complete: {output_path}")
        return str(output_path)

    def end_to_end_workflow(
        self,
        training_audio: str,
        target_audio: str,
        user_id: str
    ) -> dict:
        """
        Complete workflow: preprocess → train → convert

        Returns:
            dict with all output paths
        """
        logger.info("=== END-TO-END VOICE CLONING WORKFLOW ===")

        # Step 1: Preprocess training audio
        preprocessing_result = self.preprocess_training_audio(training_audio, user_id)

        if not preprocessing_result['quality_report']['valid']:
            logger.error("Training audio quality too low. Aborting.")
            return {'error': 'Poor audio quality', 'report': preprocessing_result['quality_report']}

        # Step 2: Train model
        model_name = f"voice_model_{user_id}"
        model_path = self.train_voice_model(
            preprocessing_result['clean_audio_path'],
            model_name
        )

        # Step 3: Convert target audio
        output_audio = self.convert_audio(model_path, target_audio, user_id)

        return {
            'clean_training_audio': preprocessing_result['clean_audio_path'],
            'quality_report': preprocessing_result['quality_report'],
            'model_path': model_path,
            'converted_audio': output_audio
        }


# Test script
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("Usage: python voice_cloning_pipeline.py <training_audio> <target_audio> <user_id>")
        print()
        print("Example:")
        print("  python voice_cloning_pipeline.py my_voice.mp3 michael_jackson.mp3 user123")
        sys.exit(1)

    pipeline = VoiceCloningPipeline()
    result = pipeline.end_to_end_workflow(sys.argv[1], sys.argv[2], sys.argv[3])

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Converted Audio: {result['converted_audio']}")
    print("=" * 60)
```

---

## Phase 2 Complete Checklist

- [ ] Voice isolator module implemented
- [ ] Speech enhancer (VAD + denoiser) implemented
- [ ] Quality validator implemented
- [ ] RVC training wrapper implemented
- [ ] F5-TTS wrapper implemented
- [ ] Voice converter (inference) implemented
- [ ] End-to-end pipeline integration complete
- [ ] All modules tested individually
- [ ] Full pipeline tested end-to-end
- [ ] RTX 3070 memory optimization verified
- [ ] Processing times measured and acceptable

**Once complete, proceed to Phase 3: Web Application Development**

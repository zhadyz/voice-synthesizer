"""
Audio Quality Validation
Checks SNR, duration, sample rate, and clipping
"""

import torch
import torchaudio
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

        # Compute RMS energy (frame-based)
        frame_length = 2048
        hop_length = 512

        # Pad audio if needed
        if len(audio) < frame_length:
            audio = np.pad(audio, (0, frame_length - len(audio)))

        # Calculate RMS energy per frame
        energy = []
        for i in range(0, len(audio) - frame_length + 1, hop_length):
            frame = audio[i:i + frame_length]
            rms = np.sqrt(np.mean(frame ** 2))
            energy.append(rms)

        energy = np.array(energy)

        # Convert to dB (add small epsilon to avoid log(0))
        energy_db = 20 * np.log10(energy + 1e-10)

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

        # Load audio using torchaudio (Python 3.13 compatible)
        waveform, sr = torchaudio.load(audio_path)

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Convert to numpy array
        audio = waveform[0].numpy()

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

        # Use ASCII characters for Windows console compatibility
        check = '[PASS]' if True else '[FAIL]'
        fail = '[FAIL]'

        report = f"""
================================================================
                 AUDIO QUALITY REPORT
================================================================
File: {Path(audio_path).name}

Duration:      {results['duration_sec']:>6.2f}s  [{check if results['duration_valid'] else fail}]
SNR:           {results['snr_db']:>6.2f} dB [{check if results['snr_valid'] else fail}]
Sample Rate:   {results['sample_rate']:>6} Hz [{check if results['sample_rate_valid'] else fail}]
Clipping:      {results['clipping_rate']*100:>6.2f}%  [{check if results['clipping_valid'] else fail}]

Quality Score: {results['quality_score']}
Overall:       {'PASS' if results['valid'] else 'FAIL'}
================================================================
"""
        return report


# Test script
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python quality_validator.py <audio_file>")
        sys.exit(1)

    validator = QualityValidator()
    report = validator.generate_report(sys.argv[1])
    print(report)

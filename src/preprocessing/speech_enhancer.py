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

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python speech_enhancer.py <input_audio> <output_audio>")
        sys.exit(1)

    enhancer = SpeechEnhancer()
    clean_audio = enhancer.extract_clean_speech(sys.argv[1], sys.argv[2])
    print(f"✓ Clean speech saved to: {clean_audio}")

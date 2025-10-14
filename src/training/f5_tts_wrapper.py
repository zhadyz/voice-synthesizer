"""
F5-TTS Zero-Shot Voice Cloning
Instant voice cloning without training
"""

import torch
import torchaudio
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class F5TTSWrapper:
    def __init__(self):
        """Initialize F5-TTS model"""
        logger.info("Loading F5-TTS model...")

        try:
            # Import F5-TTS inference functions
            from f5_tts.infer.infer import load_model, infer
            self.load_model = load_model
            self.infer = infer

            # Load model
            self.model = self.load_model()
            logger.info("F5-TTS ready")
        except ImportError as e:
            logger.warning(f"F5-TTS not available: {e}")
            logger.warning("Install with: pip install f5-tts")
            self.model = None

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
        if self.model is None:
            raise RuntimeError("F5-TTS model not loaded. Install f5-tts package.")

        logger.info(f"Cloning voice from: {reference_audio}")
        logger.info(f"Synthesizing: {target_text}")

        # Load reference audio
        ref_audio, sr = torchaudio.load(reference_audio)

        # Trim to specified duration
        max_samples = int(reference_duration_sec * sr)
        if ref_audio.shape[1] > max_samples:
            ref_audio = ref_audio[:, :max_samples]

        # Run inference
        output_audio = self.infer(
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

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 4:
        print("Usage: python f5_tts_wrapper.py <reference_audio> <text> <output>")
        sys.exit(1)

    wrapper = F5TTSWrapper()
    output = wrapper.clone_voice(sys.argv[1], sys.argv[2], sys.argv[3])
    print(f"✓ Voice cloned: {output}")

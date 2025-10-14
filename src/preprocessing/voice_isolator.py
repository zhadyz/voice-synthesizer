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
        # Note: segment_size parameter removed (not supported by audio-separator 0.39.0)
        self.separator.load_model(model_filename=model_name)

        logger.info("Voice isolator ready")

    def isolate_vocals(self, audio_path: str) -> str:
        """
        Extract vocals from audio file

        Args:
            audio_path: Path to input audio (MP3, WAV, etc.)

        Returns:
            Path to isolated vocals WAV file (absolute path)
        """
        logger.info(f"Isolating vocals from: {audio_path}")

        # Separate audio
        output_files = self.separator.separate(audio_path)

        # Find vocals file (look for "(Vocals)" or "(Vocal)" in the filename)
        vocals_path = None
        for file in output_files:
            file_lower = file.lower()
            if '(vocals)' in file_lower or '(vocal)' in file_lower:
                vocals_path = file
                break

        if vocals_path is None:
            raise ValueError("No vocals file generated")

        # Convert to absolute path
        vocals_path_abs = Path(self.output_dir) / Path(vocals_path).name
        if not vocals_path_abs.exists():
            # Try the path as-is if it's already absolute
            vocals_path_abs = Path(vocals_path)

        vocals_path_str = str(vocals_path_abs.absolute())
        logger.info(f"Vocals isolated: {vocals_path_str}")
        return vocals_path_str

    def cleanup(self):
        """Clean up GPU memory"""
        if hasattr(self, 'separator'):
            del self.separator
        torch.cuda.empty_cache()


# Test script
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python voice_isolator.py <audio_file>")
        sys.exit(1)

    isolator = VoiceIsolator()
    vocals = isolator.isolate_vocals(sys.argv[1])
    print(f"✓ Vocals saved to: {vocals}")
    isolator.cleanup()

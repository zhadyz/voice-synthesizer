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
        self.vc = None

        if not self.rvc_dir.exists():
            raise ValueError(f"RVC directory not found: {rvc_dir}")

        if model_path:
            self.load_model(model_path)

        logger.info("Voice converter initialized")

    def load_model(self, model_path: str):
        """Load trained RVC model"""
        logger.info(f"Loading model: {model_path}")

        # Add RVC to path
        if str(self.rvc_dir) not in sys.path:
            sys.path.append(str(self.rvc_dir))

        try:
            # Import RVC inference modules
            from infer.modules.vc.modules import VC

            # Initialize VC
            self.vc = VC()
            self.vc.get_vc(model_path)

            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load RVC model: {e}")
            logger.error("Make sure RVC repository is properly set up")
            raise

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
        if self.vc is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.info(f"Converting: {input_audio}")
        logger.info(f"Pitch shift: {pitch_shift}, Index rate: {index_rate}")

        try:
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
            output_tensor = torch.from_numpy(output_audio_data).unsqueeze(0)
            torchaudio.save(output_audio, output_tensor, sr)

            logger.info(f"Conversion complete: {output_audio}")
            return output_audio
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            raise


# Test script
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 4:
        print("Usage: python voice_converter.py <model_path> <input_audio> <output_audio>")
        sys.exit(1)

    converter = VoiceConverter(model_path=sys.argv[1])
    output = converter.convert_voice(sys.argv[2], sys.argv[3])
    print(f"✓ Voice converted: {output}")

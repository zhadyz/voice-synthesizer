"""
End-to-End Voice Cloning Pipeline
Orchestrates preprocessing, training, and inference
"""

import logging
import time
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

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

        start_time = time.time()

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

        elapsed = time.time() - start_time
        logger.info(f"Preprocessing complete in {elapsed:.1f}s")

        return {
            'clean_audio_path': str(clean_path),
            'quality_report': validation,
            'preprocessing_time': elapsed
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

        start_time = time.time()

        model_path = self.rvc_trainer.train_from_audio(
            clean_audio_path,
            model_name,
            epochs
        )

        elapsed = time.time() - start_time
        logger.info(f"Training complete in {elapsed/60:.1f} minutes")
        logger.info(f"Model saved: {model_path}")

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

        start_time = time.time()

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

        elapsed = time.time() - start_time
        logger.info(f"Conversion complete in {elapsed:.1f}s")
        logger.info(f"Output: {output_path}")

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
        workflow_start = time.time()

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

        total_time = time.time() - workflow_start
        logger.info(f"\n=== WORKFLOW COMPLETE ({total_time/60:.1f} minutes) ===")

        return {
            'clean_training_audio': preprocessing_result['clean_audio_path'],
            'quality_report': preprocessing_result['quality_report'],
            'model_path': model_path,
            'converted_audio': output_audio,
            'total_time': total_time
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

    if 'error' not in result:
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Clean Training Audio: {result['clean_training_audio']}")
        print(f"Trained Model: {result['model_path']}")
        print(f"Converted Audio: {result['converted_audio']}")
        print(f"Total Time: {result['total_time']/60:.1f} minutes")
        print("=" * 60)
    else:
        print(f"\nERROR: {result['error']}")

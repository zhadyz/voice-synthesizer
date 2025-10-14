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
        models_dir: str = "outputs/trained_models",
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

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 3:
        print("Usage: python rvc_trainer.py <clean_audio> <model_name>")
        sys.exit(1)

    trainer = RVCTrainer()
    model_path = trainer.train_from_audio(sys.argv[1], sys.argv[2])
    print(f"✓ Model trained: {model_path}")

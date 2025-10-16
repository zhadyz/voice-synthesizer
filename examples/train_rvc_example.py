"""
RVC Training Example
Demonstrates complete voice cloning training pipeline

This example shows how to train a custom voice model using the RVC trainer.
Requires: RVC repository installed and HuBERT model downloaded
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.rvc_trainer import RVCTrainer

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)

logger = logging.getLogger(__name__)


def example_basic_training():
    """Basic training example with minimal configuration"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Basic Training")
    print("="*70 + "\n")

    # Initialize trainer with default settings
    trainer = RVCTrainer(
        batch_size=8,  # Adjust based on your GPU
        gpu_id=0
    )

    # Train model from audio directory
    model_path = trainer.train_from_audio(
        audio_path="data/voice_samples/",
        model_name="basic_voice",
        total_epochs=200
    )

    logger.info(f"Training complete! Model saved to: {model_path}")


def example_advanced_training():
    """Advanced training with custom configuration"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Advanced Training")
    print("="*70 + "\n")

    # Initialize trainer with custom settings
    trainer = RVCTrainer(
        rvc_dir="Retrieval-based-Voice-Conversion-WebUI",
        batch_size=12,  # Higher batch size for better GPU
        gpu_id=0
    )

    # Train with advanced options
    model_path = trainer.train_from_audio(
        audio_path="data/voice_samples/",
        model_name="advanced_voice",
        total_epochs=300,           # More epochs for better quality
        save_every_epoch=10,        # Save checkpoints every 10 epochs
        sample_rate=48000,          # Higher quality audio
        version="v2",               # Use v2 model (better quality)
        f0_method="rmvpe",          # Best pitch extraction method
        pretrain_g="",              # Optional: path to pretrained generator
        pretrain_d=""               # Optional: path to pretrained discriminator
    )

    logger.info(f"Advanced training complete! Model: {model_path}")


def example_step_by_step():
    """Step-by-step training showing each pipeline stage"""
    print("\n" + "="*70)
    print("EXAMPLE 3: Step-by-Step Training")
    print("="*70 + "\n")

    trainer = RVCTrainer(batch_size=8, gpu_id=0)

    model_name = "step_by_step_voice"
    sample_rate = 40000
    version = "v2"

    # Step 1: Prepare training data
    logger.info("Step 1: Preparing training data...")
    exp_dir = trainer.prepare_training_data(
        audio_path="data/voice_samples/",
        model_name=model_name,
        sample_rate=sample_rate
    )
    logger.info(f"Data prepared in: {exp_dir}")

    # Step 2: Create configuration
    logger.info("Step 2: Creating training configuration...")
    config_path = trainer._create_config(exp_dir, sample_rate, version)
    logger.info(f"Config created: {config_path}")

    # Step 3: Preprocess audio
    logger.info("Step 3: Preprocessing audio...")
    dataset_dir = trainer.rvc_dir / "datasets" / model_name
    trainer.preprocess_audio(dataset_dir, exp_dir, sample_rate)
    logger.info("Audio preprocessing complete")

    # Step 4: Extract F0 features
    logger.info("Step 4: Extracting pitch (F0) features...")
    trainer.extract_f0(exp_dir, method="rmvpe")
    logger.info("F0 extraction complete")

    # Step 5: Extract HuBERT features
    logger.info("Step 5: Extracting HuBERT features...")
    trainer.extract_features(exp_dir, version=version)
    logger.info("Feature extraction complete")

    # Step 6: Train model
    logger.info("Step 6: Training model...")
    model_path = trainer.train_model(
        model_name=model_name,
        exp_dir=exp_dir,
        total_epochs=200,
        save_every_epoch=10,
        sample_rate=sample_rate,
        version=version
    )
    logger.info(f"Model training complete: {model_path}")


def example_low_vram():
    """Training configuration for low VRAM GPUs"""
    print("\n" + "="*70)
    print("EXAMPLE 4: Low VRAM Configuration (4GB GPU)")
    print("="*70 + "\n")

    trainer = RVCTrainer(
        batch_size=4,  # Reduced batch size
        gpu_id=0
    )

    model_path = trainer.train_from_audio(
        audio_path="data/voice_samples/",
        model_name="low_vram_voice",
        total_epochs=200,
        sample_rate=40000,  # Lower sample rate
        version="v1",       # v1 uses less memory
        f0_method="rmvpe"
    )

    logger.info(f"Low VRAM training complete: {model_path}")


def example_high_quality():
    """Training configuration for maximum quality"""
    print("\n" + "="*70)
    print("EXAMPLE 5: High Quality Configuration (24GB GPU)")
    print("="*70 + "\n")

    trainer = RVCTrainer(
        batch_size=16,  # Large batch size
        gpu_id=0
    )

    model_path = trainer.train_from_audio(
        audio_path="data/voice_samples/",
        model_name="high_quality_voice",
        total_epochs=500,           # Many epochs for convergence
        save_every_epoch=20,
        sample_rate=48000,          # Highest quality audio
        version="v2",               # Best model version
        f0_method="rmvpe"           # Best pitch extraction
    )

    logger.info(f"High quality training complete: {model_path}")


def example_validation():
    """Validate RVC installation before training"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Installation Validation")
    print("="*70 + "\n")

    try:
        trainer = RVCTrainer()
        logger.info("RVC installation validated successfully!")
        logger.info(f"RVC directory: {trainer.rvc_dir}")
        logger.info(f"Batch size: {trainer.batch_size}")
        logger.info(f"GPU ID: {trainer.gpu_id}")
    except ValueError as e:
        logger.error(f"RVC installation validation failed: {e}")
        logger.error("Please install RVC and download required models")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("RVC TRAINING EXAMPLES")
    print("="*70 + "\n")

    # Choose which example to run
    examples = {
        "1": ("Basic Training", example_basic_training),
        "2": ("Advanced Training", example_advanced_training),
        "3": ("Step-by-Step", example_step_by_step),
        "4": ("Low VRAM (4GB GPU)", example_low_vram),
        "5": ("High Quality (24GB GPU)", example_high_quality),
        "6": ("Validate Installation", example_validation),
    }

    print("Available examples:")
    for key, (name, _) in examples.items():
        print(f"  {key}. {name}")
    print()

    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("Select example (1-6, or 'all' for validation only): ").strip()

    if choice.lower() == "all":
        example_validation()
    elif choice in examples:
        name, func = examples[choice]
        logger.info(f"Running example: {name}")
        try:
            func()
        except Exception as e:
            logger.error(f"Example failed: {e}", exc_info=True)
    else:
        print(f"Invalid choice: {choice}")
        sys.exit(1)

    print("\n" + "="*70)
    print("EXAMPLES COMPLETE")
    print("="*70 + "\n")

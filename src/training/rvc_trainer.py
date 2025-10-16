"""
RVC Training Pipeline
Trains voice conversion model on user voice data

Compatible with RVC v2 (Retrieval-based-Voice-Conversion-WebUI)
Repository: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI

Training Pipeline:
1. Preprocess audio (slice and normalize)
2. Extract pitch features (f0)
3. Extract HuBERT features
4. Train generator and discriminator models

Expected RVC directory structure:
- infer/modules/train/preprocess.py
- infer/modules/train/extract/extract_f0_rmvpe.py
- infer/modules/train/extract_feature_print.py
- infer/modules/train/train.py
- assets/hubert/hubert_base.pt (required pretrained model)
- configs/v2/*.json (config templates)
"""

import os
import sys
import subprocess
import json
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class RVCTrainer:
    def __init__(
        self,
        rvc_dir: Optional[str] = None,
        models_dir: str = "outputs/trained_models",
        batch_size: int = 8,  # Optimized for RTX 3070
        gpu_id: int = 0
    ):
        """
        Initialize RVC trainer

        Args:
            rvc_dir: Path to RVC repository (default: env var RVC_DIR or "./Retrieval-based-Voice-Conversion-WebUI")
            models_dir: Directory to save trained models
            batch_size: Training batch size (lower = less VRAM, recommended: 4-12)
            gpu_id: GPU device ID (0 for first GPU)
        """
        # Get RVC directory from environment variable or use default
        if rvc_dir is None:
            rvc_dir = os.environ.get(
                "RVC_DIR",
                "Retrieval-based-Voice-Conversion-WebUI"
            )

        self.rvc_dir = Path(rvc_dir)
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.gpu_id = gpu_id

        # Validate RVC installation
        if not self.rvc_dir.exists():
            raise ValueError(
                f"RVC directory not found: {rvc_dir}\n"
                f"Set RVC_DIR environment variable or clone the repository:\n"
                f"git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI"
            )

        # Validate required scripts exist
        self._validate_rvc_installation()

        logger.info(f"RVC trainer initialized (batch_size={batch_size}, gpu={gpu_id})")

    def _validate_rvc_installation(self):
        """Validate that all required RVC scripts and models exist"""
        required_paths = [
            "infer/modules/train/preprocess.py",
            "infer/modules/train/extract/extract_f0_rmvpe.py",
            "infer/modules/train/extract_feature_print.py",
            "infer/modules/train/train.py",
        ]

        missing = []
        for path in required_paths:
            if not (self.rvc_dir / path).exists():
                missing.append(path)

        if missing:
            raise ValueError(
                f"RVC installation incomplete. Missing files:\n" +
                "\n".join(f"  - {p}" for p in missing)
            )

        # Check for HuBERT model (critical for feature extraction)
        hubert_path = self.rvc_dir / "assets" / "hubert" / "hubert_base.pt"
        if not hubert_path.exists():
            logger.warning(
                f"HuBERT model not found at {hubert_path}\n"
                f"Download from: https://huggingface.co/lj1995/VoiceConversionWebUI/tree/main"
            )

        logger.info("RVC installation validated")

    def _create_config(
        self,
        exp_dir: Path,
        sample_rate: int,
        version: str
    ) -> Path:
        """
        Create RVC training config file

        Args:
            exp_dir: Experiment directory
            sample_rate: Audio sample rate (40000 or 48000)
            version: Model version (v1 or v2)

        Returns:
            Path to created config.json
        """
        # Load template config from RVC
        config_template = self.rvc_dir / "configs" / version / f"{sample_rate}.json"

        if not config_template.exists():
            # Fallback: create minimal config
            logger.warning(f"Config template not found: {config_template}")
            config = {
                "train": {
                    "log_interval": 200,
                    "seed": 1234,
                    "epochs": 10000,
                    "learning_rate": 1e-4,
                    "betas": [0.8, 0.99],
                    "eps": 1e-9,
                    "batch_size": self.batch_size,
                    "fp16_run": True,
                    "lr_decay": 0.999875,
                    "segment_size": 17280 if sample_rate == 48000 else 12800,
                    "init_lr_ratio": 1,
                    "warmup_epochs": 0,
                    "c_mel": 45,
                    "c_kl": 1.0,
                },
                "data": {
                    "max_wav_value": 32768.0,
                    "sampling_rate": sample_rate,
                    "filter_length": 2048,
                    "hop_length": 480 if sample_rate == 48000 else 400,
                    "win_length": 2048,
                    "n_mel_channels": 128,
                    "mel_fmin": 0.0,
                    "mel_fmax": None,
                    "training_files": str(exp_dir / "filelist.txt"),
                },
                "model": {
                    "inter_channels": 192,
                    "hidden_channels": 192,
                    "filter_channels": 768,
                    "n_heads": 2,
                    "n_layers": 6,
                    "kernel_size": 3,
                    "p_dropout": 0,
                    "resblock": "1",
                    "resblock_kernel_sizes": [3, 7, 11],
                    "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    "upsample_rates": [12, 10, 2, 2] if sample_rate == 48000 else [10, 8, 2, 2],
                    "upsample_initial_channel": 512,
                    "upsample_kernel_sizes": [24, 20, 4, 4] if sample_rate == 48000 else [20, 16, 4, 4],
                    "use_spectral_norm": False,
                    "gin_channels": 256,
                    "spk_embed_dim": 109,
                },
            }
        else:
            with open(config_template, 'r') as f:
                config = json.load(f)

        # Write config to experiment directory
        config_path = exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

        logger.info(f"Config created: {config_path}")
        return config_path

    def preprocess_audio(
        self,
        audio_dir: Path,
        exp_dir: Path,
        sample_rate: int = 40000,
        n_processes: int = 4
    ):
        """
        Step 1: Preprocess audio (slice and normalize)

        Args:
            audio_dir: Directory containing raw audio files
            exp_dir: Experiment directory
            sample_rate: Target sample rate
            n_processes: Number of parallel processes
        """
        logger.info(f"[1/4] Preprocessing audio: {audio_dir}")

        preprocess_script = self.rvc_dir / "infer" / "modules" / "train" / "preprocess.py"

        cmd = [
            sys.executable,
            str(preprocess_script),
            str(audio_dir),
            str(sample_rate),
            str(n_processes),
            str(exp_dir),
            "False",  # noparallel
            "3.7"  # per (segment duration)
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=str(self.rvc_dir),
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"Preprocessing failed: {result.stderr}")
            raise RuntimeError(f"Audio preprocessing failed: {result.stderr}")

        logger.info("Audio preprocessing complete")

    def extract_f0(
        self,
        exp_dir: Path,
        method: str = "rmvpe",
        n_processes: int = 4
    ):
        """
        Step 2: Extract pitch (f0) features

        Args:
            exp_dir: Experiment directory
            method: Pitch extraction method (rmvpe recommended)
            n_processes: Number of parallel processes
        """
        logger.info(f"[2/4] Extracting f0 features using {method}")

        extract_script = self.rvc_dir / "infer" / "modules" / "train" / "extract" / f"extract_f0_{method}.py"

        if not extract_script.exists():
            raise ValueError(f"F0 extraction method not found: {method}")

        cmd = [
            sys.executable,
            str(extract_script),
            str(exp_dir),
            str(n_processes),
            str(self.gpu_id)
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=str(self.rvc_dir),
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"F0 extraction failed: {result.stderr}")
            raise RuntimeError(f"F0 extraction failed: {result.stderr}")

        logger.info("F0 extraction complete")

    def extract_features(
        self,
        exp_dir: Path,
        version: str = "v2",
        use_fp16: bool = True
    ):
        """
        Step 3: Extract HuBERT features

        Args:
            exp_dir: Experiment directory
            version: Model version (v1 or v2)
            use_fp16: Use half precision
        """
        logger.info(f"[3/4] Extracting HuBERT features (version={version})")

        extract_script = self.rvc_dir / "infer" / "modules" / "train" / "extract_feature_print.py"

        cmd = [
            sys.executable,
            str(extract_script),
            str(self.gpu_id),
            "1",  # n_part
            "0",  # i_part
            str(self.gpu_id),
            str(exp_dir),
            version,
            "True" if use_fp16 else "False"
        ]

        logger.info(f"Running: {' '.join(cmd)}")

        result = subprocess.run(
            cmd,
            cwd=str(self.rvc_dir),
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            logger.error(f"Feature extraction failed: {result.stderr}")
            raise RuntimeError(f"Feature extraction failed: {result.stderr}")

        logger.info("Feature extraction complete")

    def prepare_training_data(
        self,
        audio_path: str,
        model_name: str,
        sample_rate: int = 40000
    ) -> Path:
        """
        Prepare training data directory for RVC

        Args:
            audio_path: Path to cleaned training audio or directory
            model_name: Name for the voice model
            sample_rate: Audio sample rate

        Returns:
            Path to prepared experiment directory
        """
        logger.info(f"Preparing training data for model: {model_name}")

        # Create experiment directory
        exp_dir = self.rvc_dir / "logs" / model_name
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Create dataset directory with audio files
        audio_input = Path(audio_path)
        dataset_dir = self.rvc_dir / "datasets" / model_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        # Copy audio files to dataset directory
        if audio_input.is_file():
            shutil.copy(audio_input, dataset_dir / audio_input.name)
        elif audio_input.is_dir():
            for audio_file in audio_input.glob("*.wav"):
                shutil.copy(audio_file, dataset_dir / audio_file.name)
        else:
            raise ValueError(f"Invalid audio path: {audio_path}")

        logger.info(f"Dataset prepared: {dataset_dir}")
        logger.info(f"Experiment directory: {exp_dir}")

        return exp_dir

    def _create_filelist(self, exp_dir: Path):
        """
        Create filelist.txt for training

        Args:
            exp_dir: Experiment directory
        """
        # RVC expects filelist in format: path|speaker_id|text|lang
        # For voice cloning, we use a single speaker (id=0)
        gt_wavs_dir = exp_dir / "0_gt_wavs"
        filelist_path = exp_dir / "filelist.txt"

        with open(filelist_path, 'w', encoding='utf-8') as f:
            for wav_file in sorted(gt_wavs_dir.glob("*.wav")):
                # Format: path|speaker_id
                f.write(f"{wav_file.relative_to(exp_dir)}|0\n")

        logger.info(f"Created filelist with {len(list(gt_wavs_dir.glob('*.wav')))} files")

    def train_model(
        self,
        model_name: str,
        exp_dir: Path,
        total_epochs: int = 200,
        save_every_epoch: int = 10,
        sample_rate: int = 40000,
        version: str = "v2",
        pretrain_g: str = "",
        pretrain_d: str = "",
        cache_in_gpu: bool = False
    ) -> Path:
        """
        Step 4: Train RVC voice conversion model

        Args:
            model_name: Name of voice model
            exp_dir: Experiment directory
            total_epochs: Total training epochs
            save_every_epoch: Checkpoint save frequency
            sample_rate: Audio sample rate (40000 or 48000)
            version: Model version (v1 or v2)
            pretrain_g: Path to pretrained generator (optional)
            pretrain_d: Path to pretrained discriminator (optional)
            cache_in_gpu: Cache dataset in GPU memory (requires high VRAM)

        Returns:
            Path to trained model checkpoint
        """
        logger.info(f"[4/4] Starting RVC training for: {model_name}")
        logger.info(f"Total epochs: {total_epochs}, Save every: {save_every_epoch}")
        logger.info(f"Batch size: {self.batch_size}, Sample rate: {sample_rate}, Version: {version}")

        # Create filelist
        self._create_filelist(exp_dir)

        # RVC training script (correct path)
        train_script = self.rvc_dir / "infer" / "modules" / "train" / "train.py"

        if not train_script.exists():
            raise ValueError(f"Training script not found: {train_script}")

        # Build training command with RVC's expected arguments
        cmd = [
            sys.executable,
            str(train_script),
            "-se", str(save_every_epoch),  # save_every_epoch
            "-te", str(total_epochs),  # total_epoch
            "-pg", pretrain_g,  # pretrainG
            "-pd", pretrain_d,  # pretrainD
            "-g", str(self.gpu_id),  # gpus (single GPU)
            "-bs", str(self.batch_size),  # batch_size
            "-e", model_name,  # experiment_dir (model name)
            "-sr", str(sample_rate),  # sample_rate
            "-sw", "1",  # save_every_weights (save extracted weights)
            "-v", version,  # version (v1/v2)
            "-f0", "1",  # if_f0 (use pitch information)
            "-l", "1",  # if_latest (only save latest checkpoint)
            "-c", "1" if cache_in_gpu else "0"  # if_cache_data_in_gpu
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
            logger.info(f"Training complete")
            logger.info(f"Output: {result.stdout[-500:]}")  # Last 500 chars
        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e.stderr}")
            raise RuntimeError(f"RVC training failed: {e.stderr}")

        # Find trained model checkpoint
        # RVC saves to weights/{model_name}.pth
        weights_dir = self.rvc_dir / "weights"
        model_path = weights_dir / f"{model_name}.pth"

        if not model_path.exists():
            # Try logs directory as fallback
            alt_path = exp_dir / f"G_{total_epochs}.pth"
            if alt_path.exists():
                logger.warning(f"Model found in logs directory, copying to weights")
                weights_dir.mkdir(exist_ok=True)
                shutil.copy(alt_path, model_path)
            else:
                raise ValueError(
                    f"Trained model not found at {model_path} or {alt_path}\n"
                    f"Check training logs at {exp_dir}"
                )

        logger.info(f"Model saved: {model_path}")
        return model_path

    def train_from_audio(
        self,
        audio_path: str,
        model_name: str,
        total_epochs: int = 200,
        save_every_epoch: int = 10,
        sample_rate: int = 40000,
        version: str = "v2",
        f0_method: str = "rmvpe",
        pretrain_g: str = "",
        pretrain_d: str = ""
    ) -> Path:
        """
        End-to-end training from audio file

        Complete RVC training pipeline:
        1. Prepare dataset directory
        2. Create config file
        3. Preprocess audio (slice and normalize)
        4. Extract pitch (f0) features
        5. Extract HuBERT features
        6. Train generator and discriminator

        Args:
            audio_path: Path to cleaned training audio or directory
            model_name: Name for voice model
            total_epochs: Total training epochs (200+ recommended)
            save_every_epoch: Checkpoint save frequency
            sample_rate: Audio sample rate (40000 or 48000)
            version: Model version (v1 or v2, v2 recommended)
            f0_method: Pitch extraction method (rmvpe/harvest/dio)
            pretrain_g: Path to pretrained generator (optional, speeds up training)
            pretrain_d: Path to pretrained discriminator (optional)

        Returns:
            Path to trained model checkpoint
        """
        logger.info("=" * 60)
        logger.info(f"Starting RVC training pipeline for: {model_name}")
        logger.info("=" * 60)

        # Step 1: Prepare data directories
        exp_dir = self.prepare_training_data(audio_path, model_name, sample_rate)
        dataset_dir = self.rvc_dir / "datasets" / model_name

        # Step 2: Create config
        self._create_config(exp_dir, sample_rate, version)

        # Step 3: Preprocess audio
        self.preprocess_audio(dataset_dir, exp_dir, sample_rate)

        # Step 4: Extract f0 features
        self.extract_f0(exp_dir, method=f0_method)

        # Step 5: Extract HuBERT features
        self.extract_features(exp_dir, version=version)

        # Step 6: Train model
        model_path = self.train_model(
            model_name=model_name,
            exp_dir=exp_dir,
            total_epochs=total_epochs,
            save_every_epoch=save_every_epoch,
            sample_rate=sample_rate,
            version=version,
            pretrain_g=pretrain_g,
            pretrain_d=pretrain_d
        )

        logger.info("=" * 60)
        logger.info(f"Training complete! Model saved to: {model_path}")
        logger.info("=" * 60)

        return model_path


# Test script
if __name__ == "__main__":
    import sys
    import argparse

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Parse arguments
    parser = argparse.ArgumentParser(description="Train RVC voice conversion model")
    parser.add_argument("audio_path", help="Path to training audio file or directory")
    parser.add_argument("model_name", help="Name for the voice model")
    parser.add_argument("--rvc-dir", help="Path to RVC repository", default=None)
    parser.add_argument("--epochs", type=int, default=200, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--sample-rate", type=int, default=40000, choices=[40000, 48000],
                        help="Audio sample rate")
    parser.add_argument("--version", default="v2", choices=["v1", "v2"],
                        help="Model version")
    parser.add_argument("--f0-method", default="rmvpe", choices=["rmvpe", "harvest", "dio"],
                        help="Pitch extraction method")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")

    args = parser.parse_args()

    # Initialize trainer
    trainer = RVCTrainer(
        rvc_dir=args.rvc_dir,
        batch_size=args.batch_size,
        gpu_id=args.gpu
    )

    # Train model
    try:
        model_path = trainer.train_from_audio(
            audio_path=args.audio_path,
            model_name=args.model_name,
            total_epochs=args.epochs,
            sample_rate=args.sample_rate,
            version=args.version,
            f0_method=args.f0_method
        )
        print(f"\n{'='*60}")
        print(f"SUCCESS: Model trained and saved to:")
        print(f"  {model_path}")
        print(f"{'='*60}\n")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        sys.exit(1)

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
from typing import Optional, Dict, Any, Callable
import logging
import torch

logger = logging.getLogger(__name__)

# Import resource monitoring
try:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from backend.metrics import ResourceMonitor, get_gpu_memory_usage, clear_gpu_cache
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    logger.warning("Metrics module not available - GPU monitoring disabled")


class RVCTrainer:
    def __init__(
        self,
        rvc_dir: Optional[str] = None,
        models_dir: str = "outputs/trained_models",
        batch_size: int = 6,  # Reduced from 8 for better memory efficiency
        gpu_id: int = 0,
        use_fp16: bool = True,  # Mixed precision training
        enable_monitoring: bool = True,
        checkpoint_dir: str = "outputs/checkpoints"
    ):
        """
        Initialize RVC trainer with memory optimizations

        Args:
            rvc_dir: Path to RVC repository (default: env var RVC_DIR or "./Retrieval-based-Voice-Conversion-WebUI")
            models_dir: Directory to save trained models
            batch_size: Training batch size (6 for RTX 3070 8GB, can increase to 10 for 17GB VRAM)
            gpu_id: GPU device ID (0 for first GPU)
            use_fp16: Enable mixed precision training (reduces VRAM by ~40%)
            enable_monitoring: Enable GPU/memory monitoring
            checkpoint_dir: Directory for training checkpoints
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
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.gpu_id = gpu_id
        self.use_fp16 = use_fp16
        self.enable_monitoring = enable_monitoring and METRICS_AVAILABLE

        # Initialize resource monitor
        self.monitor = None
        if self.enable_monitoring:
            self.monitor = ResourceMonitor(
                gpu_id=gpu_id,
                vram_alert_threshold_gb=7.0,  # Alert at 7GB for RTX 3070
                ram_alert_threshold_gb=8.0
            )
            logger.info("Resource monitoring enabled")

        # Validate RVC installation
        if not self.rvc_dir.exists():
            raise ValueError(
                f"RVC directory not found: {rvc_dir}\n"
                f"Set RVC_DIR environment variable or clone the repository:\n"
                f"git clone https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI"
            )

        # Validate required scripts exist
        self._validate_rvc_installation()

        # Optimize GPU settings
        if torch.cuda.is_available():
            # Enable TF32 for faster training on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            # Enable cuDNN autotuner
            torch.backends.cudnn.benchmark = True
            logger.info("CUDA optimizations enabled (TF32, cuDNN autotuner)")

        logger.info(
            f"RVC trainer initialized (batch_size={batch_size}, gpu={gpu_id}, "
            f"fp16={use_fp16}, monitoring={self.enable_monitoring})"
        )

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

    def _validate_path(self, path: Path, base_dir: Path = None) -> Path:
        """
        Validate and sanitize file path to prevent directory traversal

        Args:
            path: Path to validate
            base_dir: Optional base directory to restrict path to

        Returns:
            Resolved absolute path

        Raises:
            ValueError: If path contains traversal attempts or is outside base_dir
        """
        # Resolve to absolute path (eliminates .. and symlinks)
        resolved_path = Path(path).resolve()

        # Check for suspicious patterns in string representation
        path_str = str(path)
        if '..' in path_str or path_str.startswith('/') or ':' in path_str[1:3]:
            logger.warning(f"Suspicious path detected: {path}")

        # If base_dir specified, ensure path is within it
        if base_dir:
            base_resolved = Path(base_dir).resolve()
            try:
                resolved_path.relative_to(base_resolved)
            except ValueError:
                raise ValueError(f"Path {path} is outside allowed directory {base_dir}")

        return resolved_path

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

        # SECURITY: Validate paths to prevent command injection
        audio_dir = self._validate_path(audio_dir)
        exp_dir = self._validate_path(exp_dir)

        # Validate numeric parameters
        if not (8000 <= sample_rate <= 48000):
            raise ValueError(f"Invalid sample rate: {sample_rate}")
        if not (1 <= n_processes <= 32):
            raise ValueError(f"Invalid n_processes: {n_processes}")

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

        # SECURITY: Validate paths and parameters
        exp_dir = self._validate_path(exp_dir)

        # Whitelist allowed methods to prevent command injection
        allowed_methods = {"rmvpe", "harvest", "dio", "pm"}
        if method not in allowed_methods:
            raise ValueError(f"Invalid f0 extraction method: {method}. Allowed: {allowed_methods}")

        if not (1 <= n_processes <= 32):
            raise ValueError(f"Invalid n_processes: {n_processes}")

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

        # SECURITY: Validate paths and parameters
        exp_dir = self._validate_path(exp_dir)

        # Whitelist allowed versions
        if version not in {"v1", "v2"}:
            raise ValueError(f"Invalid version: {version}. Allowed: v1, v2")

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

        # SECURITY: Validate model name to prevent path traversal
        # Only allow alphanumeric, underscore, hyphen
        import re
        if not re.match(r'^[a-zA-Z0-9_-]{1,50}$', model_name):
            raise ValueError(f"Invalid model name: {model_name}. Use only alphanumeric, underscore, hyphen (1-50 chars)")

        # Validate sample rate
        if not (8000 <= sample_rate <= 48000):
            raise ValueError(f"Invalid sample rate: {sample_rate}")

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

    def _find_latest_checkpoint(self, exp_dir: Path, model_name: str) -> Optional[Path]:
        """Find the latest checkpoint for resume training"""
        logs_dir = self.rvc_dir / "logs" / model_name
        if not logs_dir.exists():
            return None

        # Look for generator checkpoints (G_*.pth)
        checkpoints = list(logs_dir.glob("G_*.pth"))
        if not checkpoints:
            return None

        # Extract epoch numbers and find latest
        checkpoint_epochs = []
        for ckpt in checkpoints:
            try:
                epoch = int(ckpt.stem.split('_')[1])
                checkpoint_epochs.append((epoch, ckpt))
            except (IndexError, ValueError):
                continue

        if checkpoint_epochs:
            latest_epoch, latest_ckpt = max(checkpoint_epochs, key=lambda x: x[0])
            logger.info(f"Found checkpoint at epoch {latest_epoch}: {latest_ckpt}")
            return latest_ckpt

        return None

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
        cache_in_gpu: bool = False,
        resume_training: bool = True,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Path:
        """
        Step 4: Train RVC voice conversion model with optimizations

        Args:
            model_name: Name of voice model
            exp_dir: Experiment directory
            total_epochs: Total training epochs
            save_every_epoch: Checkpoint save frequency
            sample_rate: Audio sample rate (40000 or 48000)
            version: Model version (v1 or v2)
            pretrain_g: Path to pretrained generator (optional)
            pretrain_d: Path to pretrained discriminator (optional)
            cache_in_gpu: Cache dataset in GPU memory (DISABLED for RTX 3070)
            resume_training: Resume from latest checkpoint if available
            progress_callback: Optional callback for progress updates (epoch, total_epochs)

        Returns:
            Path to trained model checkpoint
        """
        logger.info(f"[4/4] Starting RVC training for: {model_name}")
        logger.info(f"Total epochs: {total_epochs}, Save every: {save_every_epoch}")
        logger.info(f"Batch size: {self.batch_size}, Sample rate: {sample_rate}, Version: {version}")
        logger.info(f"Mixed precision (FP16): {self.use_fp16}")

        # Start monitoring if enabled
        if self.monitor:
            self.monitor.start_operation(f"train_model_{model_name}")

        # MEMORY OPTIMIZATION: Disable GPU caching for RTX 3070
        if cache_in_gpu:
            logger.warning(
                "GPU caching disabled for memory safety on RTX 3070 (8GB VRAM). "
                "Can be enabled for GPUs with >12GB VRAM."
            )
            cache_in_gpu = False

        # Clear GPU cache before training
        if torch.cuda.is_available():
            clear_gpu_cache(self.gpu_id)
            logger.info(f"Initial GPU memory: {get_gpu_memory_usage(self.gpu_id)}")

        # Create filelist
        self._create_filelist(exp_dir)

        # Check for existing checkpoint to resume
        latest_checkpoint = None
        start_epoch = 0
        if resume_training:
            latest_checkpoint = self._find_latest_checkpoint(exp_dir, model_name)
            if latest_checkpoint:
                try:
                    epoch_num = int(latest_checkpoint.stem.split('_')[1])
                    start_epoch = epoch_num
                    pretrain_g = str(latest_checkpoint)
                    # Find corresponding discriminator
                    disc_checkpoint = latest_checkpoint.parent / f"D_{epoch_num}.pth"
                    if disc_checkpoint.exists():
                        pretrain_d = str(disc_checkpoint)
                    logger.info(f"Resuming training from epoch {start_epoch}")
                    logger.info(f"Loading checkpoint: {latest_checkpoint}")
                except Exception as e:
                    logger.warning(f"Failed to parse checkpoint epoch: {e}")

        # RVC training script (correct path)
        train_script = self.rvc_dir / "infer" / "modules" / "train" / "train.py"

        if not train_script.exists():
            raise ValueError(f"Training script not found: {train_script}")

        # Build training command with memory optimizations
        cmd = [
            sys.executable,
            str(train_script),
            "-se", str(save_every_epoch),  # save_every_epoch
            "-te", str(total_epochs),  # total_epoch
            "-pg", pretrain_g,  # pretrainG
            "-pd", pretrain_d,  # pretrainD
            "-g", str(self.gpu_id),  # gpus (single GPU)
            "-bs", str(self.batch_size),  # batch_size (optimized for 8GB)
            "-e", model_name,  # experiment_dir (model name)
            "-sr", str(sample_rate),  # sample_rate
            "-sw", "1",  # save_every_weights (save extracted weights)
            "-v", version,  # version (v1/v2)
            "-f0", "1",  # if_f0 (use pitch information)
            "-l", "0",  # if_latest (keep all checkpoints for resume)
            "-c", "0"  # if_cache_data_in_gpu (DISABLED for memory safety)
        ]

        logger.info(f"Training command: {' '.join(cmd)}")
        logger.info("Starting training subprocess (this may take 30-40 minutes)...")

        # Run training with real-time output
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(self.rvc_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Stream output and monitor resources
            for line in process.stdout:
                print(line, end='')  # Print training output

                # Sample GPU metrics periodically
                if self.monitor and 'Epoch' in line:
                    self.monitor.sample()

                # Parse progress if callback provided
                if progress_callback and 'Epoch' in line:
                    try:
                        # Example: "Epoch: 42/200"
                        parts = line.split('Epoch')
                        if len(parts) > 1:
                            epoch_str = parts[1].strip().split()[0].replace(':', '')
                            current_epoch = int(epoch_str.split('/')[0])
                            progress_callback(current_epoch, total_epochs)
                    except Exception:
                        pass

            process.wait()

            if process.returncode != 0:
                raise subprocess.CalledProcessError(
                    process.returncode,
                    cmd,
                    "Training process failed"
                )

            logger.info(f"Training complete")

        except subprocess.CalledProcessError as e:
            logger.error(f"Training failed: {e}")
            if self.monitor:
                self.monitor.end_operation(success=False, error_message=str(e))
            raise RuntimeError(f"RVC training failed: {e}")
        except Exception as e:
            logger.error(f"Unexpected training error: {e}")
            if self.monitor:
                self.monitor.end_operation(success=False, error_message=str(e))
            raise

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

        # Clear GPU cache after training
        if torch.cuda.is_available():
            clear_gpu_cache(self.gpu_id)
            logger.info(f"Final GPU memory: {get_gpu_memory_usage(self.gpu_id)}")

        # End monitoring
        if self.monitor:
            metrics = self.monitor.end_operation(success=True)
            # Save metrics
            metrics_path = self.checkpoint_dir / f"{model_name}_training_metrics.json"
            self.monitor.save_metrics(metrics, str(metrics_path))

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

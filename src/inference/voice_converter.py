"""
RVC Voice Conversion Inference with Model Caching
Converts target audio to user's voice with memory optimizations
"""

import sys
import torch
import torchaudio
from pathlib import Path
import logging
from typing import Optional, Dict, Any
from threading import Lock
import weakref
import gc

logger = logging.getLogger(__name__)

# Import resource monitoring
try:
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from backend.metrics import ResourceMonitor, get_gpu_memory_usage, clear_gpu_cache
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    logger.warning("Metrics module not available - GPU monitoring disabled")


class ModelCache:
    """
    LRU cache for RVC models to avoid reloading

    Features:
    - Weak references to allow garbage collection
    - Configurable max cache size
    - Automatic GPU memory management
    """

    def __init__(self, max_size: int = 3, gpu_id: int = 0):
        """
        Initialize model cache

        Args:
            max_size: Maximum number of models to cache
            gpu_id: GPU device ID
        """
        self.max_size = max_size
        self.gpu_id = gpu_id
        self.cache: Dict[str, Any] = {}
        self.access_order = []  # Track LRU
        self.lock = Lock()

        logger.info(f"Model cache initialized (max_size={max_size}, gpu={gpu_id})")

    def get(self, model_path: str) -> Optional[Any]:
        """Get model from cache if available"""
        with self.lock:
            if model_path in self.cache:
                # Move to end (most recently used)
                self.access_order.remove(model_path)
                self.access_order.append(model_path)
                logger.info(f"Cache HIT: {Path(model_path).name}")
                return self.cache[model_path]

            logger.info(f"Cache MISS: {Path(model_path).name}")
            return None

    def put(self, model_path: str, model: Any):
        """Add model to cache with LRU eviction"""
        with self.lock:
            # If already cached, update access order
            if model_path in self.cache:
                self.access_order.remove(model_path)
                self.access_order.append(model_path)
                return

            # Evict least recently used if cache is full
            while len(self.cache) >= self.max_size:
                lru_path = self.access_order.pop(0)
                evicted_model = self.cache.pop(lru_path)
                del evicted_model
                gc.collect()
                if torch.cuda.is_available():
                    clear_gpu_cache(self.gpu_id)
                logger.info(f"Evicted from cache: {Path(lru_path).name}")

            # Add new model
            self.cache[model_path] = model
            self.access_order.append(model_path)
            logger.info(f"Cached model: {Path(model_path).name} ({len(self.cache)}/{self.max_size})")

    def clear(self):
        """Clear all cached models"""
        with self.lock:
            for model in self.cache.values():
                del model
            self.cache.clear()
            self.access_order.clear()
            gc.collect()
            if torch.cuda.is_available():
                clear_gpu_cache(self.gpu_id)
            logger.info("Model cache cleared")

    def size(self) -> int:
        """Get current cache size"""
        return len(self.cache)


# Global model cache (shared across instances)
_global_cache = None
_cache_lock = Lock()


def get_global_cache(max_size: int = 3, gpu_id: int = 0) -> ModelCache:
    """Get or create global model cache"""
    global _global_cache
    with _cache_lock:
        if _global_cache is None:
            _global_cache = ModelCache(max_size=max_size, gpu_id=gpu_id)
        return _global_cache


class VoiceConverter:
    def __init__(
        self,
        rvc_dir: str = "Retrieval-based-Voice-Conversion-WebUI",
        model_path: str = None,
        gpu_id: int = 0,
        use_cache: bool = True,
        enable_monitoring: bool = True
    ):
        """
        Initialize voice converter with model caching

        Args:
            rvc_dir: Path to RVC repository
            model_path: Path to trained model checkpoint (lazy loaded)
            gpu_id: GPU device ID
            use_cache: Enable model caching
            enable_monitoring: Enable GPU monitoring
        """
        self.rvc_dir = Path(rvc_dir)
        self.model_path = model_path
        self.gpu_id = gpu_id
        self.use_cache = use_cache
        self.vc = None
        self.current_model = None

        # Initialize cache
        if use_cache:
            self.cache = get_global_cache(max_size=3, gpu_id=gpu_id)
        else:
            self.cache = None

        # Initialize monitoring
        self.monitor = None
        if enable_monitoring and METRICS_AVAILABLE:
            self.monitor = ResourceMonitor(
                gpu_id=gpu_id,
                vram_alert_threshold_gb=7.0,
                ram_alert_threshold_gb=8.0
            )

        if not self.rvc_dir.exists():
            raise ValueError(f"RVC directory not found: {rvc_dir}")

        # Lazy load model if path provided
        if model_path:
            self.load_model(model_path)

        logger.info(
            f"Voice converter initialized (gpu={gpu_id}, cache={use_cache}, "
            f"monitoring={self.monitor is not None})"
        )

    def load_model(self, model_path: str, force_reload: bool = False):
        """
        Load trained RVC model with caching

        Args:
            model_path: Path to model checkpoint
            force_reload: Force reload even if cached
        """
        model_path = str(Path(model_path).resolve())

        # Check if already loaded
        if not force_reload and self.current_model == model_path and self.vc is not None:
            logger.info(f"Model already loaded: {Path(model_path).name}")
            return

        logger.info(f"Loading model: {model_path}")

        # Start monitoring
        if self.monitor:
            self.monitor.start_operation(f"load_model_{Path(model_path).stem}")

        try:
            # Check cache first
            if self.use_cache and not force_reload:
                cached_vc = self.cache.get(model_path)
                if cached_vc is not None:
                    self.vc = cached_vc
                    self.current_model = model_path
                    if self.monitor:
                        self.monitor.end_operation(success=True)
                    return

            # Add RVC to path
            if str(self.rvc_dir) not in sys.path:
                sys.path.append(str(self.rvc_dir))

            # Clear GPU cache before loading
            if torch.cuda.is_available():
                clear_gpu_cache(self.gpu_id)

            # Import RVC inference modules
            from infer.modules.vc.modules import VC

            # Initialize VC
            self.vc = VC()
            self.vc.get_vc(model_path)
            self.current_model = model_path

            # Add to cache
            if self.use_cache:
                self.cache.put(model_path, self.vc)

            logger.info("Model loaded successfully")

            # Log GPU memory
            if torch.cuda.is_available():
                mem_usage = get_gpu_memory_usage(self.gpu_id)
                logger.info(f"GPU memory after load: {mem_usage}")

            # End monitoring
            if self.monitor:
                self.monitor.end_operation(success=True)

        except Exception as e:
            logger.error(f"Failed to load RVC model: {e}")
            logger.error("Make sure RVC repository is properly set up")
            if self.monitor:
                self.monitor.end_operation(success=False, error_message=str(e))
            raise

    def convert_voice(
        self,
        input_audio: str,
        output_audio: str,
        pitch_shift: int = 0,
        index_rate: float = 0.75,
        f0_method: str = "rmvpe",
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33
    ) -> str:
        """
        Convert audio to trained voice with optimizations

        Args:
            input_audio: Target audio (e.g., Michael Jackson song)
            output_audio: Output path for converted audio
            pitch_shift: Pitch adjustment in semitones
            index_rate: Feature retrieval strength (0.0-1.0)
            f0_method: Pitch extraction method (rmvpe/harvest/dio)
            filter_radius: Median filtering for pitch
            resample_sr: Resample output (0 = keep original)
            rms_mix_rate: Volume envelope mix
            protect: Protect voiceless consonants (0.0-0.5)

        Returns:
            Path to converted audio
        """
        if self.vc is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        logger.info(f"Converting: {input_audio}")
        logger.info(f"Pitch shift: {pitch_shift}, Index rate: {index_rate}, Method: {f0_method}")

        # Start monitoring
        if self.monitor:
            self.monitor.start_operation("voice_conversion")

        try:
            # Clear GPU cache before conversion
            if torch.cuda.is_available():
                clear_gpu_cache(self.gpu_id)

            # Run RVC conversion
            result = self.vc.vc_single(
                sid=0,
                input_audio_path=input_audio,
                f0_up_key=pitch_shift,
                f0_file=None,
                f0_method=f0_method,
                file_index="",
                index_rate=index_rate,
                filter_radius=filter_radius,
                resample_sr=resample_sr,
                rms_mix_rate=rms_mix_rate,
                protect=protect
            )

            # Save output
            output_audio_data, sr = result
            output_tensor = torch.from_numpy(output_audio_data).unsqueeze(0)
            torchaudio.save(output_audio, output_tensor, sr)

            logger.info(f"Conversion complete: {output_audio}")

            # Clear GPU cache after conversion
            if torch.cuda.is_available():
                clear_gpu_cache(self.gpu_id)
                mem_usage = get_gpu_memory_usage(self.gpu_id)
                logger.info(f"GPU memory after conversion: {mem_usage}")

            # End monitoring
            if self.monitor:
                self.monitor.end_operation(success=True)

            return output_audio

        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            if self.monitor:
                self.monitor.end_operation(success=False, error_message=str(e))
            raise

    def batch_convert(
        self,
        input_files: list,
        output_dir: str,
        **convert_kwargs
    ) -> list:
        """
        Convert multiple audio files in batch

        Args:
            input_files: List of input audio paths
            output_dir: Output directory
            **convert_kwargs: Arguments passed to convert_voice()

        Returns:
            List of output file paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for i, input_file in enumerate(input_files):
            try:
                input_path = Path(input_file)
                output_path = output_dir / f"{input_path.stem}_converted{input_path.suffix}"

                logger.info(f"Batch converting {i+1}/{len(input_files)}: {input_file}")

                result = self.convert_voice(
                    str(input_file),
                    str(output_path),
                    **convert_kwargs
                )
                results.append(result)

            except Exception as e:
                logger.error(f"Failed to convert {input_file}: {e}")
                results.append(None)

        success_count = sum(1 for r in results if r is not None)
        logger.info(f"Batch conversion complete: {success_count}/{len(input_files)} succeeded")

        return results

    def cleanup(self):
        """Clean up GPU memory and resources"""
        if self.vc and not self.use_cache:
            del self.vc
            self.vc = None
            self.current_model = None

        if torch.cuda.is_available():
            clear_gpu_cache(self.gpu_id)

        gc.collect()
        logger.info("Voice converter cleanup complete")

    @staticmethod
    def clear_cache():
        """Clear global model cache"""
        global _global_cache
        if _global_cache:
            _global_cache.clear()


# Test script
if __name__ == "__main__":
    import sys
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="RVC Voice Conversion with Model Caching")
    parser.add_argument("model_path", help="Path to trained RVC model")
    parser.add_argument("input_audio", help="Input audio file")
    parser.add_argument("output_audio", help="Output audio file")
    parser.add_argument("--pitch", type=int, default=0, help="Pitch shift in semitones")
    parser.add_argument("--index-rate", type=float, default=0.75, help="Index rate (0.0-1.0)")
    parser.add_argument("--f0-method", default="rmvpe", choices=["rmvpe", "harvest", "dio"],
                        help="Pitch extraction method")
    parser.add_argument("--no-cache", action="store_true", help="Disable model caching")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")

    args = parser.parse_args()

    try:
        converter = VoiceConverter(
            model_path=args.model_path,
            gpu_id=args.gpu,
            use_cache=not args.no_cache
        )

        output = converter.convert_voice(
            args.input_audio,
            args.output_audio,
            pitch_shift=args.pitch,
            index_rate=args.index_rate,
            f0_method=args.f0_method
        )

        print(f"\n{'='*60}")
        print(f"Voice conversion complete!")
        print(f"Output: {output}")
        print(f"{'='*60}\n")

    except Exception as e:
        logger.error(f"Conversion failed: {e}", exc_info=True)
        sys.exit(1)

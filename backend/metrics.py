"""
Performance Metrics and Resource Monitoring
Tracks GPU/CPU/Memory usage during ML operations
"""

import time
import logging
import psutil
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import json

logger = logging.getLogger(__name__)

# Try importing GPU monitoring libraries
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available - GPU metrics disabled")

try:
    import pynvml
    NVML_AVAILABLE = True
    pynvml.nvmlInit()
except (ImportError, Exception) as e:
    NVML_AVAILABLE = False
    logger.warning(f"NVML not available - detailed GPU metrics disabled: {e}")


@dataclass
class ResourceSnapshot:
    """Single point-in-time resource measurement"""
    timestamp: float
    cpu_percent: float
    ram_used_gb: float
    ram_percent: float
    gpu_id: Optional[int] = None
    gpu_util_percent: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_memory_percent: Optional[float] = None
    gpu_temperature: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics for an operation"""
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None

    # Peak resource usage
    peak_cpu_percent: float = 0.0
    peak_ram_gb: float = 0.0
    peak_gpu_memory_gb: Optional[float] = None
    peak_gpu_util_percent: Optional[float] = None
    peak_gpu_temperature: Optional[float] = None

    # Average resource usage
    avg_cpu_percent: float = 0.0
    avg_ram_gb: float = 0.0
    avg_gpu_memory_gb: Optional[float] = None
    avg_gpu_util_percent: Optional[float] = None

    # Snapshots
    snapshots: List[ResourceSnapshot] = None

    # Status
    success: bool = True
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.snapshots is None:
            self.snapshots = []

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['snapshots'] = [s.to_dict() for s in self.snapshots]
        return data


class ResourceMonitor:
    """
    Real-time resource monitoring for ML operations

    Features:
    - CPU and RAM tracking
    - GPU utilization and VRAM tracking
    - Temperature monitoring
    - Peak usage detection
    - Automatic alerts for resource limits
    """

    def __init__(
        self,
        gpu_id: int = 0,
        sample_interval: float = 1.0,
        vram_alert_threshold_gb: float = 7.0,
        ram_alert_threshold_gb: float = 8.0
    ):
        """
        Initialize resource monitor

        Args:
            gpu_id: GPU device ID to monitor
            sample_interval: Sampling interval in seconds
            vram_alert_threshold_gb: Alert if GPU memory exceeds this (GB)
            ram_alert_threshold_gb: Alert if RAM exceeds this (GB)
        """
        self.gpu_id = gpu_id
        self.sample_interval = sample_interval
        self.vram_alert_threshold_gb = vram_alert_threshold_gb
        self.ram_alert_threshold_gb = ram_alert_threshold_gb

        self.gpu_handle = None
        if NVML_AVAILABLE:
            try:
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                gpu_name = pynvml.nvmlDeviceGetName(self.gpu_handle)
                logger.info(f"GPU monitoring initialized: {gpu_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize GPU monitoring: {e}")

        self.current_metrics: Optional[PerformanceMetrics] = None
        self.monitoring = False

        logger.info(
            f"Resource monitor ready (GPU {gpu_id}, "
            f"VRAM alert: {vram_alert_threshold_gb}GB, "
            f"RAM alert: {ram_alert_threshold_gb}GB)"
        )

    def get_current_snapshot(self) -> ResourceSnapshot:
        """Capture current resource usage"""
        # CPU and RAM
        cpu_percent = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        ram_used_gb = mem.used / (1024 ** 3)
        ram_percent = mem.percent

        # GPU metrics
        gpu_util = None
        gpu_mem_used_gb = None
        gpu_mem_total_gb = None
        gpu_mem_percent = None
        gpu_temp = None

        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                gpu_mem_used = torch.cuda.memory_allocated(self.gpu_id)
                gpu_mem_reserved = torch.cuda.memory_reserved(self.gpu_id)
                gpu_mem_used_gb = gpu_mem_reserved / (1024 ** 3)  # Use reserved as it's more accurate

                # Try to get total memory
                if self.gpu_handle:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                    gpu_mem_total_gb = mem_info.total / (1024 ** 3)
                    gpu_mem_percent = (gpu_mem_used_gb / gpu_mem_total_gb) * 100

                    # Get utilization
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                    gpu_util = util.gpu

                    # Get temperature
                    gpu_temp = pynvml.nvmlDeviceGetTemperature(
                        self.gpu_handle,
                        pynvml.NVML_TEMPERATURE_GPU
                    )
            except Exception as e:
                logger.debug(f"Failed to get GPU metrics: {e}")

        return ResourceSnapshot(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            ram_used_gb=ram_used_gb,
            ram_percent=ram_percent,
            gpu_id=self.gpu_id,
            gpu_util_percent=gpu_util,
            gpu_memory_used_gb=gpu_mem_used_gb,
            gpu_memory_total_gb=gpu_mem_total_gb,
            gpu_memory_percent=gpu_mem_percent,
            gpu_temperature=gpu_temp
        )

    def check_resource_alerts(self, snapshot: ResourceSnapshot):
        """Check if resource usage exceeds thresholds"""
        # VRAM alert
        if snapshot.gpu_memory_used_gb and snapshot.gpu_memory_used_gb > self.vram_alert_threshold_gb:
            logger.warning(
                f"⚠️ GPU MEMORY ALERT: {snapshot.gpu_memory_used_gb:.2f}GB / "
                f"{snapshot.gpu_memory_total_gb:.2f}GB "
                f"({snapshot.gpu_memory_percent:.1f}%) - "
                f"Threshold: {self.vram_alert_threshold_gb}GB"
            )

        # RAM alert
        if snapshot.ram_used_gb > self.ram_alert_threshold_gb:
            logger.warning(
                f"⚠️ RAM ALERT: {snapshot.ram_used_gb:.2f}GB "
                f"({snapshot.ram_percent:.1f}%) - "
                f"Threshold: {self.ram_alert_threshold_gb}GB"
            )

    def start_operation(self, operation_name: str) -> PerformanceMetrics:
        """Start monitoring a new operation"""
        self.current_metrics = PerformanceMetrics(
            operation_name=operation_name,
            start_time=time.time()
        )
        self.monitoring = True

        logger.info(f"📊 Started monitoring: {operation_name}")
        return self.current_metrics

    def sample(self):
        """Take a resource snapshot and update metrics"""
        if not self.monitoring or not self.current_metrics:
            return

        snapshot = self.get_current_snapshot()
        self.current_metrics.snapshots.append(snapshot)

        # Update peak values
        self.current_metrics.peak_cpu_percent = max(
            self.current_metrics.peak_cpu_percent,
            snapshot.cpu_percent
        )
        self.current_metrics.peak_ram_gb = max(
            self.current_metrics.peak_ram_gb,
            snapshot.ram_used_gb
        )

        if snapshot.gpu_memory_used_gb:
            if self.current_metrics.peak_gpu_memory_gb is None:
                self.current_metrics.peak_gpu_memory_gb = snapshot.gpu_memory_used_gb
            else:
                self.current_metrics.peak_gpu_memory_gb = max(
                    self.current_metrics.peak_gpu_memory_gb,
                    snapshot.gpu_memory_used_gb
                )

        if snapshot.gpu_util_percent:
            if self.current_metrics.peak_gpu_util_percent is None:
                self.current_metrics.peak_gpu_util_percent = snapshot.gpu_util_percent
            else:
                self.current_metrics.peak_gpu_util_percent = max(
                    self.current_metrics.peak_gpu_util_percent,
                    snapshot.gpu_util_percent
                )

        if snapshot.gpu_temperature:
            if self.current_metrics.peak_gpu_temperature is None:
                self.current_metrics.peak_gpu_temperature = snapshot.gpu_temperature
            else:
                self.current_metrics.peak_gpu_temperature = max(
                    self.current_metrics.peak_gpu_temperature,
                    snapshot.gpu_temperature
                )

        # Check alerts
        self.check_resource_alerts(snapshot)

    def end_operation(self, success: bool = True, error_message: str = None) -> PerformanceMetrics:
        """Stop monitoring and compute final metrics"""
        if not self.current_metrics:
            logger.warning("No active monitoring session")
            return None

        self.monitoring = False
        self.current_metrics.end_time = time.time()
        self.current_metrics.duration_seconds = (
            self.current_metrics.end_time - self.current_metrics.start_time
        )
        self.current_metrics.success = success
        self.current_metrics.error_message = error_message

        # Compute averages
        if self.current_metrics.snapshots:
            n = len(self.current_metrics.snapshots)

            self.current_metrics.avg_cpu_percent = sum(
                s.cpu_percent for s in self.current_metrics.snapshots
            ) / n

            self.current_metrics.avg_ram_gb = sum(
                s.ram_used_gb for s in self.current_metrics.snapshots
            ) / n

            gpu_mem_samples = [
                s.gpu_memory_used_gb for s in self.current_metrics.snapshots
                if s.gpu_memory_used_gb is not None
            ]
            if gpu_mem_samples:
                self.current_metrics.avg_gpu_memory_gb = sum(gpu_mem_samples) / len(gpu_mem_samples)

            gpu_util_samples = [
                s.gpu_util_percent for s in self.current_metrics.snapshots
                if s.gpu_util_percent is not None
            ]
            if gpu_util_samples:
                self.current_metrics.avg_gpu_util_percent = sum(gpu_util_samples) / len(gpu_util_samples)

        # Log summary
        logger.info(
            f"✅ Completed: {self.current_metrics.operation_name} "
            f"({self.current_metrics.duration_seconds:.1f}s)"
        )
        logger.info(
            f"   Peak RAM: {self.current_metrics.peak_ram_gb:.2f}GB "
            f"({self.current_metrics.avg_ram_gb:.2f}GB avg)"
        )
        if self.current_metrics.peak_gpu_memory_gb:
            logger.info(
                f"   Peak VRAM: {self.current_metrics.peak_gpu_memory_gb:.2f}GB "
                f"({self.current_metrics.avg_gpu_memory_gb:.2f}GB avg)"
            )
        if self.current_metrics.peak_gpu_util_percent:
            logger.info(
                f"   Peak GPU: {self.current_metrics.peak_gpu_util_percent:.1f}% "
                f"({self.current_metrics.avg_gpu_util_percent:.1f}% avg)"
            )

        metrics = self.current_metrics
        self.current_metrics = None
        return metrics

    def save_metrics(self, metrics: PerformanceMetrics, output_path: str):
        """Save metrics to JSON file"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)

        logger.info(f"Metrics saved to: {output_path}")

    def cleanup(self):
        """Clean up resources"""
        self.monitoring = False
        if NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except:
                pass


class ContextMonitor:
    """
    Context manager for easy resource monitoring

    Usage:
        monitor = ResourceMonitor()
        with ContextMonitor(monitor, "training"):
            # Your code here
            pass
    """

    def __init__(
        self,
        monitor: ResourceMonitor,
        operation_name: str,
        sample_interval: float = 2.0
    ):
        self.monitor = monitor
        self.operation_name = operation_name
        self.sample_interval = sample_interval
        self.metrics = None

    def __enter__(self):
        self.metrics = self.monitor.start_operation(self.operation_name)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        success = exc_type is None
        error_msg = str(exc_val) if exc_val else None
        self.metrics = self.monitor.end_operation(success=success, error_message=error_msg)
        return False  # Don't suppress exceptions


# Convenience functions
def get_gpu_memory_usage(gpu_id: int = 0) -> Dict[str, float]:
    """Get current GPU memory usage in GB"""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    allocated = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)
    max_allocated = torch.cuda.max_memory_allocated(gpu_id) / (1024 ** 3)

    return {
        "allocated_gb": allocated,
        "reserved_gb": reserved,
        "max_allocated_gb": max_allocated
    }


def clear_gpu_cache(gpu_id: int = 0):
    """Clear GPU memory cache"""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(f"GPU {gpu_id} cache cleared")


def check_gpu_available(min_memory_gb: float = 2.0, gpu_id: int = 0) -> bool:
    """Check if GPU has sufficient free memory"""
    if not TORCH_AVAILABLE or not torch.cuda.is_available():
        return False

    try:
        if NVML_AVAILABLE:
            handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_gb = mem_info.free / (1024 ** 3)
            return free_gb >= min_memory_gb
        else:
            # Fallback: assume available if CUDA is available
            return True
    except Exception as e:
        logger.warning(f"Failed to check GPU memory: {e}")
        return True


# Test script
if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=== GPU Resource Monitor Test ===\n")

    # Test basic monitoring
    monitor = ResourceMonitor(gpu_id=0)

    # Get current snapshot
    snapshot = monitor.get_current_snapshot()
    print(f"Current Resource Usage:")
    print(f"  CPU: {snapshot.cpu_percent:.1f}%")
    print(f"  RAM: {snapshot.ram_used_gb:.2f}GB ({snapshot.ram_percent:.1f}%)")
    if snapshot.gpu_memory_used_gb:
        print(f"  GPU Memory: {snapshot.gpu_memory_used_gb:.2f}GB / {snapshot.gpu_memory_total_gb:.2f}GB")
        print(f"  GPU Util: {snapshot.gpu_util_percent}%")
        print(f"  GPU Temp: {snapshot.gpu_temperature}°C")

    # Test operation monitoring
    print("\n--- Testing Operation Monitoring ---")
    metrics = monitor.start_operation("test_operation")

    # Simulate work
    for i in range(5):
        time.sleep(1)
        monitor.sample()
        print(f"Sample {i+1}/5")

    metrics = monitor.end_operation(success=True)

    # Save metrics
    monitor.save_metrics(metrics, "test_metrics.json")
    print(f"\nMetrics saved to: test_metrics.json")

    monitor.cleanup()

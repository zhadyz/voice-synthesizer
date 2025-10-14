"""
Logging Configuration for Voice Cloning Pipeline
Sets up file and console logging with proper formatting
"""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(
    log_file: str = None,
    log_level: int = logging.INFO,
    console_output: bool = True
):
    """
    Configure logging for the pipeline

    Args:
        log_file: Path to log file (default: pipeline_YYYYMMDD_HHMMSS.log)
        log_level: Logging level (INFO, DEBUG, WARNING, ERROR)
        console_output: Whether to output to console
    """
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Generate log filename if not provided
    if log_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = logs_dir / f"pipeline_{timestamp}.log"

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    root_logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    logging.info(f"Logging initialized - Log file: {log_file}")
    return log_file


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module"""
    return logging.getLogger(name)


# Example usage
if __name__ == "__main__":
    log_file = setup_logging()
    logger = get_logger(__name__)

    logger.info("Logging test - INFO")
    logger.warning("Logging test - WARNING")
    logger.error("Logging test - ERROR")

    print(f"\nLog file created: {log_file}")

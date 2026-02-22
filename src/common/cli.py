"""Common CLI utilities for all scripts."""

import argparse
import logging
import os
import signal
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


_shutdown_requested = False


def signal_handler(signum, frame):
    """Handle SIGINT (Ctrl+C) by setting shutdown flag."""
    global _shutdown_requested
    _shutdown_requested = True
    print("\nShutdown requested. Finishing current task and exiting gracefully...", file=sys.stderr)


def setup_shutdown_handler():
    """Register signal handler for graceful shutdown on Ctrl+C."""
    signal.signal(signal.SIGINT, signal_handler)


def get_shutdown_flag() -> callable:
    """Return a callable that returns True when shutdown is requested."""
    return lambda: _shutdown_requested


def setup_logging(
    output_dir: Path,
    verbose: bool = False,
    log_filename: str = "log.txt"
) -> logging.Logger:
    """Setup logging configuration.
    
    Args:
        output_dir: Output directory for log file
        verbose: Enable DEBUG level logging
        log_filename: Name of the log file
    
    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO
    log_path = output_dir / log_filename
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode='a')
        ]
    )
    
    return logging.getLogger(__name__)


def save_log(output_dir: Path, args, log_filename: str = "log.txt") -> None:
    """Save command log to file.
    
    Args:
        output_dir: Output directory
        args: Parsed arguments (from argparse)
        log_filename: Name of the log file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = output_dir / log_filename
    with open(log_path, 'a') as f:
        f.write(f"Command: {' '.join(sys.argv)}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write("Arguments:\n")
        for key, value in vars(args).items():
            f.write(f"  {key}: {value}\n")
        f.write("\n")


def parse_args(parser: argparse.ArgumentParser, args: Optional[list] = None) -> argparse.Namespace:
    """Parse arguments with graceful shutdown support.
    
    Args:
        parser: Pre-configured ArgumentParser instance
        args: List of arguments. If None, uses sys.argv
    
    Returns:
        Parsed arguments
    """
    return parser.parse_args(args)


def validate_directory(path: Path, must_exist: bool = True, must_be_dir: bool = True) -> Path:
    """Validate directory path.
    
    Args:
        path: Path to validate
        must_exist: Whether directory must exist
        must_be_dir: Whether path must be a directory
    
    Returns:
        Validated Path object
    
    Raises:
        FileNotFoundError: If directory doesn't exist and must_exist is True
        NotADirectoryError: If path is not a directory and must_be_dir is True
    """
    path = Path(path)
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    
    if must_be_dir and not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")
    
    return path


def validate_file(path: Path, must_exist: bool = True) -> Path:
    """Validate file path.
    
    Args:
        path: Path to validate
        must_exist: Whether file must exist
    
    Returns:
        Validated Path object
    
    Raises:
        FileNotFoundError: If file doesn't exist and must_exist is True
    """
    path = Path(path)
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"File does not exist: {path}")
    
    if path.is_dir():
        raise IsADirectoryError(f"Path is a directory, not a file: {path}")
    
    return path


def validate_image_extensions(files: list, extensions: Optional[set] = None) -> list:
    """Filter files by valid image extensions.
    
    Args:
        files: List of file paths
        extensions: Set of valid extensions. Defaults to common image formats
    
    Returns:
        Filtered list of image files
    """
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    
    return [f for f in files if Path(f).suffix.lower() in extensions]

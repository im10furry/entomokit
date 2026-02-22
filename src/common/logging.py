"""Common logging utilities for all scripts."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
    log_filename: str = "log.txt"
) -> logging.Logger:
    """Setup logger with optional file output.
    
    Args:
        name: Logger name
        output_dir: Directory for log file. If None, only console output
        verbose: Enable DEBUG level logging
        log_filename: Name of the log file
    
    Returns:
        Configured logger instance
    """
    level = logging.DEBUG if verbose else logging.INFO
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if output_dir:
        log_path = Path(output_dir) / log_filename
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, mode='a')
        file_handler.setLevel(level)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_command(logger: logging.Logger, args) -> None:
    """Log command and arguments.
    
    Args:
        logger: Logger instance
        args: Parsed arguments from argparse
    """
    import sys
    from datetime import datetime
    
    logger.info(f"Command: {' '.join(sys.argv)}")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info("Arguments:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")


def save_command_log(output_dir: Path, args, log_filename: str = "log.txt") -> None:
    """Save command log to file (without timestamp prefix).
    
    Args:
        output_dir: Output directory
        args: Parsed arguments
        log_filename: Name of the log file
    """
    import sys
    from datetime import datetime
    
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

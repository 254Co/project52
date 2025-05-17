"""
Logging configuration for the Chen3 package.

This module provides a centralized logging configuration for the entire package,
with different log levels and formatting for different components. It implements
a custom logger class that supports both console and file logging with different
formatters for each output type.

The module creates a default logger instance that can be imported and used
throughout the package.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# Create logger
logger = logging.getLogger("chen3")

# Set default level
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler("chen3.log")

# Create formatters
console_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
file_formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
)

# Set formatters
console_handler.setFormatter(console_formatter)
file_handler.setFormatter(file_formatter)

# Add handlers
logger.addHandler(console_handler)
logger.addHandler(file_handler)


def configure_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_dir: Optional[str] = None,
) -> None:
    """
    Configure logging for the Chen3 package.

    Args:
        level: Logging level (default: logging.INFO)
        log_file: Path to log file (default: chen3.log)
        log_dir: Directory for log files (default: current directory)

    Example:
        >>> from chen3.utils.logging import configure_logging
        >>> import logging
        >>>
        >>> # Configure logging with default settings
        >>> configure_logging()
        >>>
        >>> # Configure logging with custom settings
        >>> configure_logging(
        ...     level=logging.DEBUG,
        ...     log_file="custom.log",
        ...     log_dir="/path/to/logs"
        ... )
    """
    # Set logging level
    logger.setLevel(level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Create file handler if log file is specified
    if log_file:
        if log_dir:
            log_path = Path(log_dir) / log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            log_path = Path(log_file)

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    # Log configuration
    logger.info(f"Logging configured with level {logging.getLevelName(level)}")
    if log_file:
        logger.info(f"Log file: {log_path}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    Args:
        name: Module name

    Returns:
        logging.Logger: Logger instance

    Example:
        >>> from chen3.utils.logging import get_logger
        >>>
        >>> # Get logger for a module
        >>> logger = get_logger("chen3.model")
        >>> logger.info("Model initialized")
    """
    return logging.getLogger(f"chen3.{name}")

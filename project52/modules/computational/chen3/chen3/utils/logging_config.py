"""
Logging Configuration for the Chen3 Model

This module provides a comprehensive logging configuration for the Chen3 package,
including different log levels, formatters, and handlers for various output destinations.
It implements a flexible logging system that supports:
- Console and file logging with different formatters
- Configurable log levels for different handlers
- Automatic log file rotation with timestamps
- Detailed and simple log formats for different use cases

The module creates a default logger instance that can be imported and used
throughout the package.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union


class Chen3Logger:
    """
    Logger configuration for the Chen3 model.

    This class provides a flexible logging configuration that supports both
    console and file logging with different formatters and log levels. It
    automatically creates log directories if they don't exist and handles
    log file rotation.

    Attributes:
        logger (logging.Logger): The configured Python logger instance
    """

    def __init__(
        self,
        name: str = "chen3",
        level: Union[int, str] = logging.INFO,
        log_file: Optional[Union[str, Path]] = None,
        console: bool = True,
        file_level: Optional[Union[int, str]] = None,
        console_level: Optional[Union[int, str]] = None,
    ):
        """
        Initialize the Chen3 logger with specified configuration.

        Args:
            name (str): Name of the logger, defaults to "chen3"
            level (Union[int, str]): Default logging level for all handlers
            log_file (Optional[Union[str, Path]]): Path to log file
            console (bool): Whether to enable console logging
            file_level (Optional[Union[int, str]]): Logging level for file handler
            console_level (Optional[Union[int, str]]): Logging level for console handler

        Note:
            If file_level or console_level are not specified, they default to
            the main level parameter. The file handler uses a detailed formatter
            while the console handler uses a simpler formatter for readability.
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Clear existing handlers to avoid duplicate logging
        self.logger.handlers = []

        # Create formatters with different levels of detail
        detailed_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )
        simple_formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )

        # Add console handler if requested
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(console_level or level)
            console_handler.setFormatter(simple_formatter)
            self.logger.addHandler(console_handler)

        # Add file handler if log file specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(file_level or level)
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)

    def get_logger(self) -> logging.Logger:
        """
        Get the configured logger instance.

        Returns:
            logging.Logger: The configured Python logger instance
        """
        return self.logger


def setup_logging(
    level: Union[int, str] = logging.INFO,
    log_dir: Optional[Union[str, Path]] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Set up logging for the Chen3 model with default configuration.

    This function creates a logger with the following features:
    - Console logging with simple format
    - File logging with detailed format
    - Automatic log file rotation using timestamps
    - Default log directory in user's home folder

    Args:
        level (Union[int, str]): Logging level (default: logging.INFO)
        log_dir (Optional[Union[str, Path]]): Directory for log files
        console (bool): Whether to enable console logging (default: True)

    Returns:
        logging.Logger: Configured logger instance

    Note:
        If log_dir is not specified, logs are stored in ~/.chen3/logs/
        with filenames including timestamps for easy rotation.
    """
    if log_dir is None:
        log_dir = Path.home() / ".chen3" / "logs"

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"chen3_{timestamp}.log"

    logger = Chen3Logger(level=level, log_file=log_file, console=console)

    return logger.get_logger()


# Create default logger instance for use throughout the package
logger = setup_logging()

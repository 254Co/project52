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


class ChenLogger:
    """
    Custom logger for the Chen3 package with different log levels and formatting.

    This class provides a wrapper around Python's logging module with pre-configured
    formatters and handlers for both console and file output. It supports different
    log levels and can be configured to write logs to both console and file.

    Attributes:
        logger (logging.Logger): The underlying Python logger instance
    """

    def __init__(
        self,
        name: str = "chen3",
        level: int = logging.INFO,
        log_file: Optional[Path] = None,
    ):
        """
        Initialize the ChenLogger with specified configuration.

        Args:
            name (str): Name of the logger, defaults to "chen3"
            level (int): Logging level, defaults to logging.INFO
            log_file (Optional[Path]): Path to log file if file logging is desired
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Create formatters with different levels of detail
        console_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
        )

        # Console handler for immediate feedback
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # File handler for detailed logging if specified
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

    def debug(self, msg: str, *args: Any, **kwargs: Dict[str, Any]) -> None:
        """
        Log a debug message.

        Args:
            msg (str): The message to log
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args: Any, **kwargs: Dict[str, Any]) -> None:
        """
        Log an info message.

        Args:
            msg (str): The message to log
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args: Any, **kwargs: Dict[str, Any]) -> None:
        """
        Log a warning message.

        Args:
            msg (str): The message to log
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args: Any, **kwargs: Dict[str, Any]) -> None:
        """
        Log an error message.

        Args:
            msg (str): The message to log
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args: Any, **kwargs: Dict[str, Any]) -> None:
        """
        Log a critical message.

        Args:
            msg (str): The message to log
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments
        """
        self.logger.critical(msg, *args, **kwargs)


# Create default logger instance for use throughout the package
logger = ChenLogger()

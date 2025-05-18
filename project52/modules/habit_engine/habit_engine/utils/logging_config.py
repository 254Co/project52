"""Logging configuration for the habit engine.

This module provides logging setup functionality for the habit engine package.
It configures both console and optional file logging with a consistent format
and configurable log levels.
"""

import logging
from pathlib import Path


_FMT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def setup_logging(level: str = "INFO", *, log_dir: Path | None = None) -> None:
    """Configure logging for the habit engine.
    
    This function sets up logging with both console output and optional
    file output. The logging format includes timestamp, log level, module
    name, and message. Log files are created in the specified directory
    if provided.
    
    Args:
        level (str, optional): Logging level (e.g., "INFO", "DEBUG", "WARNING").
            Defaults to "INFO".
        log_dir (Path | None, optional): Directory for log files.
            If None, only console logging is configured. Defaults to None.
            
    Note:
        - The log format is: "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        - Log files are named "habit_engine.log"
        - If log_dir doesn't exist, it will be created
        - Existing logging configuration is overridden (force=True)
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "habit_engine.log"))
    logging.basicConfig(level=level.upper(), format=_FMT, handlers=handlers, force=True)
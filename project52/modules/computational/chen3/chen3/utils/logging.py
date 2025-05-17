"""
Logging configuration for the Chen3 package.

This module provides a centralized logging configuration for the entire package,
with different log levels and formatting for different components.
"""

import logging
import sys
from typing import Optional
from pathlib import Path

class ChenLogger:
    """Custom logger for the Chen3 package with different log levels and formatting."""
    
    def __init__(
        self,
        name: str = "chen3",
        level: int = logging.INFO,
        log_file: Optional[Path] = None
    ):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, msg: str, *args, **kwargs):
        """Log a debug message."""
        self.logger.debug(msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        """Log an info message."""
        self.logger.info(msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        """Log a warning message."""
        self.logger.warning(msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        """Log an error message."""
        self.logger.error(msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        """Log a critical message."""
        self.logger.critical(msg, *args, **kwargs)

# Create default logger instance
logger = ChenLogger() 
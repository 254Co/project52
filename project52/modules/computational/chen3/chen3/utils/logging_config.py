"""
Logging Configuration for the Chen3 Model

This module provides a comprehensive logging configuration for the Chen3 package,
including different log levels, formatters, and handlers for various output destinations.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

class Chen3Logger:
    """Logger configuration for the Chen3 model."""
    
    def __init__(
        self,
        name: str = "chen3",
        level: Union[int, str] = logging.INFO,
        log_file: Optional[Union[str, Path]] = None,
        console: bool = True,
        file_level: Optional[Union[int, str]] = None,
        console_level: Optional[Union[int, str]] = None
    ):
        """
        Initialize the Chen3 logger.
        
        Args:
            name: Logger name
            level: Default logging level
            log_file: Path to log file (optional)
            console: Whether to log to console
            file_level: Logging level for file handler
            console_level: Logging level for console handler
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Clear existing handlers
        self.logger.handlers = []
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
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
        """Get the configured logger instance."""
        return self.logger

def setup_logging(
    level: Union[int, str] = logging.INFO,
    log_dir: Optional[Union[str, Path]] = None,
    console: bool = True
) -> logging.Logger:
    """
    Set up logging for the Chen3 model.
    
    Args:
        level: Logging level
        log_dir: Directory for log files
        console: Whether to log to console
    
    Returns:
        Configured logger instance
    """
    if log_dir is None:
        log_dir = Path.home() / ".chen3" / "logs"
    
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"chen3_{timestamp}.log"
    
    logger = Chen3Logger(
        level=level,
        log_file=log_file,
        console=console
    )
    
    return logger.get_logger()

# Create default logger instance
logger = setup_logging() 
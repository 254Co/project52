"""Configuration and logging setup for the riskfree package.

This module provides configuration management and logging setup for the riskfree package.
It includes:
1. Logging configuration with appropriate levels and formatting
2. Package-level constants and settings
3. Utility functions for configuration management

The logging system is designed to:
- Provide informative messages during curve construction
- Log warnings for data quality issues
- Track errors during data fetching and processing
- Enable debugging when needed

Key features:
    1. Configurable log levels via environment variable
    2. Consistent log formatting across the package
    3. Hierarchical logger names for better organization
    4. Runtime log level adjustment capability

Note:
    The default log level can be overridden by setting the RISKFREE_LOG_LEVEL
    environment variable to one of: DEBUG, INFO, WARNING, ERROR, CRITICAL.
"""
from __future__ import annotations
import logging
import os
from typing import Optional

# Package-level constants
PACKAGE_NAME = "riskfree"
DEFAULT_LOG_LEVEL = logging.INFO

# Initialize logging configuration
_DEFAULT_LEVEL = os.getenv("RISKFREE_LOG_LEVEL", "INFO").upper()
_logger = logging.getLogger("riskfree")
_handler = logging.StreamHandler()
_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
_handler.setFormatter(_formatter)
_logger.addHandler(_handler)
_logger.setLevel(_DEFAULT_LEVEL)

__all__ = ["set_level", "get_logger"]

def set_level(level: str) -> None:
    """Change runtime log level.
    
    This function allows changing the log level at runtime, which is useful
    for debugging or adjusting verbosity during execution.
    
    Args:
        level: New log level as a string. Must be one of:
              - "DEBUG": Detailed information for debugging
              - "INFO": General information about program execution
              - "WARNING": Indicate potential problems
              - "ERROR": Serious problems that may affect functionality
              - "CRITICAL": Critical errors that may prevent execution
              
    Example:
        >>> set_level("DEBUG")  # Enable detailed logging
        >>> set_level("ERROR")  # Show only errors
        
    Note:
        The log level can also be set via the RISKFREE_LOG_LEVEL
        environment variable before starting the program.
    """
    _logger.setLevel(level.upper())


def get_logger(name: str | None = None) -> logging.Logger:  # pragma: no cover
    """Get a logger instance for the specified module.
    
    This function returns either the root package logger or a module-specific
    logger with a hierarchical name structure.
    
    Args:
        name: Optional module name. If provided, returns a logger with name
             "riskfree.{name}". If None, returns the root package logger.
             
    Returns:
        A configured logging.Logger instance.
        
    Example:
        >>> logger = get_logger("curve")  # Get logger for curve module
        >>> logger.info("Building curve...")  # Log a message
        
    Note:
        The returned logger inherits the package's logging configuration,
        including the log level and formatting.
    """
    return _logger if name is None else logging.getLogger(f"riskfree.{name}")
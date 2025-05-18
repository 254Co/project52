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
"""
from __future__ import annotations
import logging
import os
from typing import Optional

# Package-level constants
PACKAGE_NAME = "riskfree"
DEFAULT_LOG_LEVEL = logging.INFO

_DEFAULT_LEVEL = os.getenv("RISKFREE_LOG_LEVEL", "INFO").upper()
_logger = logging.getLogger("riskfree")
_handler = logging.StreamHandler()
_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
_handler.setFormatter(_formatter)
_logger.addHandler(_handler)
_logger.setLevel(_DEFAULT_LEVEL)

__all__ = ["set_level", "get_logger"]

def set_level(level: str) -> None:
    """Change runtime log level (e.g., "ERROR", "DEBUG")."""
    _logger.setLevel(level.upper())


def get_logger(name: str | None = None) -> logging.Logger:  # pragma: no cover
    return _logger if name is None else logging.getLogger(f"riskfree.{name}")
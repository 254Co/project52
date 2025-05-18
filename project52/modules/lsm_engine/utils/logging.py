"""
Logging utilities for the LSM Engine.

This module provides JSON-formatted logging functionality for the LSM Engine.
It includes a custom JSON formatter and initialization function for setting up
structured logging output.
"""

# File: lsm_engine/utils/logging.py
import logging
import json
import sys
import time

class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging output.
    
    This formatter converts log records into JSON format with the following fields:
    - t: Timestamp in ISO format
    - lvl: Log level (lowercase)
    - msg: Log message
    - mod: Module name
    - fn: Function name
    - ln: Line number
    """
    def format(self, record):
        """
        Format a log record as JSON.
        
        Args:
            record: LogRecord instance to format
            
        Returns:
            str: JSON-formatted log record
        """
        return json.dumps({
            "t": time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(record.created)),
            "lvl": record.levelname.lower(),
            "msg": record.getMessage(),
            "mod": record.module,
            "fn": record.funcName,
            "ln": record.lineno,
        })


def init_logger(level: str = "INFO"):
    """
    Initialize the root logger with JSON formatting.
    
    This function sets up the root logger with:
    - JSON formatting for structured output
    - Output to stdout
    - Configurable log level
    
    Args:
        level: Logging level (default: "INFO")
    """
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(h)
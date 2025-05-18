"""
Logging Utilities for the LSM Engine

This module provides structured logging functionality for the LSM Engine using
JSON-formatted output. The implementation includes:

1. JsonFormatter: A custom formatter that converts log records to JSON format
   - Timestamps in ISO format
   - Standardized log levels
   - Contextual information (module, function, line number)
   - Structured message format

2. Logger initialization with configurable settings
   - JSON output to stdout
   - Configurable log levels
   - Clean handler management

The structured logging format enables:
- Easy parsing by log aggregation tools
- Consistent log format across the application
- Rich contextual information for debugging
- Integration with monitoring systems

Example log output:
    {
        "t": "2024-03-20T15:30:45",
        "lvl": "info",
        "msg": "LSM pricing completed",
        "mod": "pricer",
        "fn": "price",
        "ln": 42
    }
"""

# File: lsm_engine/utils/logging.py
import logging
import json
import sys
import time

class JsonFormatter(logging.Formatter):
    """
    JSON formatter for structured logging output.
    
    This formatter converts log records into a standardized JSON format that
    includes timestamp, log level, message, and contextual information about
    where the log was generated.
    
    The JSON output includes the following fields:
    - t: Timestamp in ISO format (YYYY-MM-DDThh:mm:ss)
    - lvl: Log level in lowercase (debug, info, warning, error, critical)
    - msg: The actual log message
    - mod: Name of the module where the log was generated
    - fn: Name of the function where the log was generated
    - ln: Line number where the log was generated
    
    This structured format enables:
    - Easy parsing by log aggregation tools
    - Consistent log format across the application
    - Rich contextual information for debugging
    - Integration with monitoring systems
    """
    
    def format(self, record):
        """
        Format a log record as a JSON string.
        
        This method takes a LogRecord instance and converts it into a JSON-formatted
        string containing all relevant information about the log event.
        
        Args:
            record: LogRecord instance containing the log event information
            
        Returns:
            str: JSON-formatted string representing the log record
            
        Example:
            >>> formatter = JsonFormatter()
            >>> record = logging.LogRecord(...)
            >>> json_str = formatter.format(record)
            >>> print(json_str)
            {"t": "2024-03-20T15:30:45", "lvl": "info", "msg": "Test message", ...}
        """
        return json.dumps({
            "t": time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(record.created)),
            "lvl": record.levelname.lower(),
            "msg": record.getMessage(),
            "mod": record.module,
            "fn": record.funcName,
            "ln": record.lineno,
        })


def init_logger(level: str = "ERROR"):
    """
    Initialize the root logger with JSON formatting.
    
    This function sets up the root logger with a standardized configuration:
    - JSON formatting for structured output
    - Output to stdout for easy integration with logging systems
    - Configurable log level
    - Clean handler management (removes existing handlers)
    
    The logger can be configured with different log levels:
    - DEBUG: Detailed information for debugging
    - INFO: General information about program execution
    - WARNING: Indicate a potential problem
    - ERROR: A more serious problem
    - CRITICAL: A critical problem that may prevent the program from running
    
    Args:
        level: Logging level as a string (default: "ERROR")
            Must be one of: "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"
            
    Example:
        >>> init_logger("INFO")
        >>> logging.info("Application started")
        {"t": "2024-03-20T15:30:45", "lvl": "info", "msg": "Application started", ...}
    """
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(h)
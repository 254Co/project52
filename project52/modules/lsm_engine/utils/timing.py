"""
Timing utilities for the LSM Engine.

This module provides a context manager for timing code execution.
It can be used to measure the performance of various components
of the LSM algorithm.
"""

# File: lsm_engine/utils/timing.py
import time

class Timer:
    """
    Context manager for timing code execution.
    
    This class provides a simple way to measure the execution time
    of a block of code using Python's context manager protocol.
    
    Example:
        >>> with Timer("Pricing"):
        ...     result = pricer.price(paths, exercise_idx)
        Pricing elapsed: 1.2345 s
    """
    def __init__(self, name: str = None):
        """
        Initialize the timer.
        
        Args:
            name: Optional name for the timer (default: "Timer")
        """
        self.name = name or "Timer"

    def __enter__(self):
        """Start timing when entering the context."""
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        """
        Stop timing and print elapsed time when exiting the context.
        
        Args:
            exc_type: Exception type if an exception occurred
            exc: Exception instance if an exception occurred
            tb: Traceback if an exception occurred
        """
        elapsed = time.time() - self.start
        print(f"{self.name} elapsed: {elapsed:.4f} s")
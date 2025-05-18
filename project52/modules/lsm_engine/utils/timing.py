"""
Timing Utilities for the LSM Engine

This module provides performance measurement tools for the LSM Engine, including:

1. Timer: A context manager for measuring code execution time
   - Simple and intuitive interface using 'with' statement
   - Automatic timing of code blocks
   - Formatted output of elapsed time
   - Optional naming of timed sections

The timing utilities are useful for:
- Performance profiling of the LSM algorithm
- Identifying bottlenecks in the implementation
- Measuring the impact of optimizations
- Comparing different parameter configurations

Example usage:
    >>> with Timer("Path Generation"):
    ...     paths = generate_gbm_paths(s0, mu, sigma, T, n_steps, n_paths)
    Path Generation elapsed: 0.1234 s
    
    >>> with Timer("LSM Pricing"):
    ...     price = pricer.price(paths, exercise_idx)
    LSM Pricing elapsed: 1.2345 s
"""

# File: lsm_engine/utils/timing.py
import time
from typing import Optional, Type, Any, TracebackType

class Timer:
    """
    Context manager for timing code execution.
    
    This class provides a simple and intuitive way to measure the execution time
    of a block of code using Python's context manager protocol. It automatically
    handles the timing of code blocks and provides formatted output of the
    elapsed time.
    
    The timer can be used in two ways:
    1. As a context manager with the 'with' statement
    2. By manually calling __enter__ and __exit__
    
    Features:
    - Automatic timing of code blocks
    - Formatted output of elapsed time in seconds
    - Optional naming of timed sections
    - Exception-safe timing (timing continues even if an exception occurs)
    
    Attributes:
        name (str): Name of the timer for output identification
        start (float): Start time of the timing period
    """
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the timer with an optional name.
        
        Args:
            name: Optional name for the timer (default: "Timer")
                This name will be used in the output message to identify
                which section of code was timed.
        """
        self.name = name or "Timer"
        self.start: float = 0.0

    def __enter__(self) -> 'Timer':
        """
        Start timing when entering the context.
        
        This method is called when entering the 'with' block. It records
        the current time as the start time for the timing period.
        
        Returns:
            self: The Timer instance for potential method chaining
        """
        self.start = time.time()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType]
    ) -> None:
        """
        Stop timing and print elapsed time when exiting the context.
        
        This method is called when exiting the 'with' block. It calculates
        the elapsed time and prints a formatted message with the timer name
        and elapsed time in seconds.
        
        The timing continues even if an exception occurs within the timed
        block, allowing for timing of error cases as well.
        
        Args:
            exc_type: Exception type if an exception occurred
            exc: Exception instance if an exception occurred
            tb: Traceback if an exception occurred
            
        Example output:
            "Path Generation elapsed: 0.1234 s"
        """
        elapsed = time.time() - self.start
        print(f"{self.name} elapsed: {elapsed:.4f} s")
# File: chen3/utils/timers.py
"""
Simple timing utilities for the Chen3 package.

This module provides a context manager for timing code execution.
It can be used to measure the performance of specific code blocks
and is particularly useful for profiling and optimization.
"""

import time
from contextlib import contextmanager
from typing import Generator, Optional


@contextmanager
def timer(name: Optional[str] = None) -> Generator[None, None, None]:
    """
    Context manager for timing code execution.

    This context manager measures the execution time of the code block
    it wraps and prints the elapsed time in seconds. It can be used
    with a custom name for better identification in the output.

    Args:
        name (Optional[str]): Name identifier for the timer output.
            If None, defaults to 'timer'.

    Yields:
        None: The context manager yields control to the wrapped code block.

    Example:
        >>> with timer("my_operation"):
        ...     # Code to time
        ...     result = some_expensive_operation()
        [my_operation] 1.2345s
    """
    start = time.time()
    yield
    end = time.time()
    print(f"[{name or 'timer'}] {end - start:.4f}s")

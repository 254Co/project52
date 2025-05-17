# File: chen3/utils/timers.py
"""Simple timing context manager."""
import time
from contextlib import contextmanager

@contextmanager
def timer(name: str = None):
    start = time.time()
    yield
    end = time.time()
    print(f"[{name or 'timer'}] {end - start:.4f}s")

# File: chen3/utils/rng.py
"""
Random number generation utilities for the Chen3 package.

This module provides functions for creating and managing random number
generators (RNGs) with reproducible seeds. It uses NumPy's modern
random number generation system for better statistical properties
and performance.
"""

import numpy as np
from typing import Optional
from numpy.random import Generator

def get_rng(seed: Optional[int] = None) -> Generator:
    """
    Create a new random number generator with an optional seed.
    
    This function creates a NumPy random number generator instance
    that can be used for generating random numbers in a reproducible
    way. If a seed is provided, the sequence of random numbers will
    be deterministic.
    
    Args:
        seed (Optional[int]): Seed for the random number generator.
            If None, a random seed is used.
    
    Returns:
        Generator: A NumPy random number generator instance
    
    Example:
        >>> rng = get_rng(42)
        >>> rng.random()
        0.7739560485559633
        >>> rng = get_rng(42)  # Same seed
        >>> rng.random()  # Same result
        0.7739560485559633
    """
    return np.random.default_rng(seed)

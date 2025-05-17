# File: chen3/utils/rng.py
"""Random number utilities."""
import numpy as np

def get_rng(seed: int = None):
    return np.random.default_rng(seed)

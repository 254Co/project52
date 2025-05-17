# File: chen3/risk/sensitivity_cache.py
"""
Cache for risk sensitivities to avoid redundant computation.
"""
import pickle
import hashlib
import os
from typing import Any, Callable

class SensitivityCache:
    def __init__(self, cache_dir: str = ".cache"):  # relative path
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_dir = cache_dir

    def _cache_file(self, key: str) -> str:
        fname = hashlib.sha1(key.encode()).hexdigest() + ".pkl"
        return os.path.join(self.cache_dir, fname)

    def get(self, key: str) -> Any:
        path = self._cache_file(key)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                return pickle.load(f)
        return None

    def set(self, key: str, value: Any) -> None:
        path = self._cache_file(key)
        with open(path, 'wb') as f:
            pickle.dump(value, f)

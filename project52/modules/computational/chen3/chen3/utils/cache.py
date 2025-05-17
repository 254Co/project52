"""
Caching utilities for the Chen3 package.

This module provides tools for caching computation results to improve
performance for repeated calculations.
"""

import functools
import hashlib
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypeVar, Union, cast

import numpy as np
import pandas as pd

from .exceptions import ConfigurationError
from .logging import logger

T = TypeVar("T")


class Cache:
    """Cache manager for computation results."""

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        max_size: int = 1000,
        ttl: Optional[timedelta] = None,
    ):
        """Initialize the cache manager.

        Args:
            cache_dir: Directory to store cache files
            max_size: Maximum number of cache entries
            ttl: Time-to-live for cache entries
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".chen3" / "cache"
        self.max_size = max_size
        self.ttl = ttl
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._initialize_cache()

    def _initialize_cache(self) -> None:
        """Initialize cache directory and load existing cache."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()
        except Exception as e:
            logger.warning(f"Failed to initialize cache: {e}")
            self._cache = {}

    def _load_cache(self) -> None:
        """Load cache from disk."""
        cache_file = self.cache_dir / "cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    self._cache = json.load(f)
                self._cleanup_cache()
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}")
                self._cache = {}

    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            cache_file = self.cache_dir / "cache.json"
            with open(cache_file, "w") as f:
                json.dump(self._cache, f)
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

    def _cleanup_cache(self) -> None:
        """Clean up expired and excess cache entries."""
        now = datetime.now()
        keys_to_remove = []

        # Remove expired entries
        if self.ttl:
            for key, entry in self._cache.items():
                timestamp = datetime.fromisoformat(entry["timestamp"])
                if now - timestamp > self.ttl:
                    keys_to_remove.append(key)

        # Remove excess entries
        if len(self._cache) > self.max_size:
            sorted_entries = sorted(
                self._cache.items(),
                key=lambda x: datetime.fromisoformat(x[1]["timestamp"]),
            )
            keys_to_remove.extend(k for k, _ in sorted_entries[: len(self._cache) - self.max_size])

        # Remove entries
        for key in keys_to_remove:
            self._remove_cache_entry(key)

    def _remove_cache_entry(self, key: str) -> None:
        """Remove a cache entry."""
        if key in self._cache:
            entry = self._cache[key]
            if "file" in entry:
                try:
                    os.remove(self.cache_dir / entry["file"])
                except Exception as e:
                    logger.warning(f"Failed to remove cache file: {e}")
            del self._cache[key]

    def _generate_key(self, *args: Any, **kwargs: Any) -> str:
        """Generate a cache key from function arguments."""
        # Convert args and kwargs to a stable string representation
        key_parts = []
        for arg in args:
            if isinstance(arg, (np.ndarray, pd.DataFrame)):
                key_parts.append(hashlib.md5(arg.tobytes()).hexdigest())
            else:
                key_parts.append(str(arg))
        for k, v in sorted(kwargs.items()):
            if isinstance(v, (np.ndarray, pd.DataFrame)):
                key_parts.append(f"{k}={hashlib.md5(v.tobytes()).hexdigest()}")
            else:
                key_parts.append(f"{k}={v}")
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get a value from the cache."""
        if key not in self._cache:
            return None

        entry = self._cache[key]
        if self.ttl:
            timestamp = datetime.fromisoformat(entry["timestamp"])
            if datetime.now() - timestamp > self.ttl:
                self._remove_cache_entry(key)
                return None

        try:
            if "file" in entry:
                file_path = self.cache_dir / entry["file"]
                if file_path.exists():
                    if entry["type"] == "numpy":
                        return np.load(file_path)
                    elif entry["type"] == "pandas":
                        return pd.read_pickle(file_path)
                    else:
                        with open(file_path, "r") as f:
                            return json.load(f)
            return entry["value"]
        except Exception as e:
            logger.warning(f"Failed to retrieve cache entry: {e}")
            self._remove_cache_entry(key)
            return None

    def set(self, key: str, value: Any) -> None:
        """Set a value in the cache."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "type": type(value).__name__,
        }

        try:
            if isinstance(value, np.ndarray):
                file_path = f"{key}.npy"
                np.save(self.cache_dir / file_path, value)
                entry["file"] = file_path
            elif isinstance(value, pd.DataFrame):
                file_path = f"{key}.pkl"
                value.to_pickle(self.cache_dir / file_path)
                entry["file"] = file_path
            elif isinstance(value, (dict, list)):
                file_path = f"{key}.json"
                with open(self.cache_dir / file_path, "w") as f:
                    json.dump(value, f)
                entry["file"] = file_path
            else:
                entry["value"] = value

            self._cache[key] = entry
            self._cleanup_cache()
            self._save_cache()
        except Exception as e:
            logger.warning(f"Failed to cache value: {e}")

    def clear(self) -> None:
        """Clear all cache entries."""
        for key in list(self._cache.keys()):
            self._remove_cache_entry(key)
        self._save_cache()


# Create global cache instance
cache = Cache()


def cached(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to cache function results."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        # Generate cache key
        key = cache._generate_key(*args, **kwargs)

        # Try to get from cache
        result = cache.get(key)
        if result is not None:
            logger.debug(f"Cache hit for {func.__name__}")
            return result

        # Compute result
        result = func(*args, **kwargs)

        # Cache result
        cache.set(key, result)
        logger.debug(f"Cached result for {func.__name__}")

        return result

    return cast(Callable[..., T], wrapper)


def clear_cache() -> None:
    """Clear the cache."""
    cache.clear()


def set_cache_dir(cache_dir: Union[str, Path]) -> None:
    """Set the cache directory."""
    global cache
    cache = Cache(cache_dir=cache_dir)


def set_cache_ttl(ttl: timedelta) -> None:
    """Set the cache time-to-live."""
    global cache
    cache = Cache(cache_dir=cache.cache_dir, ttl=ttl)


def set_cache_size(max_size: int) -> None:
    """Set the maximum cache size."""
    global cache
    cache = Cache(cache_dir=cache.cache_dir, max_size=max_size) 
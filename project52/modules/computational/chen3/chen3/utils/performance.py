"""
Performance monitoring utilities for the Chen3 package.

This module provides tools for monitoring and analyzing the performance of
various components of the Chen3 package, including execution times, memory usage,
and GPU utilization.
"""

import functools
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, Optional, TypeVar, cast

import numpy as np
import psutil

from .logging import logger

T = TypeVar("T")


class PerformanceMonitor:
    """Monitor and track performance metrics."""

    def __init__(self):
        """Initialize the performance monitor."""
        self.metrics: Dict[str, Dict[str, float]] = {}
        self._start_times: Dict[str, float] = {}
        self._process = psutil.Process()

    def start(self, name: str) -> None:
        """Start timing a named operation."""
        self._start_times[name] = time.perf_counter()

    def stop(self, name: str) -> Dict[str, float]:
        """Stop timing a named operation and record metrics."""
        if name not in self._start_times:
            raise ValueError(f"No timing started for {name}")

        end_time = time.perf_counter()
        duration = end_time - self._start_times[name]

        # Get memory usage
        memory_info = self._process.memory_info()
        memory_usage = memory_info.rss / 1024 / 1024  # Convert to MB

        # Record metrics
        metrics = {
            "duration": duration,
            "memory_usage": memory_usage,
            "cpu_percent": self._process.cpu_percent(),
        }

        # Add to history
        if name not in self.metrics:
            self.metrics[name] = {}
        self.metrics[name].update(metrics)

        # Log performance
        logger.debug(
            f"Performance metrics for {name}: "
            f"duration={duration:.3f}s, "
            f"memory={memory_usage:.1f}MB, "
            f"CPU={metrics['cpu_percent']}%"
        )

        return metrics

    def get_metrics(self, name: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Get recorded metrics for a named operation or all operations."""
        if name is not None:
            return {name: self.metrics.get(name, {})}
        return self.metrics

    def reset(self) -> None:
        """Reset all recorded metrics."""
        self.metrics.clear()
        self._start_times.clear()


# Create global performance monitor
monitor = PerformanceMonitor()


def track_performance(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to track performance of a function."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        name = f"{func.__module__}.{func.__name__}"
        monitor.start(name)
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            monitor.stop(name)

    return cast(Callable[..., T], wrapper)


@contextmanager
def performance_context(name: str):
    """Context manager for tracking performance of a code block."""
    monitor.start(name)
    try:
        yield
    finally:
        monitor.stop(name)


def get_performance_summary() -> Dict[str, Dict[str, float]]:
    """Get a summary of all recorded performance metrics."""
    return monitor.get_metrics()


def reset_performance_metrics() -> None:
    """Reset all recorded performance metrics."""
    monitor.reset()


class PerformanceStats:
    """Statistical analysis of performance metrics."""

    @staticmethod
    def analyze_metrics(metrics: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """Analyze performance metrics and compute statistics."""
        stats = {}
        for name, values in metrics.items():
            stats[name] = {
                "mean": np.mean(list(values.values())),
                "std": np.std(list(values.values())),
                "min": np.min(list(values.values())),
                "max": np.max(list(values.values())),
            }
        return stats

    @staticmethod
    def format_summary(stats: Dict[str, Dict[str, float]]) -> str:
        """Format performance statistics as a string."""
        summary = []
        for name, values in stats.items():
            summary.append(f"\n{name}:")
            for stat, value in values.items():
                summary.append(f"  {stat}: {value:.3f}")
        return "\n".join(summary)


def log_performance_summary() -> None:
    """Log a summary of all recorded performance metrics."""
    metrics = get_performance_summary()
    stats = PerformanceStats.analyze_metrics(metrics)
    summary = PerformanceStats.format_summary(stats)
    logger.info(f"Performance Summary:{summary}") 
"""
GPU Memory Management Module

This module provides comprehensive GPU memory management functionality including:
- Memory pooling and allocation
- Memory usage monitoring
- Out-of-memory handling
- Memory optimization strategies
"""

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cupy as cp
import numpy as np


@dataclass
class MemoryStats:
    """Memory usage statistics for a GPU device."""

    total_memory: int
    free_memory: int
    used_memory: int
    peak_memory: int
    allocation_count: int


class GPUMemoryManager:
    """
    Manages GPU memory allocation, pooling, and monitoring.

    This class provides a high-level interface for GPU memory management,
    including memory pooling, allocation tracking, and out-of-memory handling.
    It also provides utilities for monitoring memory usage and optimizing
    memory allocation patterns.

    Attributes:
        device_id (int): GPU device ID to manage
        memory_pool (cp.cuda.MemoryPool): CuPy memory pool instance
        allocation_tracker (Dict): Tracks memory allocations
        peak_memory (int): Peak memory usage
        logger (logging.Logger): Logger instance
    """

    def __init__(self, device_id: int = 0):
        """
        Initialize the GPU memory manager.

        Args:
            device_id (int): GPU device ID to manage
        """
        self.device_id = device_id
        self.device = cp.cuda.Device(device_id)
        self.memory_pool = cp.cuda.MemoryPool()
        self.allocation_tracker: Dict[int, Tuple[int, str]] = {}
        self.peak_memory = 0
        self.logger = logging.getLogger(__name__)

        # Set up logging
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    @contextmanager
    def allocate(self, size: int, name: str = "unnamed") -> cp.ndarray:
        """
        Allocate GPU memory with automatic cleanup.

        Args:
            size (int): Size of memory to allocate in bytes
            name (str): Name for the allocation (for tracking)

        Yields:
            cp.ndarray: Allocated memory array

        Raises:
            RuntimeError: If allocation fails
        """
        try:
            with self.device:
                array = self.memory_pool.malloc(size)
                self.allocation_tracker[id(array)] = (size, name)
                self.peak_memory = max(self.peak_memory, self.get_used_memory())
                yield array
        except cp.cuda.runtime.CUDARuntimeError as e:
            self.logger.error(f"Failed to allocate {size} bytes: {str(e)}")
            raise RuntimeError(f"GPU memory allocation failed: {str(e)}")
        finally:
            if "array" in locals():
                self.free(array)

    def free(self, array: cp.ndarray) -> None:
        """
        Free allocated GPU memory.

        Args:
            array (cp.ndarray): Array to free
        """
        if id(array) in self.allocation_tracker:
            del self.allocation_tracker[id(array)]
            with self.device:
                self.memory_pool.free(array)

    def get_memory_stats(self) -> MemoryStats:
        """
        Get current memory usage statistics.

        Returns:
            MemoryStats: Current memory statistics
        """
        with self.device:
            total = cp.cuda.runtime.memGetInfo()[1]
            free = cp.cuda.runtime.memGetInfo()[0]
            used = total - free
            return MemoryStats(
                total_memory=total,
                free_memory=free,
                used_memory=used,
                peak_memory=self.peak_memory,
                allocation_count=len(self.allocation_tracker),
            )

    def get_used_memory(self) -> int:
        """
        Get current used memory in bytes.

        Returns:
            int: Used memory in bytes
        """
        with self.device:
            total, free = cp.cuda.runtime.memGetInfo()
            return total - free

    def clear_memory(self) -> None:
        """Clear all allocated memory."""
        with self.device:
            self.memory_pool.free_all_blocks()
            self.allocation_tracker.clear()
            self.peak_memory = 0

    def optimize_memory(self) -> None:
        """
        Optimize memory usage by:
        1. Freeing unused memory
        2. Defragmenting memory pool
        3. Resetting peak memory tracking
        """
        with self.device:
            self.memory_pool.free_all_blocks()
            self.peak_memory = self.get_used_memory()

    def get_allocation_summary(self) -> Dict[str, int]:
        """
        Get summary of current memory allocations.

        Returns:
            Dict[str, int]: Dictionary mapping allocation names to sizes
        """
        summary: Dict[str, int] = {}
        for size, name in self.allocation_tracker.values():
            summary[name] = summary.get(name, 0) + size
        return summary

    def check_memory_available(self, required_size: int) -> bool:
        """
        Check if required memory is available.

        Args:
            required_size (int): Required memory size in bytes

        Returns:
            bool: True if memory is available, False otherwise
        """
        with self.device:
            _, free = cp.cuda.runtime.memGetInfo()
            return free >= required_size

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.clear_memory()

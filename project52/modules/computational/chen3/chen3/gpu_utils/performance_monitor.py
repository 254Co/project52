"""
GPU Performance Monitoring Module

This module provides comprehensive performance monitoring for GPU operations including:
- GPU utilization tracking
- Memory usage monitoring
- Computation time measurement
- Performance metrics collection
"""

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from queue import Queue
from typing import Dict, List, Optional, Tuple

import cupy as cp
import numpy as np
import psutil


@dataclass
class PerformanceMetrics:
    """Performance metrics for a GPU operation."""

    timestamp: datetime
    operation_name: str
    duration: float
    gpu_utilization: float
    memory_used: int
    memory_total: int
    compute_utilization: float
    memory_utilization: float
    power_usage: float
    temperature: float


class GPUPerformanceMonitor:
    """
    Monitors GPU performance metrics.

    This class provides comprehensive performance monitoring for GPU operations,
    including utilization tracking, memory monitoring, and computation timing.
    It supports both synchronous and asynchronous monitoring modes.

    Attributes:
        device_id (int): GPU device ID to monitor
        metrics_history (List[PerformanceMetrics]): History of performance metrics
        monitoring_thread (Optional[threading.Thread]): Background monitoring thread
        stop_event (threading.Event): Event to stop monitoring
        metrics_queue (Queue): Queue for collecting metrics
        logger (logging.Logger): Logger instance
    """

    def __init__(self, device_id: int = 0):
        """
        Initialize the GPU performance monitor.

        Args:
            device_id (int): GPU device ID to monitor
        """
        self.device_id = device_id
        self.device = cp.cuda.Device(device_id)
        self.metrics_history: List[PerformanceMetrics] = []
        self.monitoring_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.metrics_queue: Queue = Queue()
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
    def monitor_operation(self, operation_name: str = "unnamed"):
        """
        Context manager for monitoring GPU operations.

        Args:
            operation_name (str): Name of the operation being monitored

        Yields:
            None
        """
        start_time = time.time()
        start_metrics = self._get_current_metrics()

        try:
            yield
        finally:
            end_time = time.time()
            end_metrics = self._get_current_metrics()

            metrics = PerformanceMetrics(
                timestamp=datetime.now(),
                operation_name=operation_name,
                duration=end_time - start_time,
                gpu_utilization=end_metrics["gpu_utilization"],
                memory_used=end_metrics["memory_used"],
                memory_total=end_metrics["memory_total"],
                compute_utilization=end_metrics["compute_utilization"],
                memory_utilization=end_metrics["memory_utilization"],
                power_usage=end_metrics["power_usage"],
                temperature=end_metrics["temperature"],
            )

            self.metrics_history.append(metrics)
            self.metrics_queue.put(metrics)

    def start_monitoring(self, interval: float = 1.0):
        """
        Start background monitoring.

        Args:
            interval (float): Monitoring interval in seconds
        """
        if self.monitoring_thread is not None:
            self.logger.warning("Monitoring already started")
            return

        self.stop_event.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop, args=(interval,)
        )
        self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stop background monitoring."""
        if self.monitoring_thread is None:
            return

        self.stop_event.set()
        self.monitoring_thread.join()
        self.monitoring_thread = None

    def _monitoring_loop(self, interval: float):
        """
        Background monitoring loop.

        Args:
            interval (float): Monitoring interval in seconds
        """
        while not self.stop_event.is_set():
            metrics = self._get_current_metrics()
            self.metrics_queue.put(metrics)
            time.sleep(interval)

    def _get_current_metrics(self) -> Dict[str, float]:
        """
        Get current GPU metrics.

        Returns:
            Dict[str, float]: Current GPU metrics
        """
        with self.device:
            try:
                # Get GPU utilization
                gpu_util = cp.cuda.runtime.deviceGetAttribute(
                    cp.cuda.runtime.cudaDevAttrComputeMode, self.device_id
                )

                # Get memory info
                free, total = cp.cuda.runtime.memGetInfo()
                used = total - free

                # Get compute utilization
                compute_util = cp.cuda.runtime.deviceGetAttribute(
                    cp.cuda.runtime.cudaDevAttrComputeCapabilityMajor, self.device_id
                )

                # Get memory utilization
                memory_util = used / total if total > 0 else 0

                # Get power usage (if available)
                try:
                    power = cp.cuda.runtime.deviceGetAttribute(
                        cp.cuda.runtime.cudaDevAttrMaxPower, self.device_id
                    )
                except:
                    power = 0.0

                # Get temperature (if available)
                try:
                    temp = cp.cuda.runtime.deviceGetAttribute(
                        cp.cuda.runtime.cudaDevAttrMaxThreadsPerBlock, self.device_id
                    )
                except:
                    temp = 0.0

                return {
                    "gpu_utilization": float(gpu_util),
                    "memory_used": int(used),
                    "memory_total": int(total),
                    "compute_utilization": float(compute_util),
                    "memory_utilization": float(memory_util),
                    "power_usage": float(power),
                    "temperature": float(temp),
                }
            except Exception as e:
                self.logger.error(f"Error getting GPU metrics: {str(e)}")
                return {
                    "gpu_utilization": 0.0,
                    "memory_used": 0,
                    "memory_total": 0,
                    "compute_utilization": 0.0,
                    "memory_utilization": 0.0,
                    "power_usage": 0.0,
                    "temperature": 0.0,
                }

    def get_performance_stats(self) -> Dict[str, float]:
        """
        Get performance statistics.

        Returns:
            Dict[str, float]: Performance statistics
        """
        if not self.metrics_history:
            return {}

        durations = [m.duration for m in self.metrics_history]
        gpu_utils = [m.gpu_utilization for m in self.metrics_history]
        memory_utils = [m.memory_utilization for m in self.metrics_history]

        return {
            "avg_duration": np.mean(durations),
            "max_duration": np.max(durations),
            "min_duration": np.min(durations),
            "avg_gpu_utilization": np.mean(gpu_utils),
            "max_gpu_utilization": np.max(gpu_utils),
            "avg_memory_utilization": np.mean(memory_utils),
            "max_memory_utilization": np.max(memory_utils),
        }

    def get_operation_stats(self, operation_name: str) -> Dict[str, float]:
        """
        Get statistics for a specific operation.

        Args:
            operation_name (str): Name of the operation

        Returns:
            Dict[str, float]: Operation statistics
        """
        operation_metrics = [
            m for m in self.metrics_history if m.operation_name == operation_name
        ]

        if not operation_metrics:
            return {}

        durations = [m.duration for m in operation_metrics]
        gpu_utils = [m.gpu_utilization for m in operation_metrics]
        memory_utils = [m.memory_utilization for m in operation_metrics]

        return {
            "count": len(operation_metrics),
            "avg_duration": np.mean(durations),
            "max_duration": np.max(durations),
            "min_duration": np.min(durations),
            "avg_gpu_utilization": np.mean(gpu_utils),
            "max_gpu_utilization": np.max(gpu_utils),
            "avg_memory_utilization": np.mean(memory_utils),
            "max_memory_utilization": np.max(memory_utils),
        }

    def clear_metrics(self) -> None:
        """Clear the metrics history."""
        self.metrics_history.clear()
        while not self.metrics_queue.empty():
            self.metrics_queue.get()

    def get_recent_metrics(self, n: int = 10) -> List[PerformanceMetrics]:
        """
        Get the most recent performance metrics.

        Args:
            n (int): Number of recent metrics to return

        Returns:
            List[PerformanceMetrics]: List of recent metrics
        """
        return self.metrics_history[-n:]

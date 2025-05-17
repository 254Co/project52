"""
GPU Error Handling Module

This module provides comprehensive error handling for GPU operations including:
- Device error detection and recovery
- Error reporting and logging
- Automatic recovery mechanisms
- Error statistics tracking
"""

import logging
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import cupy as cp
import numpy as np


@dataclass
class GPUError:
    """Represents a GPU-related error with metadata."""

    timestamp: datetime
    error_type: str
    error_message: str
    device_id: int
    stack_trace: str
    recovery_attempted: bool
    recovery_successful: bool


class GPUErrorHandler:
    """
    Handles GPU-related errors and provides recovery mechanisms.

    This class provides a comprehensive error handling system for GPU operations,
    including error detection, reporting, and automatic recovery mechanisms.
    It also tracks error statistics and provides detailed error information.

    Attributes:
        device_id (int): GPU device ID to monitor
        error_history (List[GPUError]): History of GPU errors
        recovery_strategies (Dict[str, Callable]): Error recovery strategies
        logger (logging.Logger): Logger instance
        max_retries (int): Maximum number of recovery attempts
    """

    def __init__(self, device_id: int = 0, max_retries: int = 3):
        """
        Initialize the GPU error handler.

        Args:
            device_id (int): GPU device ID to monitor
            max_retries (int): Maximum number of recovery attempts
        """
        self.device_id = device_id
        self.device = cp.cuda.Device(device_id)
        self.error_history: List[GPUError] = []
        self.recovery_strategies: Dict[str, Callable] = {
            "CUDARuntimeError": self._handle_runtime_error,
            "OutOfMemoryError": self._handle_out_of_memory,
            "DeviceError": self._handle_device_error,
        }
        self.max_retries = max_retries
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
    def handle_errors(self, operation_name: str = "unnamed"):
        """
        Context manager for handling GPU errors.

        Args:
            operation_name (str): Name of the operation being performed

        Yields:
            None

        Raises:
            RuntimeError: If error recovery fails
        """
        try:
            yield
        except Exception as e:
            error = self._create_error(e)
            self.error_history.append(error)
            self.logger.error(f"GPU error in {operation_name}: {str(e)}")

            if isinstance(e, tuple(self.recovery_strategies.keys())):
                for _ in range(self.max_retries):
                    try:
                        self.recovery_strategies[type(e).__name__](e)
                        error.recovery_attempted = True
                        error.recovery_successful = True
                        self.logger.info(
                            f"Successfully recovered from error in {operation_name}"
                        )
                        return
                    except Exception as recovery_error:
                        self.logger.error(
                            f"Recovery attempt failed: {str(recovery_error)}"
                        )
                        error.recovery_attempted = True
                        error.recovery_successful = False

            raise RuntimeError(
                f"GPU operation failed and recovery attempts exhausted: {str(e)}"
            )

    def _create_error(self, error: Exception) -> GPUError:
        """
        Create a GPUError instance from an exception.

        Args:
            error (Exception): The exception to convert

        Returns:
            GPUError: Error instance with metadata
        """
        return GPUError(
            timestamp=datetime.now(),
            error_type=type(error).__name__,
            error_message=str(error),
            device_id=self.device_id,
            stack_trace=traceback.format_exc(),
            recovery_attempted=False,
            recovery_successful=False,
        )

    def _handle_runtime_error(self, error: cp.cuda.runtime.CUDARuntimeError) -> None:
        """
        Handle CUDA runtime errors.

        Args:
            error (cp.cuda.runtime.CUDARuntimeError): The runtime error

        Raises:
            RuntimeError: If recovery fails
        """
        self.logger.warning("Attempting to recover from CUDA runtime error")
        with self.device:
            cp.cuda.runtime.deviceReset()
            cp.cuda.runtime.deviceSynchronize()

    def _handle_out_of_memory(self, error: cp.cuda.runtime.OutOfMemoryError) -> None:
        """
        Handle out of memory errors.

        Args:
            error (cp.cuda.runtime.OutOfMemoryError): The out of memory error

        Raises:
            RuntimeError: If recovery fails
        """
        self.logger.warning("Attempting to recover from out of memory error")
        with self.device:
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    def _handle_device_error(self, error: cp.cuda.runtime.DeviceError) -> None:
        """
        Handle device errors.

        Args:
            error (cp.cuda.runtime.DeviceError): The device error

        Raises:
            RuntimeError: If recovery fails
        """
        self.logger.warning("Attempting to recover from device error")
        with self.device:
            cp.cuda.runtime.deviceReset()
            cp.cuda.runtime.deviceSynchronize()

    def get_error_stats(self) -> Dict[str, int]:
        """
        Get statistics about GPU errors.

        Returns:
            Dict[str, int]: Error statistics
        """
        stats = {
            "total_errors": len(self.error_history),
            "recovery_attempts": sum(
                1 for e in self.error_history if e.recovery_attempted
            ),
            "successful_recoveries": sum(
                1 for e in self.error_history if e.recovery_successful
            ),
        }

        # Count by error type
        error_types = {}
        for error in self.error_history:
            error_types[error.error_type] = error_types.get(error.error_type, 0) + 1
        stats["error_types"] = error_types

        return stats

    def clear_error_history(self) -> None:
        """Clear the error history."""
        self.error_history.clear()

    def get_recent_errors(self, n: int = 10) -> List[GPUError]:
        """
        Get the most recent GPU errors.

        Args:
            n (int): Number of recent errors to return

        Returns:
            List[GPUError]: List of recent errors
        """
        return self.error_history[-n:]

    def add_recovery_strategy(self, error_type: str, strategy: Callable) -> None:
        """
        Add a custom recovery strategy.

        Args:
            error_type (str): Type of error to handle
            strategy (Callable): Recovery strategy function
        """
        self.recovery_strategies[error_type] = strategy

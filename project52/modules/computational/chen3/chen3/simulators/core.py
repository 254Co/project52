# File: chen3/simulators/core.py
"""
Core Simulation Abstractions

This module provides the core abstractions for path simulation in the three-factor
Chen model. It defines the base class for path generators and a factory function
for creating appropriate simulator instances based on the desired computational
backend.

The module supports multiple computational backends:
1. CPU: Vectorized computation on CPU
2. GPU: GPU-accelerated computation
3. Ray: Distributed computation using Ray
4. Spark: Apache Spark-based distributed computation
"""

from abc import ABC, abstractmethod
from typing import Any


class PathGenerator(ABC):
    """
    Abstract base class for path generators in the three-factor Chen model.

    This class defines the interface that all path generators must implement.
    It provides a common structure for generating Monte Carlo paths across
    different computational backends while allowing for backend-specific
    optimizations.

    The class is designed to be extended by concrete implementations that
    provide specific simulation strategies for different computational
    environments (CPU, GPU, distributed).

    Attributes:
        model: The three-factor Chen model instance
        params: Simulation parameters and settings

    Example:
        >>> class MySimulator(PathGenerator):
        ...     def generate(self):
        ...         # Implement path generation logic
        ...         return paths
    """

    def __init__(self, model: Any, params: Any):
        """
        Initialize the path generator with model and parameters.

        Args:
            model: The three-factor Chen model instance
            params: Simulation parameters and settings
        """
        self.model = model
        self.params = params

    @abstractmethod
    def generate(self):
        """
        Generate Monte Carlo paths for the three-factor model.

        This method should be implemented by concrete subclasses to provide
        specific path generation logic for their computational backend.

        Returns:
            np.ndarray: Array of simulated paths with shape (n_paths, n_steps, n_factors)
                       where n_factors is the number of stochastic factors (3)

        Raises:
            NotImplementedError: Must be implemented by concrete subclasses
        """
        pass


def make_simulator(model, settings):
    """
    Factory function to create appropriate path generator based on settings.

    This function creates and returns a path generator instance appropriate
    for the specified computational backend. It supports multiple backends:
    - CPU: Vectorized computation on CPU
    - GPU: GPU-accelerated computation
    - Ray: Distributed computation using Ray
    - Spark: Apache Spark-based distributed computation

    Args:
        model: The three-factor Chen model instance
        settings: Simulation settings including backend specification

    Returns:
        PathGenerator: An instance of the appropriate path generator class

    Raises:
        ValueError: If the specified backend is not supported

    Example:
        >>> from chen3.config import Settings
        >>> settings = Settings(backend="cpu", n_paths=100000)
        >>> simulator = make_simulator(model, settings)
        >>> paths = simulator.generate()
    """
    backend = settings.backend
    if backend == "cpu":
        from .cpu_vectorized import CPUVectorizedSimulator

        return CPUVectorizedSimulator(model, settings)
    elif backend == "gpu":
        from .gpu_vectorized import GPUVectorizedSimulator

        return GPUVectorizedSimulator(model, settings)
    elif backend == "ray":
        from .ray_backend import RaySimulator

        return RaySimulator(model, settings)
    elif backend == "spark":
        from .spark_backend import SparkSimulator

        return SparkSimulator(model, settings)
    else:
        raise ValueError(f"Unknown backend: {backend}")

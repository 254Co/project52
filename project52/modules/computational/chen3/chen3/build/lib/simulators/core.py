# File: chen3/simulators/core.py
"""Simulation core abstractions."""
from abc import ABC, abstractmethod
from typing import Any

class PathGenerator(ABC):
    """Abstract path generator."""
    def __init__(self, model: Any, params: Any):
        self.model = model
        self.params = params

    @abstractmethod
    def generate(self):
        """Generate paths array of shape (n_paths, n_steps, n_factors)"""
        pass


def make_simulator(model, settings):
    """Factory to create appropriate PathGenerator based on settings."""
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

# -------------------- config.py --------------------
"""
Configuration Management Module for Chen3

This module provides configuration management for the Chen3 package using Pydantic.
It supports both Pydantic v1 and v2 through a compatibility layer. The Settings class
defines all configurable parameters for the simulation and pricing engine.

The configuration system allows for:
- Environment variable overrides
- Type validation
- Default values
- Runtime parameter validation
"""

import os
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings
from pydantic import Field
from typing import Literal

class Settings(BaseSettings):
    """
    Global configuration settings for the Chen3 package.
    
    This class manages all configurable parameters for the simulation engine,
    including computational settings, numerical parameters, and feature flags.
    
    Attributes:
        seed (int): Random number generator seed for reproducibility.
                   Default: 42
        
        backend (Literal["cpu", "gpu", "ray", "spark"]): Computational backend to use.
            - "cpu": Standard CPU-based computation
            - "gpu": GPU-accelerated computation (requires CUDA)
            - "ray": Distributed computation using Ray
            - "spark": Apache Spark-based distributed computation
            Default: "cpu"
        
        n_paths (int): Number of Monte Carlo simulation paths.
                      Higher values increase accuracy but require more computation.
                      Default: 200,000
        
        n_steps (int): Number of time steps per simulation path.
                      Higher values improve accuracy for path-dependent options.
                      Default: 480 (approximately 2 years of daily steps)
        
        dt (float): Time increment per step in years.
                   Default: 1/252 (daily steps assuming 252 trading days)
        
        use_cv (bool): Enable control variate technique for variance reduction.
                      Can significantly improve convergence for certain instruments.
                      Default: False
        
        use_aad (bool): Enable adjoint algorithmic differentiation for Greeks calculation.
                       Provides efficient computation of sensitivities.
                       Default: False
        
        cache_enabled (bool): Enable dependency caching for improved performance.
                             Useful for repeated calculations with same parameters.
                             Default: False
    """
    
    seed: int = Field(42, description="RNG seed for reproducibility")
    backend: Literal["cpu", "gpu", "ray", "spark"] = Field(
        "cpu",
        description="Computational backend (cpu/gpu/ray/spark)"
    )
    n_paths: int = Field(
        200_000,
        description="Number of Monte Carlo simulation paths",
        gt=0
    )
    n_steps: int = Field(
        480,
        description="Time steps per simulation path",
        gt=0
    )
    dt: float = Field(
        1/252,
        description="Time increment per step (in years)",
        gt=0
    )
    use_cv: bool = Field(
        False,
        description="Enable control variate for variance reduction"
    )
    use_aad: bool = Field(
        False,
        description="Enable adjoint AAD for efficient Greeks calculation"
    )
    cache_enabled: bool = Field(
        False,
        description="Enable dependency caching for improved performance"
    )

    class Config:
        """
        Pydantic configuration settings.
        
        Attributes:
            validate_assignment: Enable validation on attribute assignment
            arbitrary_types_allowed: Allow arbitrary types in the model
        """
        validate_assignment = True
        arbitrary_types_allowed = True
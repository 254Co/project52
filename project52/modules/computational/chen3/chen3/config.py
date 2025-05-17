# -------------------- config.py --------------------
"""
Configuration Management Module for the Chen Model

This module provides a comprehensive configuration system for the Chen3 package,
using Pydantic for robust parameter validation and management. It defines all
configurable parameters for model simulation, numerical methods, and performance
optimization.

The configuration system supports:
1. Computational Backend Selection:
   - NumPy for CPU-based computation
   - CUDA for GPU acceleration
   - Parallel processing options

2. Simulation Parameters:
   - Number of paths
   - Time steps
   - Discretization schemes
   - Random number generation

3. Performance Optimization:
   - Caching strategies
   - Memory management
   - Parallel processing
   - GPU utilization

4. Numerical Methods:
   - Integration schemes
   - Error control
   - Convergence criteria
   - Stability parameters

Mathematical Formulation:
------------------------
The configuration parameters affect the numerical implementation of the model:

1. Time Discretization:
   dt = T/n_steps
   where:
   - T: simulation time horizon
   - n_steps: number of time steps

2. Path Generation:
   N = n_paths * (1 + use_antithetic)
   where:
   - N: total number of paths
   - n_paths: base number of paths
   - use_antithetic: antithetic variates flag

3. Error Control:
   error_tol = max(absolute_tol, relative_tol * |value|)
   where:
   - absolute_tol: absolute error tolerance
   - relative_tol: relative error tolerance

Example Usage:
-------------
    >>> from chen3.config import Settings
    >>> import numpy as np
    >>>
    >>> # Create default settings
    >>> settings = Settings()
    >>>
    >>> # Customize settings
    >>> settings = Settings(
    ...     backend='numpy',
    ...     n_paths=10000,
    ...     n_steps=100,
    ...     dt=0.01,
    ...     use_control_variate=True,
    ...     use_antithetic=True,
    ...     use_parallel=True,
    ...     n_jobs=4,
    ...     cache_size=1000,
    ...     random_seed=42
    ... )
    >>>
    >>> # Use settings in model
    >>> from chen3 import ChenModel
    >>> model = ChenModel(params, settings=settings)
"""

from typing import Literal, Optional, Union
from pathlib import Path
from typing import Dict

import numpy as np
from pydantic import BaseModel, Field, field_validator

from .utils.exceptions import ConfigurationError
from .utils.logging import logger


class Settings(BaseModel):
    """
    Configuration settings for the Chen model.

    This class defines all configurable parameters for model simulation,
    numerical methods, and performance optimization. It uses Pydantic for
    parameter validation and type checking.

    Mathematical Properties:
    ----------------------
    1. Time Discretization:
       - Fixed time step: dt = T/n_steps
       - Adaptive time step: dt = min(dt_max, error_tol/|error|)

    2. Path Generation:
       - Base paths: n_paths
       - Antithetic paths: 2 * n_paths if use_antithetic
       - Control variate paths: n_paths if use_control_variate

    3. Error Control:
       - Absolute tolerance: absolute_tol
       - Relative tolerance: relative_tol
       - Combined: max(absolute_tol, relative_tol * |value|)

    Attributes:
        backend (str): Computational backend ('numpy' or 'cuda')
            Determines the numerical computation engine
            - 'numpy': CPU-based computation
            - 'cuda': GPU acceleration (if available)
        n_paths (int): Number of simulation paths
            Controls the accuracy of Monte Carlo simulation
            Typical range: 1000 to 100000
        n_steps (int): Number of time steps
            Controls the discretization of the time grid
            Typical range: 50 to 500
        dt (float): Time step size
            Fixed time step for Euler-Maruyama discretization
            Typical range: 0.001 to 0.1
        use_control_variate (bool): Whether to use control variates
            Reduces variance in Monte Carlo simulation
            Improves convergence rate
        use_antithetic (bool): Whether to use antithetic variates
            Reduces variance by using path pairs
            Doubles the effective number of paths
        use_parallel (bool): Whether to use parallel processing
            Enables multi-core computation
            Requires n_jobs > 1
        n_jobs (int): Number of parallel jobs
            Controls the level of parallelization
            -1: use all available cores
        cache_size (int): Size of the result cache
            Number of results to keep in memory
            Improves performance for repeated calculations
        random_seed (Optional[int]): Random number generator seed
            Ensures reproducibility of results
            None: use system time

    Example:
        >>> settings = Settings(
        ...     backend='numpy',
        ...     n_paths=10000,
        ...     n_steps=100,
        ...     dt=0.01,
        ...     use_control_variate=True,
        ...     use_antithetic=True,
        ...     use_parallel=True,
        ...     n_jobs=4,
        ...     cache_size=1000,
        ...     random_seed=42
        ... )
    """

    backend: Literal["numpy", "cuda"] = Field(
        default="numpy", description="Computational backend ('numpy' or 'cuda')"
    )

    n_paths: int = Field(
        default=10000, ge=1000, description="Number of simulation paths"
    )

    n_steps: int = Field(default=100, ge=50, description="Number of time steps")

    dt: float = Field(default=0.01, gt=0, le=0.1, description="Time step size")

    use_control_variate: bool = Field(
        default=True, description="Whether to use control variates"
    )

    use_antithetic: bool = Field(
        default=True, description="Whether to use antithetic variates"
    )

    use_parallel: bool = Field(
        default=True, description="Whether to use parallel processing"
    )

    n_jobs: int = Field(
        default=-1, ge=-1, description="Number of parallel jobs (-1 for all cores)"
    )

    cache_size: int = Field(default=1000, ge=0, description="Size of the result cache")

    random_seed: Optional[int] = Field(
        default=None, description="Random number generator seed"
    )

    @field_validator("n_jobs")
    def validate_n_jobs(cls, v, values):
        """
        Validate the number of parallel jobs.

        Args:
            v (int): Number of jobs
            values (dict): Other field values

        Returns:
            int: Validated number of jobs

        Raises:
            ValueError: If validation fails
        """
        if values.get("use_parallel") and v == 0:
            raise ValueError("n_jobs must be positive when use_parallel is True")
        return v

    @field_validator("backend")
    def validate_backend(cls, v):
        """
        Validate the computational backend.

        Args:
            v (str): Backend name

        Returns:
            str: Validated backend name

        Raises:
            ValueError: If validation fails
        """
        if v == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    raise ValueError("CUDA is not available")
            except ImportError:
                raise ValueError("PyTorch is required for CUDA backend")
        return v

    class Config:
        """
        Pydantic model configuration.

        This class configures the behavior of the Pydantic model,
        including validation, serialization, and documentation.
        """

        validate_assignment = True
        arbitrary_types_allowed = True
        extra = "forbid"


class SimulationConfig(BaseModel):
    """Configuration for simulation settings."""

    n_paths: int = Field(default=10000, gt=0, description="Number of simulation paths")
    n_steps: int = Field(default=100, gt=0, description="Number of time steps")
    dt: Optional[float] = Field(default=None, gt=0, description="Time step size")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    use_gpu: bool = Field(default=False, description="Whether to use GPU acceleration")
    batch_size: int = Field(default=1000, gt=0, description="Batch size for GPU processing")

    @field_validator("dt")
    def validate_dt(cls, v, values):
        if v is not None and v <= 0:
            raise ConfigurationError("Time step size must be positive")
        return v


class NumericalConfig(BaseModel):
    """Configuration for numerical settings."""

    tolerance: float = Field(default=1e-6, gt=0, description="Numerical tolerance")
    max_iterations: int = Field(default=1000, gt=0, description="Maximum iterations")
    adaptive_step: bool = Field(default=True, description="Use adaptive time stepping")
    min_step: float = Field(default=1e-4, gt=0, description="Minimum time step")
    max_step: float = Field(default=0.1, gt=0, description="Maximum time step")

    @field_validator("min_step", "max_step")
    def validate_step_size(cls, v):
        if v <= 0:
            raise ConfigurationError("Step size must be positive")
        return v

    @field_validator("max_step")
    def validate_max_step(cls, v, values):
        if "min_step" in values.data and v < values.data["min_step"]:
            raise ConfigurationError("Maximum step size must be greater than minimum step size")
        return v


class LoggingConfig(BaseModel):
    """Configuration for logging settings."""

    level: str = Field(default="INFO", description="Logging level")
    log_file: Optional[str] = Field(default="chen3.log", description="Log file path")
    log_dir: Optional[str] = Field(default=None, description="Log directory")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )

    @field_validator("level")
    def validate_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ConfigurationError(f"Invalid logging level: {v}")
        return v.upper()


class ChenConfig(BaseModel):
    """Main configuration container for the Chen3 package."""

    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    numerical: NumericalConfig = Field(default_factory=NumericalConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "ChenConfig":
        """Create configuration from dictionary."""
        try:
            return cls(**config_dict)
        except Exception as e:
            raise ConfigurationError(f"Failed to create configuration: {str(e)}")

    @classmethod
    def from_file(cls, config_file: Union[str, Path]) -> "ChenConfig":
        """Load configuration from file."""
        try:
            import yaml
            with open(config_file, "r") as f:
                config_dict = yaml.safe_load(f)
            return cls.from_dict(config_dict)
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from file: {str(e)}")

    def save_to_file(self, config_file: Union[str, Path]) -> None:
        """Save configuration to file."""
        try:
            import yaml
            with open(config_file, "w") as f:
                yaml.dump(self.model_dump(), f, default_flow_style=False)
        except Exception as e:
            raise ConfigurationError(f"Failed to save configuration to file: {str(e)}")

    def update(self, config_dict: Dict) -> None:
        """Update configuration with new values."""
        try:
            for key, value in config_dict.items():
                if hasattr(self, key):
                    if isinstance(value, dict):
                        getattr(self, key).update(value)
                    else:
                        setattr(self, key, value)
        except Exception as e:
            raise ConfigurationError(f"Failed to update configuration: {str(e)}")


# Create default configuration
default_config = ChenConfig()

def get_config() -> ChenConfig:
    """Get the current configuration."""
    return default_config

def set_config(config: ChenConfig) -> None:
    """Set the current configuration."""
    global default_config
    default_config = config
    logger.info("Configuration updated")

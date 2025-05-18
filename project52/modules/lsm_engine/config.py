"""
Configuration module for the LSM Engine.

This module defines the configuration settings for the Longstaff-Schwartz Method (LSM)
implementation. It provides a dataclass-based configuration system that can be
customized through environment variables.

The configuration includes settings for:
- Numerical precision (dtype)
- Discount rate for present value calculations
- Polynomial degree for regression
- Number of simulation paths
- Random seed for reproducibility
"""

# File: lsm_engine/config.py
from dataclasses import dataclass, field, fields
import os

@dataclass(slots=True, frozen=True)
class LSMConfig:
    """
    Configuration settings for the LSM Engine.
    
    Attributes:
        dtype (str): Numerical precision for calculations (default: "float64")
        discount_rate (float): Discount rate for present value calculations (default: 0.0)
        poly_degree (int): Degree of polynomial for regression (default: 3)
        n_paths (int): Number of simulation paths (default: 100,000)
        seed (int | None): Random seed for reproducibility (default: None)
    """
    dtype: str = field(default="float64")
    discount_rate: float = 0.0
    poly_degree: int = 3
    n_paths: int = 100_000
    seed: int | None = None

    @classmethod
    def from_env(cls) -> "LSMConfig":
        """
        Create a configuration instance from environment variables.
        
        Environment variables should be prefixed with 'LSM_' and uppercase.
        Example: LSM_POLY_DEGREE=4 sets the polynomial degree to 4.
        
        Returns:
            LSMConfig: A new configuration instance with values from environment variables
        """
        # Retrieve default values from dataclass fields to avoid slot descriptor issues
        defaults = {f.name: f.default for f in fields(cls)}
        overrides = {}
        for f in fields(cls):
            env_val = os.getenv(f"LSM_{f.name.upper()}")
            if env_val is not None:
                # Cast env var string to the field's type
                overrides[f.name] = f.type(env_val)  # type: ignore
        return cls(**{**defaults, **overrides})

# Global configuration instance
CONFIG = LSMConfig.from_env()
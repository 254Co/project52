"""
Configuration Module for the LSM Engine

This module provides configuration management for the Longstaff-Schwartz Method (LSM)
engine. It implements a robust configuration system that supports:
- Default values for all parameters
- Environment variable overrides
- Type-safe parameter validation
- Fallback mechanisms for critical parameters

The configuration is implemented as a frozen dataclass to ensure immutability
and thread safety. All parameters can be overridden through environment variables
prefixed with 'LSM_'.

Available configuration parameters:
- dtype: Numerical precision for calculations (default: "float64")
- discount_rate: Annual risk-free rate (default: 0.0, fallback: 0.05)
- poly_degree: Degree of polynomial regression (default: 3)
- n_paths: Number of Monte Carlo paths (default: 100,000)
- seed: Random number generator seed (default: None)

Environment variable names:
- LSM_DTYPE
- LSM_DISCOUNT_RATE
- LSM_POLY_DEGREE
- LSM_N_PATHS
- LSM_SEED
- LSM_DISCOUNT_RATE_FALLBACK (for fallback rate)
"""

# File: lsm_engine/config.py
from dataclasses import dataclass, field, fields
import os
import warnings

@dataclass(slots=True, frozen=True)
class LSMConfig:
    """
    Configuration for the LSM engine, with robust environment-variable overrides and fallback.
    
    This class implements a configuration system that combines default values with
    environment variable overrides. It provides type safety and validation for all
    parameters, with special handling for the discount rate fallback.
    
    The class is implemented as a frozen dataclass to ensure immutability and
    thread safety. All parameters can be overridden through environment variables
    prefixed with 'LSM_'.
    
    Attributes:
        dtype (str): Numerical precision for calculations (default: "float64")
        discount_rate (float): Annual risk-free rate (default: 0.0, fallback: 0.05)
        poly_degree (int): Degree of polynomial regression (default: 3)
        n_paths (int): Number of Monte Carlo paths (default: 100,000)
        seed (int | None): Random number generator seed (default: None)
    """
    dtype: str = field(default="float64")
    discount_rate: float = field(default=0.0)      # Annual risk-free rate (r)
    poly_degree: int = field(default=3)
    n_paths: int = field(default=100_000)
    seed: int | None = field(default=None)

    @classmethod
    def from_env(cls) -> "LSMConfig":
        """
        Create a configuration instance from environment variables.
        
        This method:
        1. Loads default values for all parameters
        2. Checks for environment variable overrides
        3. Validates and converts environment variable values
        4. Applies special handling for discount rate fallback
        5. Returns a new configuration instance
        
        Environment variables:
        - LSM_DTYPE: Numerical precision (e.g., "float64", "float32")
        - LSM_DISCOUNT_RATE: Annual risk-free rate
        - LSM_POLY_DEGREE: Polynomial regression degree
        - LSM_N_PATHS: Number of Monte Carlo paths
        - LSM_SEED: Random number generator seed
        - LSM_DISCOUNT_RATE_FALLBACK: Fallback rate if LSM_DISCOUNT_RATE is unset or zero
        
        Returns:
            LSMConfig: A new configuration instance with values from environment
            variables or defaults
            
        Raises:
            ValueError: If an environment variable value cannot be converted to
                      the required type
        """
        # Load defaults
        defaults = {f.name: f.default for f in fields(cls)}
        overrides: dict[str, object] = {}
        for f in fields(cls):
            env_key = f"LSM_{f.name.upper()}"
            env_val = os.getenv(env_key)
            if env_val is not None:
                try:
                    overrides[f.name] = f.type(env_val)  # type: ignore
                except Exception:
                    raise ValueError(
                        f"Invalid value for {env_key}: cannot cast '{env_val}' to {f.type}"
                    )
        # Fallback for discount_rate if unset or zero
        if 'discount_rate' not in overrides or overrides.get('discount_rate') == 0.0:
            # Default fallback rate (can override via LSM_DISCOUNT_RATE_FALLBACK)
            fb_key = 'LSM_DISCOUNT_RATE_FALLBACK'
            fb_val = os.getenv(fb_key)
            if fb_val is not None:
                try:
                    fb_rate = float(fb_val)
                except ValueError:
                    raise ValueError(f"Invalid fallback rate for {fb_key}: '{fb_val}'")
            else:
                fb_rate = 0.05
            warnings.warn(
                f"No LSM_DISCOUNT_RATE provided; falling back to {fb_rate}",
                UserWarning
            )
            overrides['discount_rate'] = fb_rate
        return cls(**{**defaults, **overrides})

# Global configuration instance
CONFIG = LSMConfig.from_env()
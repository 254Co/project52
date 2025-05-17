# File: chen3/config.py
"""Application settings via pydantic."""
from pydantic import BaseSettings, Field
from typing import Literal

class Settings(BaseSettings):
    seed: int = Field(42, description="RNG seed")
    backend: Literal["cpu", "gpu", "ray", "spark"] = "cpu"
    n_paths: int = Field(200_000, description="Number of Monte Carlo paths")
    n_steps: int = Field(480, description="Time steps per path")
    dt: float = Field(1/252, description="Time increment per step (in years)")
    use_cv: bool = Field(False, description="Enable control variate")
    use_aad: bool = Field(False, description="Enable adjoint AAD greeks")
    cache_enabled: bool = Field(False, description="Enable dependency caching")

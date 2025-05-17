# File: chen3/config.py
"""Application settings via pydantic."""
from pydantic import BaseSettings, Field, BaseModel
from typing import Literal, Optional

class SparkConfig(BaseModel):
    """Spark-specific configuration."""
    app_name: str = "Chen3SparkSimulator"
    master: str = "local[*]"  # Default to local mode
    executor_memory: str = "4g"
    driver_memory: str = "2g"
    max_result_size: str = "2g"
    shuffle_partitions: int = 200
    arrow_enabled: bool = True
    arrow_max_records: int = 10000
    chunk_size: int = 10000  # Default chunk size for path generation

class Settings(BaseSettings):
    seed: int = Field(42, description="RNG seed")
    backend: Literal["cpu", "gpu", "ray", "spark"] = "cpu"
    n_paths: int = Field(200_000, description="Number of Monte Carlo paths")
    n_steps: int = Field(480, description="Time steps per path")
    dt: float = Field(1/252, description="Time increment per step (in years)")
    use_cv: bool = Field(False, description="Enable control variate")
    use_aad: bool = Field(False, description="Enable adjoint AAD greeks")
    cache_enabled: bool = Field(False, description="Enable dependency caching")
    spark_config: Optional[SparkConfig] = None

    class Config:
        arbitrary_types_allowed = True

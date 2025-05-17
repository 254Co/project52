"""
Configuration management utilities for the Chen3 package.

This module provides tools for managing configuration settings, including
model parameters, simulation settings, and numerical methods configuration.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from .exceptions import ConfigurationError
from .logging import logger


@dataclass
class SimulationConfig:
    """Configuration for simulation settings."""

    num_paths: int = 10000
    num_steps: int = 252
    seed: Optional[int] = None
    antithetic: bool = True
    parallel: bool = True
    num_threads: Optional[int] = None

    def validate(self) -> None:
        """Validate simulation configuration."""
        if self.num_paths <= 0:
            raise ConfigurationError("Number of paths must be positive")
        if self.num_steps <= 0:
            raise ConfigurationError("Number of steps must be positive")
        if self.num_threads is not None and self.num_threads <= 0:
            raise ConfigurationError("Number of threads must be positive")


@dataclass
class NumericalConfig:
    """Configuration for numerical methods."""

    tolerance: float = 1e-6
    max_iterations: int = 1000
    method: str = "euler"  # euler, milstein, or exact
    adaptive: bool = True
    min_step_size: float = 1e-4
    max_step_size: float = 1.0

    def validate(self) -> None:
        """Validate numerical configuration."""
        if self.tolerance <= 0:
            raise ConfigurationError("Tolerance must be positive")
        if self.max_iterations <= 0:
            raise ConfigurationError("Maximum iterations must be positive")
        if self.method not in ["euler", "milstein", "exact"]:
            raise ConfigurationError("Invalid numerical method")
        if self.min_step_size <= 0 or self.max_step_size <= 0:
            raise ConfigurationError("Step sizes must be positive")
        if self.min_step_size >= self.max_step_size:
            raise ConfigurationError("Minimum step size must be less than maximum step size")


@dataclass
class ModelConfig:
    """Configuration for model parameters."""

    rate_params: Dict[str, float] = field(default_factory=dict)
    equity_params: Dict[str, float] = field(default_factory=dict)
    correlation_type: str = "constant"
    correlation_params: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """Validate model configuration."""
        if not self.rate_params:
            raise ConfigurationError("Rate parameters are required")
        if not self.equity_params:
            raise ConfigurationError("Equity parameters are required")
        if self.correlation_type not in ["constant", "time-dependent", "stochastic"]:
            raise ConfigurationError("Invalid correlation type")


@dataclass
class ChenConfig:
    """Main configuration class for the Chen3 package."""

    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    numerical: NumericalConfig = field(default_factory=NumericalConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    log_level: str = "INFO"
    log_file: Optional[str] = None
    cache_dir: Optional[str] = None

    def validate(self) -> None:
        """Validate the complete configuration."""
        self.simulation.validate()
        self.numerical.validate()
        self.model.validate()
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ConfigurationError("Invalid log level")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Convert configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_yaml(self) -> str:
        """Convert configuration to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ChenConfig":
        """Create configuration from dictionary."""
        simulation = SimulationConfig(**config_dict.get("simulation", {}))
        numerical = NumericalConfig(**config_dict.get("numerical", {}))
        model = ModelConfig(**config_dict.get("model", {}))
        return cls(
            simulation=simulation,
            numerical=numerical,
            model=model,
            log_level=config_dict.get("log_level", "INFO"),
            log_file=config_dict.get("log_file"),
            cache_dir=config_dict.get("cache_dir"),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "ChenConfig":
        """Create configuration from JSON string."""
        config_dict = json.loads(json_str)
        return cls.from_dict(config_dict)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "ChenConfig":
        """Create configuration from YAML string."""
        config_dict = yaml.safe_load(yaml_str)
        return cls.from_dict(config_dict)

    @classmethod
    def from_file(cls, file_path: Union[str, Path]) -> "ChenConfig":
        """Create configuration from file."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise ConfigurationError(f"Configuration file not found: {file_path}")

        with open(file_path, "r") as f:
            content = f.read()

        if file_path.suffix.lower() == ".json":
            return cls.from_json(content)
        elif file_path.suffix.lower() in [".yaml", ".yml"]:
            return cls.from_yaml(content)
        else:
            raise ConfigurationError(f"Unsupported configuration file format: {file_path.suffix}")

    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save configuration to file."""
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if file_path.suffix.lower() == ".json":
            content = self.to_json()
        elif file_path.suffix.lower() in [".yaml", ".yml"]:
            content = self.to_yaml()
        else:
            raise ConfigurationError(f"Unsupported configuration file format: {file_path.suffix}")

        with open(file_path, "w") as f:
            f.write(content)

        logger.info(f"Configuration saved to {file_path}")


def get_default_config() -> ChenConfig:
    """Get default configuration."""
    return ChenConfig()


def load_config(config_path: Optional[Union[str, Path]] = None) -> ChenConfig:
    """Load configuration from file or return default if not specified."""
    if config_path is None:
        return get_default_config()

    try:
        config = ChenConfig.from_file(config_path)
        config.validate()
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise ConfigurationError(f"Failed to load configuration: {e}") 
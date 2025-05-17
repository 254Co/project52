# -------------------- chen3/numerical_engines/__init__.py --------------------
"""
Numerical engines for the Chen3 model.

This module provides various numerical engines for the Chen3 model,
including Monte Carlo simulation, path generation, and integration schemes.
"""

# Optional GPU imports
import importlib.util

from .integration import IntegrationScheme
from .monte_carlo import MonteCarloEngine, MonteCarloEngineBase
from .path_generator import PathGenerator
from .quasi_mc import QuasiMonteCarloEngine
from .random import RandomNumberGenerator
from .variance_reduction import (
    AntitheticVariates,
    ControlVariates,
    ImportanceSampling,
    MultiLevelMonteCarlo,
    StratifiedSampling,
    VarianceReductionTechnique,
    create_variance_reduction,
)

if importlib.util.find_spec("cupy") is not None:
    from .gpu_mc_engine import GPUMonteCarloEngine
    from .multi_gpu_distributor import DistributedMonteCarloEngine

    HAS_GPU = True
else:
    HAS_GPU = False

__all__ = [
    "MonteCarloEngineBase",
    "MonteCarloEngine",
    "QuasiMonteCarloEngine",
    "PathGenerator",
    "IntegrationScheme",
    "RandomNumberGenerator",
    "VarianceReductionTechnique",
    "AntitheticVariates",
    "ControlVariates",
    "ImportanceSampling",
    "StratifiedSampling",
    "MultiLevelMonteCarlo",
    "create_variance_reduction",
]

if HAS_GPU:
    __all__.extend(["GPUMonteCarloEngine", "DistributedMonteCarloEngine"])

# -------------------- chen3/numerical_engines/__init__.py --------------------
"""
Numerical engines for the Chen3 model.

This module provides various numerical engines for the Chen3 model,
including Monte Carlo simulation, path generation, and integration schemes.
"""

from .monte_carlo import MonteCarloEngineBase, MonteCarloEngine
from .gpu_mc_engine import GPUMonteCarloEngine
from .quasi_mc import QuasiMonteCarloEngine
from .multi_gpu_distributor import DistributedMonteCarloEngine
from .path_generator import PathGenerator
from .integration import IntegrationScheme
from .random import RandomNumberGenerator
from .variance_reduction import (
    VarianceReductionTechnique,
    AntitheticVariates,
    ControlVariates,
    ImportanceSampling,
    StratifiedSampling,
    MultiLevelMonteCarlo,
    create_variance_reduction
)

__all__ = [
    'MonteCarloEngineBase',
    'MonteCarloEngine',
    'GPUMonteCarloEngine',
    'QuasiMonteCarloEngine',
    'DistributedMonteCarloEngine',
    'PathGenerator',
    'IntegrationScheme',
    'RandomNumberGenerator',
    'VarianceReductionTechnique',
    'AntitheticVariates',
    'ControlVariates',
    'ImportanceSampling',
    'StratifiedSampling',
    'MultiLevelMonteCarlo',
    'create_variance_reduction'
]
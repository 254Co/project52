# -------------------- chen3/numerical_engines/__init__.py --------------------
"""
Numerical engines for the Chen3 model.

This module provides various numerical engines for the Chen3 model,
including Monte Carlo simulation, path generation, and integration schemes.
"""

from .monte_carlo import MonteCarloEngineBase, MonteCarloEngine
from .quasi_mc import QuasiMonteCarloEngine
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

# Optional GPU imports
import importlib.util
if importlib.util.find_spec('cupy') is not None:
    from .gpu_mc_engine import GPUMonteCarloEngine
    from .multi_gpu_distributor import DistributedMonteCarloEngine
    HAS_GPU = True
else:
    HAS_GPU = False

__all__ = [
    'MonteCarloEngineBase',
    'MonteCarloEngine',
    'QuasiMonteCarloEngine',
    'PathGenerator',
    'IntegrationScheme',
    'RandomNumberGenerator',
    'VarianceReductionTechnique',
    'AntitheticVariates',
    'ControlVariates',
    'ImportanceSampling',
    'StratifiedSampling',
    'MultiLevelMonteCarlo',
    'create_variance_reduction',
]

if HAS_GPU:
    __all__.extend(['GPUMonteCarloEngine', 'DistributedMonteCarloEngine'])
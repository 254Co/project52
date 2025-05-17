"""
Correlation Module for the Chen3 Model

This module provides various correlation structures for the Chen3 model,
including time-dependent, state-dependent, regime-switching, stochastic,
copula-based, network-based, fractal, wavelet, spectral, hierarchical,
and dynamic correlations.

The module is organized into the following submodules:

1. core: Base classes and interfaces
2. models: Concrete correlation implementations
3. utils: Utility functions and helpers
4. tests: Unit tests and examples

Example Usage:
-------------
    >>> from chen3.correlation import TimeDependentCorrelation
    >>> import numpy as np
    >>>
    >>> # Create time-dependent correlation
    >>> time_points = np.array([0.0, 1.0, 2.0])
    >>> corr_matrices = [
    ...     np.array([[1.0, 0.5, 0.3],
    ...              [0.5, 1.0, 0.2],
    ...              [0.3, 0.2, 1.0]]),
    ...     np.array([[1.0, 0.6, 0.4],
    ...              [0.6, 1.0, 0.3],
    ...              [0.4, 0.3, 1.0]]),
    ...     np.array([[1.0, 0.7, 0.5],
    ...              [0.7, 1.0, 0.4],
    ...              [0.5, 0.4, 1.0]])
    ... ]
    >>> time_corr = TimeDependentCorrelation(
    ...     time_points=time_points,
    ...     correlation_matrices=corr_matrices
    ... )
    >>>
    >>> # Get correlation matrix at time t=0.5
    >>> corr = time_corr.get_correlation_matrix(t=0.5)
"""

from .core.base import BaseCorrelation

# Basic correlation models
from .models.time_dependent import TimeDependentCorrelation
from .models.state_dependent import (
    StateDependentCorrelation,
    linear_state_correlation,
    exponential_state_correlation
)
from .models.regime_switching import (
    RegimeSwitchingCorrelation,
    create_two_regime_correlation
)
from .models.stochastic import (
    StochasticCorrelation,
    create_stochastic_correlation
)
from .models.copula import (
    CopulaCorrelation,
    create_gaussian_copula_correlation,
    create_student_copula_correlation
)

# Advanced correlation models
from .models.network import (
    NetworkCorrelation,
    create_network_correlation
)
from .models.fractal import (
    FractalCorrelation,
    create_fractal_correlation
)
from .models.wavelet import (
    WaveletCorrelation,
    create_wavelet_correlation
)
from .models.spectral import (
    SpectralCorrelation,
    create_spectral_correlation
)
from .models.hierarchical import (
    HierarchicalCorrelation,
    create_hierarchical_correlation
)
from .models.dynamic import (
    DynamicCorrelation,
    create_dynamic_correlation
)

__all__ = [
    # Base class
    'BaseCorrelation',
    
    # Basic correlation models
    'TimeDependentCorrelation',
    'StateDependentCorrelation',
    'linear_state_correlation',
    'exponential_state_correlation',
    'RegimeSwitchingCorrelation',
    'create_two_regime_correlation',
    'StochasticCorrelation',
    'create_stochastic_correlation',
    'CopulaCorrelation',
    'create_gaussian_copula_correlation',
    'create_student_copula_correlation',
    
    # Advanced correlation models
    'NetworkCorrelation',
    'create_network_correlation',
    'FractalCorrelation',
    'create_fractal_correlation',
    'WaveletCorrelation',
    'create_wavelet_correlation',
    'SpectralCorrelation',
    'create_spectral_correlation',
    'HierarchicalCorrelation',
    'create_hierarchical_correlation',
    'DynamicCorrelation',
    'create_dynamic_correlation'
] 
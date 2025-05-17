"""
Correlation Models for the Chen3 Package

This package provides various correlation structures for the Chen3 model,
implementing different approaches to modeling dependencies between factors.
These correlation structures allow for flexible and realistic modeling of
market relationships and dependencies.

Available Correlation Models:
--------------------------
1. Network Correlation (NetworkCorrelation)
   - Graph theory-based correlation structure
   - Edge weights and path lengths define correlations
   - Supports time-dependent and state-dependent weights
   - Useful for modeling complex market dependencies

2. Fractal Correlation (FractalCorrelation)
   - Fractal geometry-based correlation structure
   - Hurst exponent controls correlation decay
   - Captures long-range dependencies
   - Suitable for modeling self-similar market patterns

3. Wavelet Correlation (WaveletCorrelation)
   - Wavelet decomposition-based correlation structure
   - Multi-scale correlation modeling
   - Different wavelet families supported
   - Effective for modeling scale-dependent relationships

4. Spectral Correlation (SpectralCorrelation)
   - Spectral decomposition-based correlation structure
   - Principal component-based correlations
   - Time-dependent eigenvalues
   - Useful for factor-based correlation modeling

Usage:
-----
These correlation structures can be used individually or in combination
to create sophisticated correlation models. For example:

>>> from chen3.correlation.models import NetworkCorrelation, FractalCorrelation
>>> # Create a network correlation
>>> network_corr = NetworkCorrelation(
...     n_factors=3,
...     weight_type='decay',
...     decay_rate=0.1
... )
>>> # Create a fractal correlation
>>> fractal_corr = FractalCorrelation(
...     n_factors=3,
...     hurst_type='constant',
...     decay_rate=0.1
... )

Mathematical Properties:
---------------------
All correlation structures ensure:
1. Positive definiteness of correlation matrices
2. Symmetry of correlations
3. Unit diagonal elements
4. Valid correlation values in [-1, 1]

Time Evolution:
------------
Each correlation structure supports time-dependent and state-dependent
correlations through configurable functions that can model:
- Market regime changes
- Time-varying dependencies
- State-dependent relationships
- Dynamic correlation patterns
"""

from .fractal import FractalCorrelation, create_fractal_correlation
from .network import NetworkCorrelation, create_network_correlation
from .spectral import SpectralCorrelation, create_spectral_correlation
from .wavelet import WaveletCorrelation, create_wavelet_correlation

__all__ = [
    # Network correlation
    "NetworkCorrelation",
    "create_network_correlation",
    # Fractal correlation
    "FractalCorrelation",
    "create_fractal_correlation",
    # Wavelet correlation
    "WaveletCorrelation",
    "create_wavelet_correlation",
    # Spectral correlation
    "SpectralCorrelation",
    "create_spectral_correlation",
]

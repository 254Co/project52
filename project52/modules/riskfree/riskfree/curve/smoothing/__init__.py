"""Yield curve smoothing methods for risk-free rate curves.

This package provides a collection of methods for smoothing yield curves,
each implementing different mathematical approaches to curve fitting.
The methods range from simple polynomial fits to sophisticated parametric
models like Svensson.

Available Methods
----------------
1. SvenssonSmoother: Extended Nelson-Siegel model with additional curvature
2. PolynomialSmoother: Simple polynomial regression
3. CubicSplineSmoother: Piecewise cubic polynomial interpolation
4. GPSmoother: Gaussian Process regression with RBF kernel
5. ExponentialSmoother: Exponential decay model
6. KernelSmoother: Non-parametric kernel regression
7. PSplineSmoother: Penalized B-spline regression
8. LoessSmoother: Local polynomial regression

Usage
-----
>>> from riskfree.curve.smoothing import get_smoother
>>> smooth = get_smoother("svensson").fit(times, yields)
>>> y_hat = smooth.predict(new_times)

Notes
-----
- All smoothers implement the BaseSmoother interface
- Methods can be accessed via get_smoother() factory function
- Each method has its own strengths and use cases
- Some methods may require additional dependencies
"""

from .base import BaseSmoother, get_smoother      # noqa: F401
from .svensson import SvenssonSmoother            # noqa: F401
from .polynomial import PolynomialSmoother        # noqa: F401
from .spline import CubicSplineSmoother           # noqa: F401
from .gaussian_process import GPSmoother          # noqa: F401
from .exponential import ExponentialSmoother      # noqa: F401
from .kernel import KernelSmoother                # noqa: F401
from .pspline import PSplineSmoother              # noqa: F401
from .loess import LoessSmoother                  # noqa: F401

__all__ = [
    "BaseSmoother",
    "SvenssonSmoother",
    "PolynomialSmoother",
    "CubicSplineSmoother",
    "GPSmoother",
    "ExponentialSmoother",
    "KernelSmoother",
    "PSplineSmoother",
    "LoessSmoother",
    "get_smoother",
]

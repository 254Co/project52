"""Penalized spline (GAM) implementation for yield curve smoothing.

This module implements a penalized spline approach to yield curve smoothing
using Generalized Additive Models (GAMs). This method combines the flexibility
of splines with a penalty term to control smoothness.

The penalized spline model:
    - Uses B-spline basis functions
    - Includes a penalty term for smoothness
    - Automatically selects optimal smoothing parameter
    - Provides a good balance between fit and smoothness

Advantages:
    1. Flexible functional form
    2. Automatic smoothness control
    3. Good statistical properties
    4. Handles non-linear relationships well

Note:
    This implementation uses pygam's LinearGAM class and requires the
    pygam package to be installed.
"""

import numpy as np
from pygam import LinearGAM, s
from .base import BaseSmoother, register, ArrayLike


@register("pspline")
class PSplineSmoother(BaseSmoother):
    """Penalized spline (GAM) smoother for yield curves.
    
    This class implements a penalized spline approach to yield curve
    smoothing using Generalized Additive Models. It combines B-spline
    basis functions with a penalty term to control the smoothness of
    the fitted curve.
    
    The model is particularly useful when:
    1. A flexible but controlled fit is needed
    2. Automatic smoothness selection is desired
    3. Statistical properties are important
    4. Non-linear relationships are present
    
    Attributes:
        _gam (LinearGAM): Fitted GAM model
    """

    def __init__(self):
        """Initialize a new penalized spline smoother.
        
        The smoother uses a single smoothing spline term with automatic
        selection of the smoothing parameter via generalized cross-validation.
        """
        super().__init__()
        self._gam: LinearGAM | None = None

    def fit(self, times: ArrayLike, yields: ArrayLike):
        """Fit penalized spline model to observed yields.
        
        This method fits a GAM model to the observed yields, automatically
        selecting the optimal smoothing parameter using generalized
        cross-validation.
        
        Args:
            times: Array of tenors in years
            yields: Array of observed yields (same length as times)
            
        Returns:
            self for method chaining
            
        Note:
            The model uses a single smoothing spline term (s(0)) with
            automatic selection of the smoothing parameter. The penalty
            term helps prevent overfitting while maintaining flexibility.
        """
        t = np.asarray(times, dtype=float)
        y = np.asarray(yields, dtype=float)
        self._gam = LinearGAM(s(0)).fit(t, y)
        self._fitted = True
        return self

    def predict(self, times: ArrayLike):
        """Generate smoothed yields for given tenors.
        
        Args:
            times: Array of tenors to predict yields for
            
        Returns:
            Array of predicted yields
            
        Raises:
            RuntimeError: If called before fit()
            
        Note:
            The predictions are computed using the fitted GAM model,
            which provides a smooth curve that balances fit to the
            observed data with overall smoothness.
        """
        self._check()
        return self._gam.predict(times)

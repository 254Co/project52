"""Exponential smoothing implementation for yield curves.

This module implements a Holt-Winters exponential smoothing approach to
yield curve smoothing. This method is particularly effective for curves
that exhibit a clear trend but no seasonal patterns.

The exponential smoothing model:
    - Uses weighted averages of past observations
    - Can incorporate trend components
    - Adapts to changes in the level and trend
    - Provides a simple but effective smoothing approach

Advantages:
    1. Simple and interpretable
    2. Adapts to changes in the curve
    3. Handles trend components well
    4. Computationally efficient

Note:
    This implementation uses statsmodels' ExponentialSmoothing class
    and assumes no seasonal patterns in the yield curve.
"""

import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from .base import BaseSmoother, register, ArrayLike


@register("exp_smoothing")
class ExponentialSmoother(BaseSmoother):
    """Holt-Winters exponential smoother for yield curves.
    
    This class implements a Holt-Winters exponential smoothing approach
    to yield curve smoothing. It uses weighted averages of past observations
    to generate a smooth curve that adapts to changes in the level and trend.
    
    The model is particularly useful when:
    1. The yield curve exhibits a clear trend
    2. Simple and interpretable smoothing is desired
    3. Computational efficiency is important
    4. No seasonal patterns are present
    
    Attributes:
        trend (str): Trend component type ("add" or "mul")
        _fit_res: Fitted exponential smoothing model
    """

    def __init__(self, trend: str | None = "add"):
        """Initialize a new exponential smoother.
        
        Args:
            trend: Trend component type:
                  - "add": Additive trend
                  - "mul": Multiplicative trend
                  - None: No trend component
        """
        super().__init__()
        self.trend = trend
        self._fit_res = None

    def fit(self, times: ArrayLike, yields: ArrayLike):
        """Fit exponential smoothing model to observed yields.
        
        This method fits a Holt-Winters exponential smoothing model to the
        observed yields. The model is fitted to the sorted yields, as the
        Holt-Winters method assumes ordered observations.
        
        Args:
            times: Array of tenors in years
            yields: Array of observed yields (same length as times)
            
        Returns:
            self for method chaining
            
        Note:
            The model ignores the actual tenor values and assumes ordered
            observations. The yields are sorted before fitting to ensure
            proper trend estimation.
        """
        # Holt-Winters ignores x values; ensure sorted
        idx = np.argsort(times)
        y_sorted = np.asarray(yields, dtype=float)[idx]
        self._fit_res = ExponentialSmoothing(
            y_sorted, trend=self.trend, seasonal=None
        ).fit()
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
            Since the model ignores actual tenor values, the predictions
            are mapped to the requested tenors using a simple index-based
            approach. This may not be ideal for extrapolation.
        """
        self._check()
        n = len(times)
        return self._fit_res.fittedvalues[-n:]  # rough mapping

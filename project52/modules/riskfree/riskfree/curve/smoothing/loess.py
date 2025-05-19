"""Locally weighted scatter-plot smoothing (LOWESS) implementation.

This module implements a LOWESS (Locally Weighted Scatter-plot Smoothing)
approach to yield curve smoothing. This non-parametric method uses local
polynomial regression with robust weighting to handle outliers.

The LOWESS model:
    - Uses local polynomial regression
    - Applies robust weighting to handle outliers
    - Iteratively refines the fit
    - Provides a smooth, locally adaptive curve

Advantages:
    1. Robust to outliers
    2. Adapts to local structure
    3. No assumption about functional form
    4. Handles non-linear relationships well

Note:
    This implementation uses statsmodels' lowess function and provides
    a good balance between flexibility and robustness.
"""

import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from .base import BaseSmoother, register, ArrayLike


@register("loess")
class LoessSmoother(BaseSmoother):
    """Locally weighted scatter-plot smoother (LOWESS) for yield curves.
    
    This class implements a LOWESS approach to yield curve smoothing.
    It uses local polynomial regression with robust weighting to handle
    outliers and adapt to local structure in the data.
    
    The model is particularly useful when:
    1. The data may contain outliers
    2. Local structure needs to be preserved
    3. A robust smoothing approach is desired
    4. Non-linear relationships are present
    
    Attributes:
        frac (float): Fraction of points to use in local regression
        it (int): Number of robustifying iterations
        _sorted (tuple): Sorted (times, yields) for prediction
    """

    def __init__(self, frac: float = 0.25, it: int = 3):
        """Initialize a new LOWESS smoother.
        
        Args:
            frac: Fraction of points to use in local regression.
                  Higher values produce smoother curves.
            it: Number of robustifying iterations. Higher values
                make the fit more robust to outliers.
        """
        super().__init__()
        self.frac = frac
        self.it = it
        self._sorted = None
        self._fitted = False

    def fit(self, times: ArrayLike, yields: ArrayLike):
        """Store data for LOWESS smoothing.
        
        This method stores the input data for later use in prediction.
        The actual LOWESS smoothing is performed during prediction to
        ensure proper handling of new points.
        
        Args:
            times: Array of tenors in years
            yields: Array of observed yields (same length as times)
            
        Returns:
            self for method chaining
        """
        t = np.asarray(times, dtype=float)
        y = np.asarray(yields, dtype=float)
        self._sorted = (t, y)
        self._fitted = True
        return self

    def predict(self, times: ArrayLike):
        """Generate smoothed yields for given tenors.
        
        This method performs LOWESS smoothing on the stored data and
        interpolates the results to the requested tenors.
        
        Args:
            times: Array of tenors to predict yields for
            
        Returns:
            Array of predicted yields
            
        Raises:
            RuntimeError: If called before fit()
            
        Note:
            The predictions are computed by:
            1. Performing LOWESS smoothing on the stored data
            2. Interpolating the smoothed values to the requested tenors
            This approach ensures proper handling of new points while
            maintaining the benefits of LOWESS smoothing.
        """
        self._check()
        t, y = self._sorted
        preds = lowess(y, t, frac=self.frac, it=self.it, return_sorted=False)
        # map via simple interpolation
        return np.interp(times, t, preds)

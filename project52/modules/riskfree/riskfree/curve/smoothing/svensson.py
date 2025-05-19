"""Svensson yield curve smoothing implementation.

This module implements the Svensson (1994) model for yield curve smoothing,
which extends the Nelson-Siegel model with an additional curvature term.
The model is particularly well-suited for fitting yield curves with complex
shapes and multiple humps.

The model uses six parameters:
- β0: Level (long-term rate)
- β1: Slope (short-term vs long-term spread)
- β2: First curvature (medium-term deviation)
- β3: Second curvature (additional hump)
- τ1: First decay parameter
- τ2: Second decay parameter

Reference:
    Svensson, L. E. O. (1994). Estimating and interpreting forward interest
    rates: Sweden 1992-1994. NBER Working Paper Series, No. 4871.
"""

import numpy as np
from scipy.optimize import least_squares
from .base import BaseSmoother, register, ArrayLike


@register("svensson")
class SvenssonSmoother(BaseSmoother):
    """Svensson 6-parameter yield curve smoother.
    
    This class implements the Svensson model for yield curve smoothing,
    which extends the Nelson-Siegel model with an additional curvature term.
    The model is particularly effective at fitting yield curves with complex
    shapes and multiple humps.
    
    Model:
        r(t) = β0 +
               β1 * ((1 - e^{-t/τ1}) / (t/τ1)) +
               β2 * (((1 - e^{-t/τ1}) / (t/τ1)) - e^{-t/τ1}) +
               β3 * (((1 - e^{-t/τ2}) / (t/τ2)) - e^{-t/τ2})
               
    where:
        - β0: Level (long-term rate)
        - β1: Slope (short-term vs long-term spread)
        - β2: First curvature (medium-term deviation)
        - β3: Second curvature (additional hump)
        - τ1: First decay parameter
        - τ2: Second decay parameter
        
    The model is fitted using non-linear least squares optimization with
    reasonable bounds on the parameters to ensure stability.
    
    Attributes:
        params (np.ndarray): Fitted model parameters [β0, β1, β2, β3, τ1, τ2]
    """

    def __init__(self, guess: ArrayLike = (0.02, -0.02, 0.02, 0.0, 1.0, 3.0)):
        """Initialize a new Svensson smoother.
        
        Args:
            guess: Initial parameter guess [β0, β1, β2, β3, τ1, τ2].
                  Default values are reasonable starting points for
                  typical yield curves.
        """
        super().__init__()
        self.params = np.asarray(guess, dtype=float)

    # --------------------------------------------------------------------- #
    def _svensson(self, t, p):
        """Calculate Svensson model rates for given tenors.
        
        Args:
            t: Array of tenors in years
            p: Model parameters [β0, β1, β2, β3, τ1, τ2]
            
        Returns:
            Array of predicted rates
        """
        β0, β1, β2, β3, τ1, τ2 = p
        term1 = (1 - np.exp(-t / τ1)) / (t / τ1)
        term2 = term1 - np.exp(-t / τ1)
        term3 = (1 - np.exp(-t / τ2)) / (t / τ2) - np.exp(-t / τ2)
        return β0 + β1 * term1 + β2 * term2 + β3 * term3

    # --------------------------------------------------------------------- #
    def fit(self, times: ArrayLike, yields: ArrayLike):
        """Fit the Svensson model to observed yields.
        
        This method uses non-linear least squares to find the optimal
        parameters that minimize the sum of squared errors between the
        model and observed yields.
        
        Args:
            times: Array of tenors in years
            yields: Array of observed yields (same length as times)
            
        Returns:
            self for method chaining
            
        Note:
            The optimization uses the following bounds:
            - β parameters: [-1, 1]
            - τ parameters: [0.001, 100]
        """
        t = np.asarray(times, dtype=float)
        y = np.asarray(yields, dtype=float)

        def _resid(p):
            return self._svensson(t, p) - y

        res = least_squares(_resid, x0=self.params, bounds=([-1, -1, -1, -1, 1e-3, 1e-3],
                                                            [1, 1, 1, 1, 100, 100]))
        self.params = res.x
        self._fitted = True
        return self

    # --------------------------------------------------------------------- #
    def predict(self, times: ArrayLike):
        """Generate smoothed yields for given tenors.
        
        Args:
            times: Array of tenors to predict yields for
            
        Returns:
            Array of predicted yields
            
        Raises:
            RuntimeError: If called before fit()
        """
        self._check()
        t = np.asarray(times, dtype=float)
        return self._svensson(t, self.params)

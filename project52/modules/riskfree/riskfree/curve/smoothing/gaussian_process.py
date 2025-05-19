"""Gaussian Process yield curve smoothing implementation.

This module implements a Gaussian Process regression approach to yield curve
smoothing using a Radial Basis Function (RBF) kernel. This non-parametric
method provides both point estimates and uncertainty estimates for the
smoothed curve.

The Gaussian Process model:
    - Uses an RBF kernel to capture smoothness in the yield curve
    - Provides uncertainty estimates through the posterior variance
    - Automatically adapts to the scale of the data
    - Handles noisy observations through the alpha parameter

Advantages:
    1. Provides uncertainty estimates
    2. Adapts to local structure in the data
    3. Handles noisy observations well
    4. No need to specify functional form

Note:
    This implementation requires scikit-learn and may be computationally
    intensive for large datasets.
"""

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from .base import BaseSmoother, register, ArrayLike


@register("gp")
class GPSmoother(BaseSmoother):
    """Gaussian Process yield curve smoother with RBF kernel.
    
    This class implements a Gaussian Process regression approach to yield
    curve smoothing. It uses a combination of a constant kernel and an RBF
    kernel to model the yield curve, providing both point estimates and
    uncertainty estimates.
    
    The model is particularly useful when:
    1. Uncertainty estimates are needed
    2. The data is noisy
    3. The functional form is unknown
    4. Local structure needs to be preserved
    
    Attributes:
        gp (GaussianProcessRegressor): Fitted GP model
        alpha (float): Noise level in the observations
    """

    def __init__(self, alpha: float = 1e-6):
        """Initialize a new GP smoother.
        
        Args:
            alpha: Noise level in the observations. Higher values make the
                  model more robust to noise but may oversmooth the curve.
        """
        super().__init__()
        # Define kernel as product of constant and RBF kernels
        # Constant kernel: C(1.0, (1e-3, 1e3)) - signal variance
        # RBF kernel: RBF(1.0, (1e-2, 1e2)) - length scale
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True)

    def fit(self, times: ArrayLike, yields: ArrayLike):
        """Fit GP model to observed yields.
        
        This method fits a Gaussian Process model to the observed yields,
        optimizing the kernel hyperparameters to maximize the marginal
        likelihood of the data.
        
        Args:
            times: Array of tenors in years
            yields: Array of observed yields (same length as times)
            
        Returns:
            self for method chaining
            
        Note:
            The model automatically optimizes the kernel hyperparameters
            during fitting to adapt to the scale and structure of the data.
        """
        t = np.asarray(times, dtype=float).reshape(-1, 1)
        y = np.asarray(yields, dtype=float)
        self.gp.fit(t, y)
        self._fitted = True
        return self

    def predict(self, times: ArrayLike, return_std: bool = False):
        """Generate smoothed yields for given tenors.
        
        Args:
            times: Array of tenors to predict yields for
            return_std: Whether to return standard deviations of predictions
            
        Returns:
            If return_std is False:
                Array of predicted yields
            If return_std is True:
                Tuple of (predicted yields, standard deviations)
                
        Raises:
            RuntimeError: If called before fit()
            
        Note:
            The standard deviations provide a measure of uncertainty in the
            predictions, with higher values indicating less confidence.
        """
        self._check()
        x = np.asarray(times, dtype=float).reshape(-1, 1)
        return self.gp.predict(x, return_std=return_std)

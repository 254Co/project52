"""Kernel regression implementation for yield curve smoothing.

This module implements a non-parametric kernel regression approach to
yield curve smoothing using a Gaussian kernel. This method provides
a flexible way to estimate the yield curve without assuming a specific
functional form.

The kernel regression model:
    - Uses local weighted averages of observations
    - Bandwidth selection via cross-validation
    - Gaussian kernel for smooth weighting
    - Non-parametric approach to curve fitting

Advantages:
    1. No assumption about functional form
    2. Adapts to local structure in the data
    3. Handles non-linear relationships well
    4. Provides smooth estimates

Note:
    This implementation uses statsmodels' KernelReg class and automatically
    selects the optimal bandwidth using cross-validation.
"""

import numpy as np
from statsmodels.nonparametric.kernel_regression import KernelReg
from .base import BaseSmoother, register, ArrayLike


@register("kernel")
class KernelSmoother(BaseSmoother):
    """Gaussian kernel regression smoother for yield curves.
    
    This class implements a non-parametric kernel regression approach to
    yield curve smoothing. It uses a Gaussian kernel to compute locally
    weighted averages of the observed yields.
    
    The model is particularly useful when:
    1. The functional form is unknown
    2. Local structure needs to be preserved
    3. Non-linear relationships are present
    4. A flexible smoothing approach is desired
    
    Attributes:
        bw (str): Bandwidth selection method
        _kr (KernelReg): Fitted kernel regression model
        _times (np.ndarray): Original tenors used for fitting
    """

    def __init__(self, bw: str = "cv_ls"):
        """Initialize a new kernel smoother.
        
        Args:
            bw: Bandwidth selection method:
                - "cv_ls": Least squares cross-validation
                - "aic": Akaike Information Criterion
                - "normal_reference": Normal reference rule
        """
        super().__init__()
        self.bw = bw
        self._kr: KernelReg | None = None
        self._times: np.ndarray | None = None

    def fit(self, times: ArrayLike, yields: ArrayLike):
        """Fit kernel regression model to observed yields.
        
        This method fits a kernel regression model to the observed yields,
        automatically selecting the optimal bandwidth using the specified
        method (default: cross-validation).
        
        Args:
            times: Array of tenors in years
            yields: Array of observed yields (same length as times)
            
        Returns:
            self for method chaining
            
        Note:
            The bandwidth selection is crucial for the performance of the
            model. Cross-validation (cv_ls) is recommended for most cases
            as it adapts to the local structure of the data.
        """
        t = np.asarray(times, dtype=float)
        y = np.asarray(yields, dtype=float)
        self._kr = KernelReg(y, t, var_type="c", bw=self.bw)
        self._times = t
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
            The predictions are computed using locally weighted averages
            of the observed yields, with weights determined by the Gaussian
            kernel and the selected bandwidth.
        """
        self._check()
        t_new = np.asarray(times, dtype=float)
        m, _ = self._kr.fit(t_new)
        return m

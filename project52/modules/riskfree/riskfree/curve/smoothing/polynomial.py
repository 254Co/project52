"""Polynomial yield curve smoothing implementation.

This module implements a simple polynomial regression approach to yield curve
smoothing. While less sophisticated than parametric models like Svensson,
polynomial fitting can be effective for curves with simple shapes and when
computational efficiency is important.

The polynomial model:
    r(t) = c₀ + c₁t + c₂t² + ... + cₙtⁿ

where:
    - cᵢ are the polynomial coefficients
    - n is the polynomial degree
    - t is the tenor in years

Note:
    Higher degree polynomials can lead to overfitting and oscillatory behavior
    at the ends of the curve. A degree of 3-4 is often sufficient for typical
    yield curves.
"""

import numpy as np
from .base import BaseSmoother, register, ArrayLike


@register("polynomial")
class PolynomialSmoother(BaseSmoother):
    """Polynomial regression yield curve smoother.
    
    This class implements a simple polynomial regression approach to yield
    curve smoothing. It fits an n-degree polynomial to the observed yields
    using least squares regression.
    
    The model is particularly useful when:
    1. The yield curve has a simple shape
    2. Computational efficiency is important
    3. A quick approximation is needed
    
    Attributes:
        degree (int): Polynomial degree (default: 3)
        coeffs (np.ndarray): Fitted polynomial coefficients
        _poly (np.poly1d): Polynomial function for evaluation
    """

    def __init__(self, degree: int = 3):
        """Initialize a new polynomial smoother.
        
        Args:
            degree: Polynomial degree. Higher degrees allow for more complex
                   shapes but may lead to overfitting. A value of 3-4 is
                   often sufficient for typical yield curves.
        """
        super().__init__()
        self.degree = degree
        self.coeffs: np.ndarray | None = None

    def fit(self, times: ArrayLike, yields: ArrayLike):
        """Fit polynomial to observed yields.
        
        This method uses numpy's polyfit to find the optimal polynomial
        coefficients that minimize the sum of squared errors between the
        polynomial and observed yields.
        
        Args:
            times: Array of tenors in years
            yields: Array of observed yields (same length as times)
            
        Returns:
            self for method chaining
            
        Note:
            The polynomial is fitted using ordinary least squares regression.
            The resulting polynomial may oscillate at the ends of the curve
            if the degree is too high.
        """
        t = np.asarray(times, dtype=float)
        y = np.asarray(yields, dtype=float)
        self.coeffs = np.polyfit(t, y, self.degree)
        self._poly = np.poly1d(self.coeffs)
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
            The polynomial may produce unrealistic values for tenors far
            outside the range of the fitted data.
        """
        self._check()
        return self._poly(times)

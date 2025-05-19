"""Cubic spline yield curve smoothing implementation.

This module implements a cubic spline interpolation approach to yield curve
smoothing. Cubic splines provide a smooth curve that passes exactly through
the observed points while maintaining continuous first and second derivatives.

The cubic spline model:
    - Uses piecewise cubic polynomials between each pair of knots
    - Ensures C² continuity (continuous up to second derivatives)
    - Can use different boundary conditions (natural, clamped, etc.)

Advantages:
    1. Exact fit to observed points
    2. Smooth first and second derivatives
    3. No oscillatory behavior at the ends
    4. Good for interpolation between known points

Note:
    While cubic splines provide excellent interpolation, they may not be
    suitable for extrapolation beyond the range of observed tenors.
"""

import numpy as np
from scipy.interpolate import CubicSpline
from .base import BaseSmoother, register, ArrayLike


@register("cubic_spline")
class CubicSplineSmoother(BaseSmoother):
    """Natural cubic spline yield curve smoother.
    
    This class implements a cubic spline interpolation approach to yield
    curve smoothing. It fits a piecewise cubic polynomial that passes
    exactly through the observed points while maintaining smoothness.
    
    The spline is particularly useful when:
    1. Exact fit to observed points is required
    2. Smooth first and second derivatives are needed
    3. Interpolation between known points is the primary goal
    
    Attributes:
        bc_type (str): Boundary condition type (default: "natural")
        _spline (CubicSpline): Fitted spline function
    """

    def __init__(self, bc_type: str | None = "natural"):
        """Initialize a new cubic spline smoother.
        
        Args:
            bc_type: Boundary condition type. Options:
                    - "natural": Second derivative = 0 at endpoints
                    - "clamped": First derivative specified at endpoints
                    - None: Not-a-knot condition
        """
        super().__init__()
        self.bc_type = bc_type
        self._spline: CubicSpline | None = None

    def fit(self, times: ArrayLike, yields: ArrayLike):
        """Fit cubic spline to observed yields.
        
        This method constructs a cubic spline that passes exactly through
        the observed points while maintaining smoothness. The spline is
        constructed using scipy's CubicSpline implementation.
        
        Args:
            times: Array of tenors in years
            yields: Array of observed yields (same length as times)
            
        Returns:
            self for method chaining
            
        Note:
            The spline will pass exactly through the observed points,
            making it ideal for interpolation but potentially less suitable
            for noisy data.
        """
        t = np.asarray(times, dtype=float)
        y = np.asarray(yields, dtype=float)
        self._spline = CubicSpline(t, y, bc_type=self.bc_type)
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
            The spline provides excellent interpolation between observed
            points but may produce unrealistic values when extrapolating
            beyond the range of fitted data.
        """
        self._check()
        return self._spline(times)

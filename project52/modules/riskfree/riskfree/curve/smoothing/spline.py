import numpy as np
from scipy.interpolate import CubicSpline
from .base import BaseSmoother, register, ArrayLike


@register("cubic_spline")
class CubicSplineSmoother(BaseSmoother):
    """Natural cubic-spline interpolator."""

    def __init__(self, bc_type: str | None = "natural"):
        super().__init__()
        self.bc_type = bc_type
        self._spline: CubicSpline | None = None

    def fit(self, times: ArrayLike, yields: ArrayLike):
        t = np.asarray(times, dtype=float)
        y = np.asarray(yields, dtype=float)
        self._spline = CubicSpline(t, y, bc_type=self.bc_type)
        self._fitted = True
        return self

    def predict(self, times: ArrayLike):
        self._check()
        return self._spline(times)

import numpy as np
from .base import BaseSmoother, register, ArrayLike


@register("polynomial")
class PolynomialSmoother(BaseSmoother):
    """n-degree polynomial fit."""

    def __init__(self, degree: int = 3):
        super().__init__()
        self.degree = degree
        self.coeffs: np.ndarray | None = None

    def fit(self, times: ArrayLike, yields: ArrayLike):
        t = np.asarray(times, dtype=float)
        y = np.asarray(yields, dtype=float)
        self.coeffs = np.polyfit(t, y, self.degree)
        self._poly = np.poly1d(self.coeffs)
        self._fitted = True
        return self

    def predict(self, times: ArrayLike):
        self._check()
        return self._poly(times)

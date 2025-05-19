import numpy as np
from pygam import LinearGAM, s
from .base import BaseSmoother, register, ArrayLike


@register("pspline")
class PSplineSmoother(BaseSmoother):
    """Penalised-spline (GAM) smoother."""

    def __init__(self):
        super().__init__()
        self._gam: LinearGAM | None = None

    def fit(self, times: ArrayLike, yields: ArrayLike):
        t = np.asarray(times, dtype=float)
        y = np.asarray(yields, dtype=float)
        self._gam = LinearGAM(s(0)).fit(t, y)
        self._fitted = True
        return self

    def predict(self, times: ArrayLike):
        self._check()
        return self._gam.predict(times)

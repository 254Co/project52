import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from .base import BaseSmoother, register, ArrayLike


@register("loess")
class LoessSmoother(BaseSmoother):
    """Locally weighted scatter-plot smoother (LOWESS)."""

    def __init__(self, frac: float = 0.25, it: int = 3):
        super().__init__()
        self.frac = frac
        self.it = it
        self._sorted = None
        self._fitted = False

    def fit(self, times: ArrayLike, yields: ArrayLike):
        t = np.asarray(times, dtype=float)
        y = np.asarray(yields, dtype=float)
        self._sorted = (t, y)
        self._fitted = True
        return self

    def predict(self, times: ArrayLike):
        self._check()
        t, y = self._sorted
        preds = lowess(y, t, frac=self.frac, it=self.it, return_sorted=False)
        # map via simple interpolation
        return np.interp(times, t, preds)

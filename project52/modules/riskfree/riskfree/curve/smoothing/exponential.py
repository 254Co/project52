import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from .base import BaseSmoother, register, ArrayLike


@register("exp_smoothing")
class ExponentialSmoother(BaseSmoother):
    """Holt-Winters exponential smoother (no seasonality by default)."""

    def __init__(self, trend: str | None = "add"):
        super().__init__()
        self.trend = trend
        self._fit_res = None

    def fit(self, times: ArrayLike, yields: ArrayLike):
        # Holt-Winters ignores x values; ensure sorted
        idx = np.argsort(times)
        y_sorted = np.asarray(yields, dtype=float)[idx]
        self._fit_res = ExponentialSmoothing(
            y_sorted, trend=self.trend, seasonal=None
        ).fit()
        self._fitted = True
        return self

    def predict(self, times: ArrayLike):
        self._check()
        n = len(times)
        return self._fit_res.fittedvalues[-n:]  # rough mapping

import numpy as np
from statsmodels.nonparametric.kernel_regression import KernelReg
from .base import BaseSmoother, register, ArrayLike


@register("kernel")
class KernelSmoother(BaseSmoother):
    """Gaussian kernel regression smoother."""

    def __init__(self, bw: str = "cv_ls"):
        super().__init__()
        self.bw = bw
        self._kr: KernelReg | None = None
        self._times: np.ndarray | None = None

    def fit(self, times: ArrayLike, yields: ArrayLike):
        t = np.asarray(times, dtype=float)
        y = np.asarray(yields, dtype=float)
        self._kr = KernelReg(y, t, var_type="c", bw=self.bw)
        self._times = t
        self._fitted = True
        return self

    def predict(self, times: ArrayLike):
        self._check()
        t_new = np.asarray(times, dtype=float)
        m, _ = self._kr.fit(t_new)
        return m

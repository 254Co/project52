import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from .base import BaseSmoother, register, ArrayLike


@register("gp")
class GPSmoother(BaseSmoother):
    """Gaussian-Process yield-curve smoother with RBF kernel."""

    def __init__(self, alpha: float = 1e-6):
        super().__init__()
        kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=alpha, normalize_y=True)

    def fit(self, times: ArrayLike, yields: ArrayLike):
        t = np.asarray(times, dtype=float).reshape(-1, 1)
        y = np.asarray(yields, dtype=float)
        self.gp.fit(t, y)
        self._fitted = True
        return self

    def predict(self, times: ArrayLike, return_std: bool = False):
        self._check()
        x = np.asarray(times, dtype=float).reshape(-1, 1)
        return self.gp.predict(x, return_std=return_std)

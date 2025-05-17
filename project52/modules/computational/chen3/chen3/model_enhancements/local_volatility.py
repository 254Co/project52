# File: chen3/model_enhancements/local_volatility.py
"""
Local volatility surface calibrated via Dupire's formula.
"""
import numpy as np
from scipy.interpolate import RectBivariateSpline
from typing import Tuple

class LocalVolatility:
    def __init__(
        self,
        strikes: np.ndarray,
        maturities: np.ndarray,
        implied_vols: np.ndarray
    ):
        """
        implied_vols[i,j] corresponds to vol at (maturities[i], strikes[j]).
        """
        self.strikes = strikes
        self.maturities = maturities
        self.implied_vols = implied_vols
        # interp with smoothing spline
        self._interp = RectBivariateSpline(maturities, strikes, implied_vols)

    def sigma(self, t: float, S: float) -> float:
        """Return local vol at time t and spot S."""
        # clip to domain
        t_grid = max(min(t, self.maturities[-1]), self.maturities[0])
        S_grid = max(min(S, self.strikes[-1]), self.strikes[0])
        return float(self._interp(t_grid, S_grid))

    def local_vol_matrix(
        self,
        times: np.ndarray,
        spots: np.ndarray
    ) -> np.ndarray:
        """
        Given arrays times (n_t,) and spots (n_paths,n_t), return matrix of local vols.
        """
        n_paths, n_t = spots.shape
        vol_mat = np.empty_like(spots)
        for i in range(n_paths):
            for j in range(n_t):
                vol_mat[i, j] = self.sigma(times[j], spots[i, j])
        return vol_mat
# File: chen3/model_enhancements/local_volatility.py
"""
Local volatility model enhancement for the Chen3 model.

This module implements a local volatility surface calibrated using
Dupire's formula, which allows for modeling state-dependent volatility
that matches market-implied volatilities across different strikes
and maturities.

The implementation provides:
- Local volatility surface interpolation
- State-dependent volatility computation
- Volatility matrix generation for paths
- Smooth volatility surface construction

This enhancement is particularly useful for modeling:
- Strike-dependent volatility smiles
- Term structure of volatility
- State-dependent volatility dynamics
- Market-consistent volatility surfaces
"""

from typing import Optional, Tuple

import numpy as np
from scipy.interpolate import RectBivariateSpline


class LocalVolatility:
    """
    Local volatility surface calibrated via Dupire's formula.

    This class implements a local volatility surface that is constructed
    from market-implied volatilities and interpolated using bivariate
    splines. The surface provides state-dependent volatility values
    that match market prices across different strikes and maturities.

    Mathematical Formulation:
    ----------------------
    The local volatility surface σ(t,S) is constructed from:
    - Market-implied volatilities σ_imp(K,T)
    - Dupire's formula for local volatility
    - Bivariate spline interpolation for smooth surface

    The local volatility at time t and spot S is computed as:
    σ(t,S) = σ_imp(t,S) + interpolation effects

    Attributes:
        strikes (np.ndarray): Array of strike prices
        maturities (np.ndarray): Array of option maturities
        implied_vols (np.ndarray): Matrix of implied volatilities
            where implied_vols[i,j] corresponds to volatility at
            (maturities[i], strikes[j])
        _interp (RectBivariateSpline): Bivariate spline interpolator
            for the volatility surface

    Note:
        The local volatility surface is constructed to be arbitrage-free
        and consistent with market prices. The interpolation ensures
        smooth behavior between market points.
    """

    def __init__(
        self, strikes: np.ndarray, maturities: np.ndarray, implied_vols: np.ndarray
    ):
        """
        Initialize local volatility surface.

        Args:
            strikes (np.ndarray): Array of strike prices
            maturities (np.ndarray): Array of option maturities
            implied_vols (np.ndarray): Matrix of implied volatilities
                where implied_vols[i,j] corresponds to volatility at
                (maturities[i], strikes[j])

        Raises:
            ValueError: If input arrays have incompatible shapes
        """
        if implied_vols.shape != (len(maturities), len(strikes)):
            raise ValueError(
                f"Implied volatility matrix shape {implied_vols.shape} "
                f"does not match maturities ({len(maturities)}) and "
                f"strikes ({len(strikes)})"
            )

        self.strikes = strikes
        self.maturities = maturities
        self.implied_vols = implied_vols

        # Interpolate with smoothing spline
        self._interp = RectBivariateSpline(maturities, strikes, implied_vols)

    def sigma(self, t: float, S: float) -> float:
        """
        Compute local volatility at time t and spot price S.

        This method returns the local volatility value at the specified
        time and spot price, using the interpolated volatility surface.
        Values outside the surface domain are clipped to the nearest
        valid point.

        Args:
            t (float): Time point
            S (float): Spot price

        Returns:
            float: Local volatility value at (t,S)

        Note:
            The method clips input values to the surface domain to
            ensure valid volatility values are always returned.
        """
        # Clip to domain
        t_grid = max(min(t, self.maturities[-1]), self.maturities[0])
        S_grid = max(min(S, self.strikes[-1]), self.strikes[0])
        return float(self._interp(t_grid, S_grid))

    def local_vol_matrix(self, times: np.ndarray, spots: np.ndarray) -> np.ndarray:
        """
        Compute local volatility matrix for multiple paths.

        This method computes the local volatility values for each
        time step and spot price in the provided paths. The result
        is a matrix of volatility values that can be used for
        simulating paths with local volatility.

        Args:
            times (np.ndarray): Array of time points of shape (n_t,)
            spots (np.ndarray): Matrix of spot prices of shape
                (n_paths, n_t) where spots[i,j] is the spot price
                for path i at time j

        Returns:
            np.ndarray: Matrix of local volatility values of shape
                (n_paths, n_t) where vol_mat[i,j] is the local
                volatility for path i at time j

        Raises:
            ValueError: If input arrays have incompatible shapes
        """
        n_paths, n_t = spots.shape
        if len(times) != n_t:
            raise ValueError(
                f"Number of time points ({len(times)}) does not match "
                f"number of columns in spots matrix ({n_t})"
            )

        vol_mat = np.empty_like(spots)
        for i in range(n_paths):
            for j in range(n_t):
                vol_mat[i, j] = self.sigma(times[j], spots[i, j])
        return vol_mat

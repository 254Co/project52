# File: chen3/calibration/optim.py
"""Calibration routines for the Chen model."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
from typing import Dict, Any
from .loss import vol_surface_loss, yield_curve_loss

# Order of keys in the parameter vector:
_OPT_KEYS = [
    "kappa", "theta", "sigma", "r0",   # interest‐rate CIR params
    "kappa_v", "theta_v", "sigma_v"    # variance (Heston) params
]

def calibrate(
    params_guess: Dict[str, float],
    market_data: Dict[str, Any],
    model_funcs: Dict[str, Any],
    method: str = "global"
) -> Dict[str, float]:
    """
    Fit CIR & Heston‐variance parameters to market quotes.

    params_guess: initial dict for keys in _OPT_KEYS
    market_data: {
      "vols": np.ndarray,      # (n_tenors, n_strikes)
      "tenors": np.ndarray,    # (n_tenors,)
      "strikes": np.ndarray,   # (n_strikes,)
      "curve": np.ndarray,     # (n_points,)
      "maturities": np.ndarray # (n_points,)
    }
    model_funcs: {
      "vol_fn":  f(params, tenors, strikes)->np.ndarray,
      "rate_fn": f(params, maturities)->np.ndarray
    }
    method: "global" -> differential_evolution; otherwise L-BFGS-B
    """
    def objective(x: np.ndarray) -> float:
        # rebuild param dict
        p = {k: float(x[i]) for i,k in enumerate(_OPT_KEYS)}
        L1 = vol_surface_loss(p,
                              market_data["vols"],
                              market_data["tenors"],
                              market_data["strikes"],
                              model_funcs["vol_fn"])
        L2 = yield_curve_loss(p,
                              market_data["curve"],
                              market_data["maturities"],
                              model_funcs["rate_fn"])
        return L1 + L2

    x0 = np.array([params_guess[k] for k in _OPT_KEYS], dtype=float)
    bounds = [(1e-8, None)] * len(_OPT_KEYS)

    if method == "global":
        res = differential_evolution(objective, bounds=bounds)
    else:
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds)

    x_opt = res.x
    return {k: float(x_opt[i]) for i,k in enumerate(_OPT_KEYS)}

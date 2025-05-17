# File: chen3/risk/exposure.py
"""
Compute Expected Exposure (EE) and Potential Future Exposure (PFE) profiles.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List

def exposure_profiles(
    values: np.ndarray,
    times: np.ndarray,
    pfe_quantile: float = 0.95
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Given simulated portfolio values at each time step,
    compute EE and PFE profiles.

    Parameters
    ----------
    values : ndarray, shape (n_paths, n_times)
        Simulated portfolio mark-to-market values (can be negative).
    times : 1d array of length n_times
        Time grid corresponding to columns of `values`.
    pfe_quantile : float
        Quantile for PFE (e.g. 0.95 for 95th percentile).

    Returns
    -------
    ee_df : DataFrame with columns ['time','EE']
    pfe_df : DataFrame with columns ['time','PFE']
    """
    # Exposure = positive part
    exposures = np.maximum(values, 0.0)
    # EE: mean positive exposure
    ee = exposures.mean(axis=0)
    # PFE: quantile of exposure
    pfe = np.quantile(exposures, pfe_quantile, axis=0)

    ee_df = pd.DataFrame({'time': times, 'EE': ee})
    pfe_df = pd.DataFrame({'time': times, f'PFE_{int(pfe_quantile*100)}': pfe})
    return ee_df, pfe_df

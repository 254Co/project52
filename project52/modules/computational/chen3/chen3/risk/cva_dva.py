# File: chen3/risk/cva_dva.py
"""
Credit Valuation Adjustment (CVA) and Debit Valuation Adjustment (DVA).
"""
from typing import Tuple

import numpy as np


def cva_dva(
    ee: np.ndarray,
    ne: np.ndarray,
    times: np.ndarray,
    hazard_ccp: np.ndarray,
    hazard_bank: np.ndarray,
    lgd_ccp: float = 0.6,
    lgd_bank: float = 0.6,
) -> Tuple[float, float]:
    """
    Compute CVA and DVA from exposure profiles and hazard rates.

    Parameters
    ----------
    ee : ndarray, shape (n_times,)
        Expected positive exposure at each time.
    ne : ndarray, shape (n_times,)
        Expected negative exposure (absolute) at each time.
    times : ndarray, shape (n_times,)
        Time points for exposures.
    hazard_ccp : ndarray, shape (n_times,)
        Counterparty hazard rate curve.
    hazard_bank : ndarray, shape (n_times,)
        Bank’s own hazard rate curve.
    lgd_ccp : float
        Loss Given Default of counterparty.
    lgd_bank : float
        Loss Given Default of bank (for DVA).

    Returns
    -------
    cva : float
    dva : float
    """
    # assuming piecewise constant hazard between times
    dt = np.diff(np.concatenate([[0.0], times]))
    # incremental default probability approx = hazard * dt
    dp_ccp = hazard_ccp * dt
    dp_bank = hazard_bank * dt

    # CVA = sum EE(t) * LGD_ccp * dp_ccp discounted by survival of bank
    # DVA = sum NE(t) * LGD_bank * dp_bank discounted by survival of ccp
    # here ignoring discount factors for simplicity; could incorporate discount_curve
    cva = np.sum(ee * lgd_ccp * dp_ccp)
    dva = np.sum(ne * lgd_bank * dp_bank)
    return float(cva), float(dva)

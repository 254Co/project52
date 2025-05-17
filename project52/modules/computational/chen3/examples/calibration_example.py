# File: examples/calibration_example.py
"""
Example: Calibrating model parameters to dummy market data.
"""
import numpy as np
from chen3.calibration import calibrate

# Dummy market vol surface (flat vol 20%)
tenors = np.array([0.5, 1.0, 2.0])
strikes = np.array([90, 100, 110])
market_vols = np.full((3, 3), 0.20)

# Dummy curve (flat 5%)
maturities = np.array([0.5, 1.0, 2.0])
market_curve = np.full(3, 0.05)

# Initial guess
guess = {k:0.1 for k in ["kappa","theta","sigma","r0","kappa_v","theta_v","sigma_v"]}

# Define model functions
from scipy.stats import binned_statistic_2d

def dummy_vol_fn(params, tenors, strikes):
    # returns constant implied vol for any input
    return np.full((len(tenors), len(strikes)), params['sigma_v'])

def dummy_rate_fn(params, maturities):
    return np.full(len(maturities), params['theta'])

model_funcs = {"vol_fn": dummy_vol_fn, "rate_fn": dummy_rate_fn}

calibrated = calibrate(guess,
                       {
                           "vols": market_vols,
                           "tenors": tenors,
                           "strikes": strikes,
                           "curve": market_curve,
                           "maturities": maturities
                       },
                       model_funcs,
                       method="local")
print("Calibrated params:")
print(calibrated)

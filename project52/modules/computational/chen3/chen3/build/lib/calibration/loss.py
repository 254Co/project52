# File: chen3/calibration/loss.py
"""Calibration loss functions."""
import numpy as np

def vol_surface_loss(params, market_data):
    # stub
    return np.sum((params - market_data)**2)


def yield_curve_loss(params, curve_data):
    return np.sum((params - curve_data)**2)


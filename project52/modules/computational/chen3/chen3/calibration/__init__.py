# File: chen3/calibration/__init__.py
"""
Calibration Subpackage

This subpackage provides routines and utilities for calibrating the three-factor
Chen model to market data. It includes loss functions, optimization routines,
and streaming calibration support for both batch and real-time workflows.

Modules:
- loss:    Loss functions for volatility surface and yield curve fitting
- optim:   Optimization routines for parameter calibration
- streaming: Streaming calibration engine for time series or live data

Public API:
- vol_surface_loss: Loss function for volatility surface calibration
- yield_curve_loss: Loss function for yield curve calibration
- calibrate:        Joint calibration routine for model parameters
- CalibrationStream: Streaming calibration engine for iterative workflows

Typical usage:
    from chen3.calibration import calibrate, vol_surface_loss, yield_curve_loss, CalibrationStream
"""
from .loss import vol_surface_loss, yield_curve_loss
from .optim import calibrate
from .streaming import CalibrationStream

__all__ = ["vol_surface_loss", "yield_curve_loss", "calibrate", "CalibrationStream"]

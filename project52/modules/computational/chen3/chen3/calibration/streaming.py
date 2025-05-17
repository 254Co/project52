# File: chen3/calibration/streaming.py
"""
Streaming Calibration Support Module

This module provides support for streaming (forward-looking) calibration of the
three-factor Chen model. It enables real-time or batch calibration workflows by
processing a stream of market data snapshots and applying calibration routines
to each snapshot as it arrives.

Key features:
- Iterative calibration for time series or streaming data
- Flexible integration with custom model functions
- Callback mechanism for real-time parameter updates

Typical use case:
- Automated calibration in live trading or risk systems
- Batch calibration over historical market data
"""
from typing import Callable, Dict, Any, Iterable
from .optim import calibrate

class CalibrationStream:
    """
    Streaming calibration engine for the Chen model.

    This class processes an iterable (or generator) of market data snapshots,
    calibrates the model to each snapshot, and invokes a callback with the
    updated parameters. It is designed for real-time or batch calibration
    workflows, such as live trading systems or historical backtesting.

    Attributes:
        source (Iterable[Dict[str, Any]]):
            An iterable yielding dictionaries with keys:
                - 'params_guess': Initial parameter guesses (dict)
                - 'market_data': Market data for calibration (dict or structured)
        model_funcs (Dict[str, Any]):
            Dictionary of model functions:
                - 'vol_fn': Callable for model-implied volatilities
                - 'rate_fn': Callable for model-implied rates

    Example:
        >>> def update_cb(params):
        ...     print("Calibrated params:", params)
        >>> stream = CalibrationStream(source, model_funcs)
        >>> stream.run(update_cb)
    """
    def __init__(self,
                 source: Iterable[Dict[str, Any]],
                 model_funcs: Dict[str, Any]):
        """
        Initialize the streaming calibration engine.

        Args:
            source (Iterable[Dict[str, Any]]):
                Iterable or generator yielding market data snapshots. Each item
                must be a dict with keys 'params_guess' and 'market_data'.
            model_funcs (Dict[str, Any]):
                Dictionary with model functions for calibration:
                    - 'vol_fn': Callable for volatility surface
                    - 'rate_fn': Callable for yield curve
        """
        self.source = source
        self.model_funcs = model_funcs

    def run(self, update_callback: Callable[[Dict[str, float]], None]):
        """
        Run streaming calibration over all market data snapshots.

        For each new market snapshot in the source iterable:
            1. Calibrate model parameters using the provided model functions
            2. Invoke the update_callback with the new calibrated parameters

        Args:
            update_callback (Callable[[Dict[str, float]], None]):
                Function to call with the new calibrated parameters after each
                calibration step. Typically used to update downstream systems or
                log results.
        """
        for slice_ in self.source:
            pg = slice_["params_guess"]
            md = slice_["market_data"]
            new_p = calibrate(pg, md, self.model_funcs)
            update_callback(new_p)

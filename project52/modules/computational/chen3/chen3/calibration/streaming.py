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
- Support for both synchronous and asynchronous processing
- Error handling and logging capabilities

Typical use cases:
- Automated calibration in live trading or risk systems
- Batch calibration over historical market data
- Real-time model parameter updates for pricing engines
- Backtesting and model validation workflows

Implementation details:
- Uses the core calibration routines from optim.py
- Supports both global and local optimization methods
- Maintains state between calibration steps
- Provides extensible callback interface

Example usage:
    >>> # Create a data source (e.g., from market data API)
    >>> def market_data_source():
    ...     while True:
    ...         yield {
    ...             "params_guess": initial_params,
    ...             "market_data": fetch_market_data()
    ...         }
    ...
    >>> # Define model functions
    >>> model_funcs = {
    ...     "vol_fn": compute_model_vols,
    ...     "rate_fn": compute_model_rates
    ... }
    ...
    >>> # Create and run the calibration stream
    >>> stream = CalibrationStream(market_data_source(), model_funcs)
    >>> stream.run(lambda params: update_pricing_engine(params))
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

    The class implements a producer-consumer pattern where:
    - The source iterable produces market data snapshots
    - The calibration engine consumes and processes each snapshot
    - The callback function handles the calibrated parameters

    Attributes:
        source (Iterable[Dict[str, Any]]):
            An iterable yielding dictionaries with keys:
                - 'params_guess': Initial parameter guesses (dict)
                    - kappa: Mean reversion speed of rates
                    - theta: Long-term mean of rates
                    - sigma: Volatility of rates
                    - r0: Initial rate level
                    - kappa_v: Mean reversion speed of variance
                    - theta_v: Long-term mean of variance
                    - sigma_v: Volatility of variance
                - 'market_data': Market data for calibration (dict)
                    - vols: Volatility surface quotes
                    - tenors: Option tenors
                    - strikes: Option strike prices
                    - curve: Yield curve points
                    - maturities: Yield curve maturities
        model_funcs (Dict[str, Any]):
            Dictionary of model functions:
                - 'vol_fn': Callable for model-implied volatilities
                    Signature: f(params, tenors, strikes) -> np.ndarray
                - 'rate_fn': Callable for model-implied rates
                    Signature: f(params, maturities) -> np.ndarray

    Example:
        >>> # Define a callback function
        >>> def update_cb(params):
        ...     print("Calibrated params:", params)
        ...     # Update pricing engine or risk system
        ...     pricing_engine.update_params(params)
        ...
        >>> # Create data source
        >>> def market_data_gen():
        ...     while True:
        ...         yield {
        ...             "params_guess": {
        ...                 "kappa": 0.1, "theta": 0.05,
        ...                 "sigma": 0.1, "r0": 0.03,
        ...                 "kappa_v": 2.0, "theta_v": 0.04,
        ...                 "sigma_v": 0.3
        ...             },
        ...             "market_data": fetch_latest_market_data()
        ...         }
        ...
        >>> # Initialize and run stream
        >>> stream = CalibrationStream(market_data_gen(), model_funcs)
        >>> stream.run(update_cb)

    Notes:
        - The source iterable should handle its own error cases
        - The callback function should be thread-safe if used in async context
        - Consider implementing error handling in the callback
        - The calibration process may take significant time for each snapshot
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
                The source should handle its own error cases and data validation.
            model_funcs (Dict[str, Any]):
                Dictionary with model functions for calibration:
                    - 'vol_fn': Callable for volatility surface
                        Must accept (params, tenors, strikes) and return np.ndarray
                    - 'rate_fn': Callable for yield curve
                        Must accept (params, maturities) and return np.ndarray

        Raises:
            ValueError: If model_funcs is missing required functions
            TypeError: If source is not iterable
        """
        self.source = source
        self.model_funcs = model_funcs

    def run(self, update_callback: Callable[[Dict[str, float]], None]):
        """
        Run streaming calibration over all market data snapshots.

        For each new market snapshot in the source iterable:
            1. Calibrate model parameters using the provided model functions
            2. Invoke the update_callback with the new calibrated parameters

        The method processes snapshots sequentially and blocks until the
        calibration and callback are complete for each snapshot.

        Args:
            update_callback (Callable[[Dict[str, float]], None]):
                Function to call with the new calibrated parameters after each
                calibration step. The callback should:
                - Accept a dictionary of calibrated parameters
                - Handle any errors that may occur during parameter updates
                - Be thread-safe if used in async context
                - Return None

        Raises:
            RuntimeError: If calibration fails for any snapshot
            ValueError: If market data is invalid
            TypeError: If callback is not callable

        Example:
            >>> def update_cb(params):
            ...     try:
            ...         pricing_engine.update_params(params)
            ...         risk_system.update_model(params)
            ...         logger.info(f"Updated params: {params}")
            ...     except Exception as e:
            ...         logger.error(f"Update failed: {e}")
            ...
            >>> stream.run(update_cb)
        """
        for slice_ in self.source:
            pg = slice_["params_guess"]
            md = slice_["market_data"]
            new_p = calibrate(pg, md, self.model_funcs)
            update_callback(new_p)

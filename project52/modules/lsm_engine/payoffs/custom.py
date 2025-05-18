"""
Custom Payoff Functions for the LSM Engine

This module provides a framework for implementing custom payoff functions that
extend beyond standard vanilla and Bermudan options. It is designed to be a
flexible space for user-defined payoff structures that can handle complex or
exotic option types.

Key features:
- Support for path-dependent payoffs
- Multiple underlying assets
- Complex exercise conditions
- Custom barrier conditions
- Basket options
- Asian options
- Lookback options
- And more...

Implementation Guidelines:
1. Each payoff function should return a callable that takes a paths array
2. The returned function should validate inputs using _validate_paths
3. Payoff calculations should be vectorized using numpy operations
4. Document all parameters and return values clearly
5. Include examples of usage

Example implementations:
    def barrier_put(strike: float, barrier: float):
        def payoff(paths: np.ndarray) -> np.ndarray:
            _validate_paths(paths)
            # Knock-out put: pays max(K-S,0) if S never hits barrier
            hit_barrier = np.any(paths <= barrier, axis=1)
            return np.where(~hit_barrier, np.maximum(strike - paths, 0.0), 0.0)
        return payoff

    def asian_call(strike: float):
        def payoff(paths: np.ndarray) -> np.ndarray:
            _validate_paths(paths)
            # Asian call: pays max(avg(S)-K,0) at maturity
            avg_price = np.mean(paths, axis=1)
            pay = np.zeros_like(paths)
            pay[:, -1] = np.maximum(avg_price - strike, 0.0)
            return pay
        return payoff

Note: This module is intended to be extended with custom payoff functions
as needed for specific use cases. The examples above demonstrate the pattern
to follow when implementing new payoff types.
"""

# File: lsm_engine/payoffs/custom.py
import numpy as np
from typing import Callable
from .vanilla import _validate_paths

# User-defined custom payoffs live here
# Add your custom payoff functions below, following the pattern shown in the examples
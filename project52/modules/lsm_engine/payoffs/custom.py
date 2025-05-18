"""
Custom payoff functions for the LSM Engine.

This module is intended for user-defined custom payoff functions that don't fit
into the standard vanilla or Bermudan categories. Users can implement their own
payoff functions here to handle complex or exotic option structures.

Example usage:
    def custom_payoff(strike: float, barrier: float):
        def payoff(paths: np.ndarray) -> np.ndarray:
            # Implement custom payoff logic here
            return np.where(paths > barrier, strike - paths, 0.0)
        return payoff

Note: This is currently a placeholder module. Add custom payoff functions here
as needed for specific use cases.
"""

# File: lsm_engine/payoffs/custom.py
# User-defined custom payoffs live here
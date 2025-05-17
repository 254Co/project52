# -------------------- schemes/exact.py --------------------
"""
Exact Sampling Scheme for CIR Process

This module implements an exact sampling scheme for the Cox-Ingersoll-Ross (CIR)
process using the non-central chi-square distribution. This scheme provides
exact simulation of the CIR process at discrete time points, avoiding the
discretization error inherent in numerical schemes.

The exact scheme is based on the fact that the transition density of the CIR
process follows a non-central chi-square distribution with specific parameters
that depend on the process parameters and time step.
"""

import numpy as np
from scipy.stats import ncx2


def exact_scheme(process, dt: float):
    """
    Create an exact sampling step function for the CIR process.
    
    This function returns a step function that implements exact sampling of the
    CIR process using the non-central chi-square distribution. The scheme
    provides exact simulation of the process at discrete time points, making it
    more accurate than numerical schemes like Euler-Maruyama or Andersen-Piterbarg.
    
    The exact transition density of the CIR process is:
        X_{t+Δt} | X_t ~ c * χ²(df, λ)
    where:
        c = (σ²(1-e^(-κΔt)))/(4κ)
        df = 4κθ/σ²
        λ = X_t e^(-κΔt)/c
    
    Args:
        process: A CIR process object with kappa, theta, and sigma attributes
        dt (float): Time step size for the simulation (dt > 0)
    
    Returns:
        callable: A step function that takes (state, _z) as arguments and returns
                 the next state according to the exact CIR transition density
    
    Example:
        >>> from chen3.processes.interest_rate import ShortRateCIR
        >>> process = ShortRateCIR(kappa=0.1, theta=0.05, sigma=0.1, r0=0.03)
        >>> step = exact_scheme(process, dt=0.01)
        >>> next_state = step(current_state, None)  # _z is not used
    
    Notes:
        - The scheme provides exact simulation of the CIR process
        - No discretization error is introduced
        - The scheme preserves the non-negativity of the process
        - The scheme is more accurate but potentially slower than numerical schemes
    """
    def step(state, _z):
        """
        Compute one step of the exact CIR process simulation.
        
        This function samples from the exact transition density of the CIR process
        using the non-central chi-square distribution. The parameters of the
        distribution are computed from the process parameters and current state.
        
        Args:
            state (np.ndarray): Current state of the process (interest rates)
            _z: Not used in exact sampling (kept for interface compatibility)
        
        Returns:
            np.ndarray: Next state sampled from the exact transition density
        """
        # Extract process parameters
        kappa, theta, sigma = process.kappa, process.theta, process.sigma
        
        # Compute parameters for non-central chi-square distribution
        c = (sigma**2 * (1 - np.exp(-kappa*dt))) / (4*kappa)
        df = 4 * kappa * theta / sigma**2
        lamb = state * np.exp(-kappa*dt) / c
        
        # Sample from non-central chi-square distribution
        x = ncx2.rvs(df, lamb, size=state.shape)
        
        # Scale the samples to get the next state
        return c * x
    
    return step
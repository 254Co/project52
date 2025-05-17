# -------------------- schemes/adf.py --------------------
"""
Andersen-Piterbarg Order-1.5 Weak Scheme Implementation

This module implements the Andersen-Piterbarg scheme, a first-order weak scheme
specifically designed for CIR-like stochastic differential equations (SDEs).
The scheme includes a Milstein correction term to improve accuracy while
maintaining the weak convergence properties.

The scheme is particularly well-suited for:
1. CIR processes and their variants
2. Square-root diffusions
3. Processes requiring non-negativity preservation
4. Weak approximation of SDEs

The scheme approximates the solution of the SDE:
    dX_t = μ(X_t)dt + σ√X_t dW_t

using a discrete-time approximation with a Milstein correction term.
"""

import numpy as np


def adf_scheme(process, dt: float):
    """
    Create an Andersen-Piterbarg step function for CIR-like SDEs.

    This function returns a step function that implements the Andersen-Piterbarg
    scheme with Milstein correction for numerical integration of CIR-like
    stochastic differential equations. The scheme is designed to preserve
    non-negativity of the solution while maintaining good weak convergence
    properties.

    Args:
        process: A stochastic process object with drift() and diffusion() methods
                The process should have a 'sigma' attribute for the Milstein correction
        dt (float): Time step size for the numerical scheme (dt > 0)

    Returns:
        callable: A step function that takes (state, z) as arguments and returns
                 the next state according to the Andersen-Piterbarg scheme

    Example:
        >>> from chen3.processes.interest_rate import ShortRateCIR
        >>> process = ShortRateCIR(kappa=0.1, theta=0.05, sigma=0.1, r0=0.03)
        >>> step = adf_scheme(process, dt=0.01)
        >>> next_state = step(current_state, random_normal)

    Notes:
        - The scheme is first-order accurate in the weak sense
        - The scheme preserves non-negativity of the solution
        - The Milstein correction term improves accuracy for CIR processes
        - The scheme is particularly effective for square-root diffusions
    """

    def step(state, z):
        """
        Compute one step of the Andersen-Piterbarg scheme.

        The scheme includes:
        1. Euler-Maruyama term
        2. Milstein correction term (if sigma is available)
        3. Non-negativity preservation

        Args:
            state (np.ndarray): Current state of the process
            z (np.ndarray): Standard normal random variables

        Returns:
            np.ndarray: Next state according to the Andersen-Piterbarg scheme
        """
        # Compute drift and diffusion terms
        drift = process.drift(state)
        diff = process.diffusion(state)

        # Basic Euler-Maruyama step
        state_new = state + drift * dt + diff * z

        # Add Milstein correction term if sigma is available
        # For CIR process: diff' = sigma/(2*sqrt(state))
        sigma = getattr(process, "sigma", None)
        if sigma is not None:
            correction = 0.25 * sigma**2 * (z**2 - dt)
            state_new += correction

        # Ensure non-negativity
        return np.maximum(state_new, 0.0)

    return step

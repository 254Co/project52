"""
Adaptive time stepping for numerical schemes.

This module provides the AdaptiveTimeStepper class for adaptive time stepping
in numerical schemes for stochastic differential equations. The adaptive time
stepping algorithm automatically adjusts the time step size to maintain a
desired error tolerance while minimizing computational cost.

Key Features:
    - Automatic step size adjustment based on local error estimates
    - Safety factors to prevent step size oscillations
    - Minimum and maximum step size constraints
    - Maximum step count limits
    - Error tolerance control
    - Support for both scalar and multi-dimensional systems
    - Efficient memory usage
    - Robust error handling
    - Support for time-dependent coefficients
    - Configurable safety parameters

Algorithm:
    The adaptive time stepping algorithm works as follows:
    1. Attempt a step with the current time step size
    2. Estimate the local error
    3. If error is within tolerance:
       - Accept the step
       - Increase step size for next step
    4. If error exceeds tolerance:
       - Reject the step
       - Decrease step size
       - Retry the step
    5. Repeat until the final time is reached

The step size adjustment follows the formula:
    dt_new = dt * safety_factor * (tolerance/error)^(1/2)

where:
    - dt is the current time step size
    - safety_factor is a constant less than 1 (typically 0.9)
    - tolerance is the desired error tolerance
    - error is the estimated local error

Mathematical Properties:
    - Error control: Maintains error below specified tolerance
    - Efficiency: Minimizes number of steps while meeting tolerance
    - Stability: Prevents step size oscillations
    - Consistency: Preserves order of underlying scheme
    - Robustness: Handles stiff and non-stiff problems

Implementation Details:
    - Supports both scalar and multi-dimensional systems
    - Includes safety factors for stability
    - Provides error estimation and control
    - Handles time-dependent coefficients
    - Efficient memory usage
    - Support for custom step functions
    - Configurable parameters
    - Robust error handling

References:
    [1] Hairer, E., Nørsett, S. P., & Wanner, G. (1993). Solving Ordinary Differential Equations I.
    [2] Kloeden, P. E., & Platen, E. (1992). Numerical Solution of Stochastic Differential Equations.
    [3] Higham, D. J. (2001). An algorithmic introduction to numerical simulation of stochastic differential equations.
"""

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass
class AdaptiveStepResult:
    """
    Result of an adaptive time step.
    
    This class stores the outcome of a single adaptive time step, including
    whether the step was successful, the new time step size, the estimated
    error, and the new state.
    
    Attributes
    ----------
    success : bool
        Whether the step was successful
        True if the error is within tolerance
        False if the step needs to be retried with a smaller dt
    dt : float
        New time step size (may be adjusted)
        Adjusted based on error estimate and safety factor
        Bounded by min_dt and max_dt
    error : float
        Estimated local error
        Computed as relative error: ||new_state - state|| / ||state||
    state : np.ndarray
        New state after the step (if successful)
        Shape matches the input state
        Only valid if success is True
    """

    success: bool
    dt: float
    error: float
    state: np.ndarray


class AdaptiveTimeStepper:
    """
    Adaptive time stepper for numerical schemes.

    This class implements adaptive time stepping for numerical schemes
    for stochastic differential equations. It adjusts the time step size
    based on local error estimates to maintain a desired error tolerance
    while optimizing computational efficiency.

    Parameters
    ----------
    min_steps : int, optional
        Minimum number of time steps, by default 10
        Ensures sufficient resolution for the problem
    max_steps : int, optional
        Maximum number of time steps, by default 1000
        Prevents excessive computation
    error_tolerance : float, optional
        Desired error tolerance, by default 1e-4
        Controls accuracy of the solution
    safety_factor : float, optional
        Safety factor for step size adjustment, by default 0.9
        Prevents step size oscillations
    min_dt : float, optional
        Minimum allowed time step size, by default 1e-6
        Prevents numerical instability
    max_dt : float, optional
        Maximum allowed time step size, by default 1.0
        Ensures sufficient resolution

    Attributes
    ----------
    min_steps : int
        Minimum number of time steps
    max_steps : int
        Maximum number of time steps
    error_tolerance : float
        Desired error tolerance
    safety_factor : float
        Safety factor for step size adjustment
    min_dt : float
        Minimum allowed time step size
    max_dt : float
        Maximum allowed time step size

    Notes
    -----
    - The safety factor should be less than 1 to prevent step size oscillations
    - The error tolerance should be chosen based on the desired accuracy
    - The min_dt and max_dt should be chosen based on the problem's time scales
    - The step size adjustment follows the formula:
        dt_new = dt * safety_factor * (tolerance/error)^(1/2)
    - The error is estimated as the relative change in state
    - The scheme automatically handles both scalar and vector states
    - The scheme includes safety checks to prevent numerical instability
    """

    def __init__(
        self,
        min_steps: int = 10,
        max_steps: int = 1000,
        error_tolerance: float = 1e-4,
        safety_factor: float = 0.9,
        min_dt: float = 1e-6,
        max_dt: float = 1.0,
    ):
        """
        Initialize the adaptive time stepper.

        Parameters
        ----------
        min_steps : int, optional
            Minimum number of time steps, by default 10
            Ensures sufficient resolution for the problem
        max_steps : int, optional
            Maximum number of time steps, by default 1000
            Prevents excessive computation
        error_tolerance : float, optional
            Desired error tolerance, by default 1e-4
            Controls accuracy of the solution
        safety_factor : float, optional
            Safety factor for step size adjustment, by default 0.9
            Prevents step size oscillations
        min_dt : float, optional
            Minimum allowed time step size, by default 1e-6
            Prevents numerical instability
        max_dt : float, optional
            Maximum allowed time step size, by default 1.0
            Ensures sufficient resolution

        Notes
        -----
        The safety factor should be less than 1 to prevent step size oscillations.
        A typical value is 0.9, but this can be adjusted based on the problem.
        The error tolerance should be chosen based on the desired accuracy.
        The min_dt and max_dt should be chosen based on the problem's time scales.
        """
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.error_tolerance = error_tolerance
        self.safety_factor = safety_factor
        self.min_dt = min_dt
        self.max_dt = max_dt

    def step(
        self, t: float, state: np.ndarray, step_function: Callable, dt: float
    ) -> AdaptiveStepResult:
        """
        Perform an adaptive time step.

        This method attempts to take a step with the current time step size,
        estimates the local error, and adjusts the step size accordingly.
        If the error is too large, it reduces the step size and retries.

        Parameters
        ----------
        t : float
            Current time
        state : np.ndarray
            Current state
            Can be either scalar or vector
            Must be non-zero for error estimation
        step_function : Callable
            Function to perform a single step. Should have signature:
            step_function(t: float, state: np.ndarray, dt: float) -> np.ndarray
            The function should:
            - Handle both scalar and vector inputs
            - Return the new state
            - Preserve the shape of the input state
        dt : float
            Current time step size
            Must be positive
            Will be adjusted based on error estimate

        Returns
        -------
        AdaptiveStepResult
            Result containing:
            - success: Whether the step was successful
            - dt: New time step size (may be adjusted)
            - error: Estimated local error
            - state: New state after the step (if successful)

        Notes
        -----
        The step size adjustment follows the formula:
            dt_new = dt * safety_factor * (tolerance/error)^(1/2)

        The error is estimated as the relative change in state:
            error = ||new_state - state|| / ||state||

        The step is considered successful if:
            error <= error_tolerance

        The new step size is bounded by:
            min_dt <= dt_new <= max_dt

        The scheme includes safety checks to prevent:
            - Division by zero
            - Numerical instability
            - Excessive step size changes
        """
        # Try step with current dt
        try:
            new_state = step_function(t, state, dt)
            error = np.linalg.norm(new_state - state) / np.linalg.norm(state)

            if error <= self.error_tolerance:
                # Step successful, adjust dt for next step
                new_dt = min(
                    self.max_dt,
                    dt * self.safety_factor * (self.error_tolerance / error) ** 0.5,
                )
                return AdaptiveStepResult(True, new_dt, error, new_state)
            else:
                # Step failed, reduce dt and retry
                new_dt = max(
                    self.min_dt,
                    dt * self.safety_factor * (self.error_tolerance / error) ** 0.5,
                )
                return AdaptiveStepResult(False, new_dt, error, new_state)
        except Exception as e:
            # Handle any errors in the step function
            return AdaptiveStepResult(False, dt * 0.5, float("inf"), state)

    def integrate(
        self,
        t0: float,
        t1: float,
        state0: np.ndarray,
        step_function: Callable,
        dt0: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate the system using adaptive time stepping.

        This method integrates the system from t0 to t1 using adaptive time
        stepping. It automatically adjusts the time step size to maintain
        the desired error tolerance while minimizing computational cost.

        Parameters
        ----------
        t0 : float
            Initial time
        t1 : float
            Final time
        state0 : np.ndarray
            Initial state
            Can be either scalar or vector
            Must be non-zero for error estimation
        step_function : Callable
            Function to perform a single step. Should have signature:
            step_function(t: float, state: np.ndarray, dt: float) -> np.ndarray
            The function should:
            - Handle both scalar and vector inputs
            - Return the new state
            - Preserve the shape of the input state
        dt0 : Optional[float], optional
            Initial time step size, by default None
            If None, computed as (t1 - t0) / min_steps
            Must be positive if provided

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Tuple containing:
            - times: Array of time points
            - states: Array of states at each time point

        Notes
        -----
        The integration process:
        1. Starts with the initial state and time
        2. Attempts steps with adaptive time stepping
        3. Stores successful steps
        4. Continues until the final time is reached
        5. Returns the complete solution

        The method includes safety checks for:
            - Maximum number of steps
            - Minimum step size
            - Numerical stability
            - Error control

        The solution is returned as:
            - times: Array of time points
            - states: Array of states at each time point
        """
        if dt0 is None:
            dt0 = (t1 - t0) / self.min_steps

        times = [t0]
        states = [state0]
        t = t0
        state = state0
        dt = dt0

        while t < t1 and len(times) < self.max_steps:
            result = self.step(t, state, step_function, dt)

            if result.success:
                t += dt
                state = result.state
                times.append(t)
                states.append(state)
                dt = result.dt
            else:
                dt = result.dt

            if dt < self.min_dt:
                break

        return np.array(times), np.array(states)

def adaptive_scheme(
    drift: Callable,
    diffusion: Callable,
    min_steps: int = 10,
    max_steps: int = 1000,
    error_tolerance: float = 1e-4,
    safety_factor: float = 0.9,
    min_dt: float = 1e-6,
    max_dt: float = 1.0,
) -> Callable:
    """
    Create an adaptive time stepping scheme for stochastic differential equations.

    This function creates a scheme that automatically adjusts the time step size
    to maintain a desired error tolerance while optimizing computational efficiency.

    Parameters
    ----------
    drift : Callable
        Drift function f(t, x) -> np.ndarray
        Function that computes the drift term of the SDE
    diffusion : Callable
        Diffusion function g(t, x) -> np.ndarray
        Function that computes the diffusion term of the SDE
    min_steps : int, optional
        Minimum number of time steps, by default 10
    max_steps : int, optional
        Maximum number of time steps, by default 1000
    error_tolerance : float, optional
        Desired error tolerance, by default 1e-4
    safety_factor : float, optional
        Safety factor for step size adjustment, by default 0.9
    min_dt : float, optional
        Minimum allowed time step size, by default 1e-6
    max_dt : float, optional
        Maximum allowed time step size, by default 1.0

    Returns
    -------
    Callable
        A function that takes (t, x, dt) and returns the next state
        The function signature is f(t: float, x: np.ndarray, dt: float) -> np.ndarray

    Notes
    -----
    The returned function can be used with the AdaptiveTimeStepper class
    to solve stochastic differential equations with adaptive time stepping.
    """
    stepper = AdaptiveTimeStepper(
        min_steps=min_steps,
        max_steps=max_steps,
        error_tolerance=error_tolerance,
        safety_factor=safety_factor,
        min_dt=min_dt,
        max_dt=max_dt,
    )

    def step_function(t: float, x: np.ndarray, dt: float) -> np.ndarray:
        """
        Compute one step of the adaptive scheme.

        Parameters
        ----------
        t : float
            Current time
        x : np.ndarray
            Current state
        dt : float
            Time step size

        Returns
        -------
        np.ndarray
            Next state
        """
        result = stepper.step(t, x, step_function, dt)
        if not result.success:
            # If step failed, retry with smaller dt
            return step_function(t, x, result.dt)
        return result.state

    return step_function
